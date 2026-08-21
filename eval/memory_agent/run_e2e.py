r"""Memory Agent 端到端评测（评测方案 §4）。

把 LongMemEval 的 haystack 对话：
  A 组：喂给 v22 memory agent 提取 → 写入 Qdrant → 召回 → 回答
  B 组：原文直接当 memory 写入 Qdrant（不过 memory agent）→ 召回 → 回答
然后写 {question_id, hypothesis, ...} 到 hyp.jsonl，后续用 eval.longmemeval.judge / metrics 打分。

与 C 组（旧 2.1 SQLite 关键词链路）的对照直接复用 eval/longmemeval/run.py 的 outputs，不在此重跑。

前置：Qdrant 起在 127.0.0.1:6333、conda Catenv、OPENAI_API_KEY/BASE_URL 指向智谱、CT_MEMORY_V22_ENABLED=true。

用法（在项目根 D:/Project/CatTogether 下）：
  $py = "C:\Users\20772\miniconda3\envs\Catenv\python.exe"
  & $py -m eval.memory_agent.run_e2e --data eval/longmemeval/data/longmemeval_oracle.json --group A --limit 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# --- path bootstrap: project root + backend on sys.path ----------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]            # eval/memory_agent -> eval -> CatTogether
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("CT_MEMORY_V22_ENABLED", "true")

# backend imports
from config import settings  # noqa: E402
from bootstrap import register_builtins  # noqa: E402
from memory.v22 import get_memory_store, get_memory_extractor, reset_for_tests  # noqa: E402
from memory.v22.models import Memory  # noqa: E402
from memory.v22.render import MemoryRenderer  # noqa: E402
from memory.v22.extractor import ExtractionContext  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# per-question Qdrant collection prefix; we drop+recreate per question for a clean slate
EVAL_COLLECTION_PREFIX = "eval_mem"

EVAL_USER_ID = "feishu:eval:u1"
EVAL_PROJECT_ID = "eval-project"
EVAL_CHANNEL = "eval-channel"
EVAL_THREAD = "eval-thread"


# ---------------------------------------------------------------------------
# Qdrant clean slate
# ---------------------------------------------------------------------------

def _qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def reset_collection(collection: str) -> None:
    """Clear all points in the collection so the next question starts empty.

    We do NOT delete_collection + recreate — Qdrant sometimes leaves on-disk
    collection data after delete ('Collection data already exists at
    ./storage/...' 400 error on recreate), which breaks the whole run.
    Instead we delete all points in-place: the collection + its vector config
    (dense + bm25 sparse slots) persist, only the points are wiped. Cheaper too.
    """
    client = _qdrant_client()
    try:
        if client.collection_exists(collection):
            client.delete(collection_name=collection, points_selector="*")
    except Exception:
        # collection may not exist yet on the first question — that's fine,
        # get_memory_store().ensure_collection() will create it.
        pass


def _scroll_all_memories(store) -> list:
    """Scroll the whole collection — the 'extract-all, no truncation' recall mode.

    Returns MemorySearchResult-like objects (with .memory) so the renderer works.
    Used by group A with recall_mode=all: every extracted memory is fed to the
    reader, no relevance ranking, no top_k truncation.
    """
    from qdrant_client.models import ScrollFilter
    from memory.v22.models import Memory, MemorySearchResult
    client = store._client
    col = store._collection
    out: list[MemorySearchResult] = []
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=col,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in pts:
            out.append(MemorySearchResult(memory=Memory.from_payload(str(p.id), p.payload), score=0.0, signals=[]))
        if not offset:
            break
    return out


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

async def ingest_via_extractor(question: dict, store, extractor, concurrency: int = 8) -> dict:
    """A 组：每个 haystack session 喂给 memory agent 提取，写入 Qdrant。

    喂法（v2）：把整个 session 的所有轮次拼成完整对话文本，作为 user_text 喂给
    memory agent——而不是只取末两轮。这样 memory agent 能看到 session 里全部事实，
    避免漏提取（早期轮次的事实不会被"只看末两轮"丢掉）。

    并发：一个题内的多个 session 并发提取（semaphore 限流到 concurrency），
    提取是 IO 密集（memory agent 是 Claude CLI 子进程），并发能把每题墙钟时间
    从 ~47×10s 压到 ~47/8×10s。

    返回 (stats, extracted_memories)：extracted_memories 是本题提取出的全部
    Memory 列表，供 A2/A8 共用缓存——同一题的提取只做一次，A2（全量喂 reader）
    和 A8（top_k 召回）复用同一批记忆，避免重复提取。
    """
    from memory.db import memory_db
    from memory.history_store import HistoryStore

    history = HistoryStore(memory_db)
    try:
        history.clear_thread(tenant_id="default", channel_id=EVAL_CHANNEL, thread_id=EVAL_THREAD)
    except Exception:
        pass

    # 1. 先把所有 session 的 turn 灌进 session_messages，并构造每个 session 的完整转录
    sessions = question.get("haystack_sessions", []) or []
    transcripts: list[str] = []
    n_turns = 0
    for sess in sessions:
        full_transcript: list[str] = []
        for turn in sess:
            role = turn.get("role", "user")
            content = turn.get("content", "") or ""
            if not content:
                continue
            n_turns += 1
            try:
                history.append_message(
                    tenant_id="default",
                    channel_id=EVAL_CHANNEL,
                    thread_id=EVAL_THREAD,
                    role=role,
                    content=content,
                    agent_id="eval-sut" if role != "user" else None,
                )
            except Exception:
                pass
            full_transcript.append(f"[{role}] {content}")
        if full_transcript:
            transcripts.append("\n".join(full_transcript))

    # 2. 并发触发提取：每个 session 一个 ExtractionContext，semaphore 限流
    sem = asyncio.Semaphore(concurrency)
    counters = {"extracted": 0, "dup": 0, "failed": 0}
    written_mems: list[Memory] = []

    async def _extract_one(transcript: str) -> None:
        ctx = ExtractionContext(
            event_type="agent_invocation",
            agent_id="eval-sut",
            agent_name="eval-sut",
            user_text=transcript,
            final_text="",
            mutation_paths=[],
            succeeded=True,
            run_id=question.get("question_id"),
            channel_id=EVAL_CHANNEL,
            thread_id=EVAL_THREAD,
            project_id=EVAL_PROJECT_ID,
            user_id=EVAL_USER_ID,
        )
        async with sem:
            res = await extractor.extract(ctx)
        counters["extracted"] += len(res.written)
        counters["dup"] += res.skipped_dup
        if res.failed:
            counters["failed"] += 1
        written_mems.extend(res.written)

    await asyncio.gather(*(_extract_one(t) for t in transcripts))
    stats = {
        "n_haystack_turns": n_turns,
        "n_extracted": counters["extracted"],
        "n_skipped_dup": counters["dup"],
        "n_extract_failed": counters["failed"],
    }
    return stats, written_mems


def ingest_raw(question: dict, store) -> dict:
    """B 组：每个 haystack turn 原文直接当一条 memory 写入 Qdrant，不过 memory agent。

    domain：user turn → user，assistant turn → project（与 A 组 domain 推导口径一致）。
    同样走 store.upsert（dense+bm25），保证 A vs B 只差「提取 vs 不提取」。
    """
    n1 = 0
    n_dup = 0
    sessions = question.get("haystack_sessions", []) or []
    for sess in sessions:
        for turn in sess:
            role = turn.get("role", "user")
            content = (turn.get("content", "") or "").strip()
            if not content:
                continue
            text = f"[{role}] {content}"
            domain = "user" if role == "user" else "project"
            attributed = "user" if role == "user" else "assistant"
            try:
                if store.exists_by_hash(domain, text):
                    n_dup += 1
                    continue
            except Exception:
                pass
            mem = Memory.create(
                text=text,
                domain=domain,  # type: ignore[arg-type]
                attributed_to=attributed,  # type: ignore[arg-type]
                run_id=question.get("question_id"),
                channel_id=EVAL_CHANNEL,
                thread_id=EVAL_THREAD,
                agent_id="eval-raw",
                event_type="raw_ingest",
            )
            try:
                store.upsert(mem)
                n1 += 1
            except Exception:
                pass
    return {"n_raw_written": n1, "n_skipped_dup": n_dup}


# ---------------------------------------------------------------------------
# Token accounting (run_e2e-level, per the eval design — NOT on ExtractionResult)
# ---------------------------------------------------------------------------

class TokenCounters:
    """Per-question token usage across the memory system's three cost centers.

    #1 extraction (memory agent → glm-5.2): prompt/completion/total
    #2 answering (answer model → glm-5.2): prompt/completion/total
    #4+#5 embedding (local BGE, equivalent tokens): bge + uniform tokenizer,
       split write (upsert memory.text) vs read (search query)
    """

    def __init__(self) -> None:
        self.extract_prompt = 0
        self.extract_completion = 0
        self.extract_total = 0
        self.answer_prompt = 0
        self.answer_completion = 0
        self.answer_total = 0
        self.embed_bge_write = 0
        self.embed_bge_read = 0
        self.embed_uniform_write = 0
        self.embed_uniform_read = 0

    def add_extract_usage(self, usage: dict | None) -> None:
        if not usage:
            return
        self.extract_prompt += int(usage.get("prompt_tokens") or 0)
        self.extract_completion += int(usage.get("completion_tokens") or 0)
        self.extract_total += int(usage.get("total_tokens") or 0)

    def add_answer_usage(self, usage: dict | None) -> None:
        if not usage:
            return
        self.answer_prompt += int(usage.get("prompt_tokens") or 0)
        self.answer_completion += int(usage.get("completion_tokens") or 0)
        self.answer_total += int(usage.get("total_tokens") or 0)

    def add_embed_stats(self, stats: dict) -> None:
        self.embed_bge_write += int(stats.get("bge_write") or 0)
        self.embed_bge_read += int(stats.get("bge_read") or 0)
        self.embed_uniform_write += int(stats.get("uniform_write") or 0)
        self.embed_uniform_read += int(stats.get("uniform_read") or 0)

    def to_dict(self) -> dict:
        return {
            "extract_prompt_tokens": self.extract_prompt,
            "extract_completion_tokens": self.extract_completion,
            "extract_total_tokens": self.extract_total,
            "answer_prompt_tokens": self.answer_prompt,
            "answer_completion_tokens": self.answer_completion,
            "answer_total_tokens": self.answer_total,
            "embed_bge_write": self.embed_bge_write,
            "embed_bge_read": self.embed_bge_read,
            "embed_uniform_write": self.embed_uniform_write,
            "embed_uniform_read": self.embed_uniform_read,
        }


# ---------------------------------------------------------------------------
# Recall + answer
# ---------------------------------------------------------------------------

async def answer_question(question: dict, store, answer_model: str, top_k: int, recall_mode: str = "topk", extracted_mems: list | None = None, counters: TokenCounters | None = None) -> tuple[str, list]:
    q_text = question["question"]
    q_date = question.get("question_date", "")
    if recall_mode == "all":
        # 全部记忆喂给 Reader，不做召回截断。
        # 优先用提取阶段缓存的 extracted_mems（A2/A8 共用，避免重复提取）；
        # 没有缓存时回退到 scroll 整个 collection。
        if extracted_mems is not None:
            from memory.v22.models import MemorySearchResult
            hits = [MemorySearchResult(memory=m, score=0.0, signals=[]) for m in extracted_mems]
        else:
            hits = _scroll_all_memories(store)
    else:
        hits = store.search(
            q_text,
            domains=["project", "user", "agent"],
            top_k=top_k,
        )
    memory_block = MemoryRenderer.for_prompt(hits) if hits else "(no relevant memories recalled)"
    system = (
        "You are a helpful chat assistant with long-term memory of past "
        "conversations with the user. Use the recalled memories below to "
        "answer the user's question. If the memories do not contain the "
        "answer, say you don't know. Be concise."
    )
    user = (
        f"Current date: {q_date}\n\n"
        f"Recalled memories (most relevant first):\n{memory_block}\n\n"
        f"Question: {q_text}\n\nAnswer:"
    )
    from agents.llm import providers
    messages = [{"role": "user", "content": user}]
    for attempt in range(8):
        try:
            chunks: list[str] = []
            answer_usage: dict | None = None
            async for delta, usage in providers.stream_openai_with_usage(messages, answer_model, system=system, temperature=0.0):
                if delta:
                    chunks.append(delta)
                if usage:
                    answer_usage = usage
            if counters is not None:
                counters.add_answer_usage(answer_usage)
            return "".join(chunks).strip(), hits
        except Exception as e:
            msg = str(e)
            if ("429" in msg or "Too Many Requests" in msg) and attempt < 7:
                await asyncio.sleep(min(2 ** attempt, 60))
                continue
            if attempt < 7:
                await asyncio.sleep(min(2 ** attempt, 30))
                continue
            raise


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

async def run(args) -> Path:
    data_path = Path(args.data)
    questions = json.loads(data_path.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]
    if args.category:
        questions = [q for q in questions if q.get("question_type") == args.category]

    print(f"Loaded {len(questions)} questions | group={args.group} | answer={args.answer_model} | top_k={args.top_k} | recall={args.recall_mode}")

    register_builtins()
    reset_for_tests()
    store = get_memory_store()
    if store is None:
        print("FATAL: v22 store unavailable (Qdrant down or v22 disabled).", file=sys.stderr)
        sys.exit(2)
    extractor = get_memory_extractor() if args.group == "A" else None
    if args.group == "A" and extractor is None:
        print("FATAL: group A needs the memory agent; extractor unavailable.", file=sys.stderr)
        sys.exit(2)

    stem = data_path.stem
    # tag distinguishes A2 (recall-mode all) from A8 (recall-mode topk) so they
    # don't overwrite each other's hyp file.
    if args.tag:
        tag = args.tag
    elif args.group == "A":
        tag = "A2" if args.recall_mode == "all" else "A8"
    else:
        tag = args.group
    hyp_path = OUT_DIR / f"{stem}.{tag}.hyp.jsonl"
    stats_path = OUT_DIR / f"{stem}.{tag}.extraction_stats.json"

    # --- resume: skip questions already in the hyp file ---------------------
    done_qids: set[str] = set()
    all_stats: list[dict] = []
    if args.resume and hyp_path.exists():
        with hyp_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done_qids.add(rec["question_id"])
                    all_stats.append(rec)
                except Exception:
                    pass
        print(f"Resume: {len(done_qids)} questions already done, skipping them.")

    todo = [q for q in questions if q["question_id"] not in done_qids]
    print(f"Todo: {len(todo)} questions")

    t0 = time.time()
    # append mode when resuming, else truncate
    mode = "a" if (args.resume and done_qids) else "w"
    with hyp_path.open(mode, encoding="utf-8") as f:
        for i, q in enumerate(questions, 1):
            qid = q["question_id"]
            if qid in done_qids:
                continue
            # clean slate per question
            reset_collection(settings.qdrant_collection)
            reset_for_tests()
            store = get_memory_store()
            # Qdrant can transiently time out (long run, GC pauses, disk flush).
            # If the store init failed this round, retry a few times before
            # giving up — a single timeout must not kill a 500-question run.
            for _retry in range(5):
                if store is not None:
                    break
                print(f"  [store init retry {_retry+1}/5] Qdrant unreachable, waiting 10s...")
                await asyncio.sleep(10)
                reset_for_tests()
                store = get_memory_store()
            if store is None:
                print(f"  [skip] {qid}: Qdrant store unavailable after retries, skipping (will retry on resume)", file=sys.stderr)
                continue
            if args.group == "A":
                extractor = get_memory_extractor()

            counters = TokenCounters()
            store.reset_token_stats()

            ingest_stats: dict = {}
            extracted_mems = None
            if args.group == "A":
                ingest_stats, extracted_mems = await ingest_via_extractor(q, store, extractor, concurrency=args.concurrency)
                # #1 extraction token usage (memory agent → glm-5.2)
                counters.add_extract_usage(getattr(extractor, "_last_usage", None))
            else:
                ingest_stats = ingest_raw(q, store)

            # #4+#5 embedding tokens (local BGE, equivalent token counts)
            counters.add_embed_stats(store.token_stats)

            # A2 (recall_mode=all) 用提取缓存的全部记忆；A8 (topk) 走 Qdrant 召回。
            hyp, hits = await answer_question(
                q, store, args.answer_model, args.top_k,
                recall_mode=args.recall_mode, extracted_mems=extracted_mems,
                counters=counters,
            )
            rec = {
                "question_id": qid,
                "question_type": q.get("question_type"),
                "hypothesis": hyp,
                "group": args.group,
                **ingest_stats,
                "n_recalled": len(hits),
                **counters.to_dict(),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            all_stats.append(rec)
            dt = time.time() - t0
            preview = hyp.replace("\n", " ")[:70]
            print(f"[{i}/{len(questions)}] {qid} ({q.get('question_type')}) "
                  f"{ingest_stats} recalled={len(hits)} t={dt:.1f}s ans={preview!r} "
                  f"tokens(extract={counters.extract_total} answer={counters.answer_total} "
                  f"emb_bge_w={counters.embed_bge_write} emb_bge_r={counters.embed_bge_read})")
            await asyncio.sleep(args.pace)

    stats_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    # Token usage summary across all questions.
    tok = TokenCounters()
    for r in all_stats:
        tok.extract_total += r.get("extract_total_tokens", 0)
        tok.answer_total += r.get("answer_total_tokens", 0)
        tok.embed_bge_write += r.get("embed_bge_write", 0)
        tok.embed_bge_read += r.get("embed_bge_read", 0)
        tok.embed_uniform_write += r.get("embed_uniform_write", 0)
        tok.embed_uniform_read += r.get("embed_uniform_read", 0)
    n = len(all_stats) or 1
    print(f"\nHypotheses -> {hyp_path}")
    print(f"Stats      -> {stats_path}")
    print(f"\nToken totals ({len(all_stats)} questions):")
    print(f"  extract:  total={tok.extract_total}  per-q={tok.extract_total//n}")
    print(f"  answer:   total={tok.answer_total}   per-q={tok.answer_total//n}")
    print(f"  embed:    bge_write={tok.embed_bge_write} bge_read={tok.embed_bge_read} "
          f"uniform_write={tok.embed_uniform_write} uniform_read={tok.embed_uniform_read}")
    print(f"\nNext: judge + metrics:")
    print(f"  python -m eval.longmemeval.judge --hyp {hyp_path} --ref {args.data} --judge-model glm-4.7-flash")
    return hyp_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Memory agent end-to-end eval (A=extract / B=raw)")
    ap.add_argument("--data", default=str(_PROJECT_ROOT / "eval/longmemeval/data/longmemeval_oracle.json"))
    ap.add_argument("--group", choices=["A", "B"], default="A", help="A=via memory agent extract, B=raw ingest")
    ap.add_argument("--answer-model", default=os.environ.get("CT_ANSWER_MODEL", "glm-4.7-flash"))
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--category", default="")
    ap.add_argument("--pace", type=float, default=1.0)
    ap.add_argument("--recall-mode", choices=["topk", "all"], default="topk",
                    help="topk=store.search top_k truncation; all=scroll whole collection, no truncation")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="concurrent session-level extractions per question (group A)")
    ap.add_argument("--resume", action="store_true",
                    help="skip questions already in the hyp file (append mode)")
    ap.add_argument("--tag", default="",
                    help="output filename tag (default: group + recall-mode suffix, e.g. A2/A8/B)")
    args = ap.parse_args(argv)
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
