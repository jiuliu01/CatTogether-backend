r"""Run LongMemEval against the CatTogether memory + LLM stack.

Pipeline per question:
  1. Reset the agent's long-term memory (clean slate).
  2. Ingest the haystack_sessions as memory entries (one entry per turn).
  3. Build a memory context via MemoryManager.recall (keyword retrieval over the
     haystack) and assemble an OpenAI-compatible chat request.
  4. Stream the answer model's reply.
  5. Write {question_id, hypothesis} to a JSONL hypothesis file.

Then a separate judge pass scores each hypothesis with a GLM-4.7-Flash
OpenAI-compatible judge and aggregates accuracy by question_type.

This is an offline harness: it does NOT start the FastAPI server or touch
Feishu. It reuses backend memory/llm code directly so we exercise the real
CatTogether retrieval path.

Usage (from backend/):
  $env:OPENAI_API_KEY="<zhipu key>"
  $env:OPENAI_BASE_URL="https://open.bigmodel.cn/api/paas/v4/"
  .\.venv\Scripts\python.exe -m eval.longmemeval.run \
      --data eval/longmemeval/data/longmemeval_oracle.json \
      --answer-model glm-4.7-flash \
      --limit 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Backend imports (run this from backend/ so config/ is importable)
from config import settings
from memory.memory_api import MemoryAPI, WriteProposal
from memory.models import RetrievalQuery
from memory.permissions import ActorContext
from memory.db import memory_db

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_AGENT_ID = "longmemeval-sut"  # system-under-test agent id
EVAL_PROJECT_ID = None  # global memory, not project-scoped

# Singleton v2.1 API instance.
_eval_api = MemoryAPI()
_eval_actor = ActorContext(
    tenant_id="default",
    role="system",
    actor_id=EVAL_AGENT_ID,
    display_name=EVAL_AGENT_ID,
)


# --- memory helpers ---------------------------------------------------------

async def reset_memory() -> None:
    """Clear all agent-domain facts for the eval agent."""
    db = memory_db
    conn = db.get_connection()
    try:
        conn.execute(
            "DELETE FROM facts WHERE tenant_id = ? AND domain = 'agent' AND scope_id = ?",
            ("default", EVAL_AGENT_ID),
        )
        conn.execute(
            "DELETE FROM fact_embeddings WHERE tenant_id = ? AND fact_id IN "
            "(SELECT id FROM facts WHERE tenant_id = ? AND domain = 'agent' AND scope_id = ?)",
            ("default", "default", EVAL_AGENT_ID),
        )
        conn.commit()
    finally:
        db.return_connection(conn)


async def ingest_haystack(question: dict) -> int:
    """Ingest every turn of every haystack session as one memory entry each.

    Returns the number of entries written. Each turn is tagged with its
    session id and role so recall can surface them; the text is prefixed with
    a role marker so the keyword tokenizer sees who said what.
    """
    count = 0
    for sid, sess in zip(question.get("haystack_session_ids", []),
                         question.get("haystack_sessions", [])):
        for turn in sess:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if not content:
                continue
            text = f"[{role}] {content}"
            tags = [str(sid), role]
            # Slightly boost user turns; they carry the personal facts.
            importance = 0.7 if role == "user" else 0.5
            proposal = WriteProposal(
                text=text,
                domain="agent",
                scope_hint=EVAL_AGENT_ID,
                kind="project_fact",
                tags=tags,
                importance=importance,
                confidence=0.9,
                source_type="agent_result",
                actor_id=EVAL_AGENT_ID,
            )
            await _eval_api.write_propose(proposal, actor=_eval_actor)
            count += 1
    return count


# --- LLM call ---------------------------------------------------------------

async def answer_question(question: dict, answer_model: str, top_k: int) -> str:
    """Recall memory, then ask the answer model to answer the question."""
    q_text = question["question"]
    q_date = question.get("question_date", "")

    # Recall top-k memory entries relevant to the question.
    query = RetrievalQuery(
        text=q_text,
        tenant_id="default",
        domain="agent",
        scope_id=EVAL_AGENT_ID,
        intent="recall",
        top_k=top_k,
    )
    results = await _eval_api.search(query, actor=_eval_actor)

    memory_block = "\n".join(f"- {r.fact.text}" for r in results) if results else "(no relevant memories recalled)"
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

    # Call the OpenAI-compatible endpoint (works for OpenAI, Zhipu, etc.).
    # Retry on 429/5xx with exponential backoff (Zhipu free tier rate-limits).
    # 智谱免费档限流窗口是 60s 级，原来 5 次/最长 8s(共 15s) 等不到重置就崩，
    # 改成 8 次、最长 60s，并在 429 时多睡一会让窗口恢复。
    from agents.llm import providers
    messages = [{"role": "user", "content": user}]
    for attempt in range(8):
        try:
            chunks: list[str] = []
            async for delta in providers.stream_openai(messages, answer_model, system=system, temperature=0.0):
                if delta:
                    chunks.append(delta)
            return "".join(chunks).strip()
        except Exception as e:
            msg = str(e)
            is_rate = "429" in msg or "Too Many Requests" in msg
            if is_rate and attempt < 7:
                # 指数 backoff，封顶 60s；智谱窗口 ~60s，要能等到重置
                wait = min(2 ** attempt, 60)
                print(f"    [429] backoff {wait}s (attempt {attempt+1}/8)")
                await asyncio.sleep(wait)
                continue
            if attempt < 7:
                await asyncio.sleep(min(2 ** attempt, 30))
                continue
            raise


# --- main loop --------------------------------------------------------------

async def run(args) -> Path:
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data file not found: {data_path}", file=sys.stderr)
        print("Run: python -m eval.longmemeval.download_data --variant oracle", file=sys.stderr)
        sys.exit(2)

    questions = json.loads(data_path.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]
    if args.category:
        questions = [q for q in questions if q.get("question_type") == args.category]

    print(f"Loaded {len(questions)} questions from {data_path.name}")
    print(f"Answer model: {args.answer_model}  |  top_k: {args.top_k}")
    if not settings.openai_api_key:
        print("WARNING: OPENAI_API_KEY not set; answer model will return empty.", file=sys.stderr)

    stem = data_path.stem
    hyp_path = OUT_DIR / f"{stem}.model-{args.answer_model}.hyp.jsonl"

    t0 = time.time()
    with hyp_path.open("w", encoding="utf-8") as f:
        for i, q in enumerate(questions, 1):
            qid = q["question_id"]
            await reset_memory()
            n_ingested = await ingest_haystack(q)
            hyp = await answer_question(q, args.answer_model, args.top_k)
            rec = {"question_id": qid, "hypothesis": hyp, "n_memories": n_ingested}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            dt = time.time() - t0
            preview = hyp.replace("\n", " ")[:80]
            print(f"[{i}/{len(questions)}] {qid} ({q.get('question_type')}) "
                  f"mem={n_ingested} t={dt:.1f}s  ans={preview!r}")
            # gentle pacing to stay under Zhipu free-tier rate limits
            await asyncio.sleep(args.pace)
    print(f"\nHypotheses written to {hyp_path}")
    return hyp_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run LongMemEval through CatTogether memory")
    ap.add_argument("--data", default=str(DATA_DIR / "longmemeval_oracle.json"))
    ap.add_argument("--answer-model", default=os.environ.get("CT_ANSWER_MODEL", "glm-4.7-flash"))
    ap.add_argument("--top-k", type=int, default=settings.long_term_top_k)
    ap.add_argument("--limit", type=int, default=0, help="only run first N questions (0 = all)")
    ap.add_argument("--category", default="", help="only run one question_type")
    ap.add_argument("--pace", type=float, default=1.0, help="seconds to wait between questions (rate limit)")
    args = ap.parse_args(argv)
    hyp_path = asyncio.run(run(args))
    print("\nNext: judge the hypotheses with:")
    print(f"  python -m eval.longmemeval.judge --hyp {hyp_path} --ref {args.data}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
