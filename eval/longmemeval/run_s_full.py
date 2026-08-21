r"""一键跑 LongMemEval **S 变体**全量 500 题，带进度条。

S 变体是真正的考场：每题 haystack ~40 个 session / ~115k token，混入大量干扰
session，检验 CatTogether 的关键词检索能不能把证据 turn 从干扰里捞进 top_k。
（oracle 变体只含证据 session，已经 smoke 过 acc=1.0，不测检索难点。）

这个脚本把四步串成一个程序：
  1. 下载 S 变体数据（已存在则跳过）
  2. 逐题：清记忆 → 灌 haystack → recall → LLM 答  （进度条 #1）
  3. 逐题：GLM-4.7-Flash judge 打 yes/no              （进度条 #2）
  4. 汇总 overall / 按题型 / 弃权 accuracy

它复用 run.py / judge.py 里的真实逻辑（ingest_haystack、answer_question、
get_anscheck_prompt、call_judge），只把两个外层循环换成 tqdm，所以走的是
CatTogether 真实的 memory_manager.recall 关键词检索路径，不是另起一套。

用法（在 backend/ 下用 venv 跑）：
  $env:OPENAI_API_KEY  = "<你的智谱 key>"
  $env:OPENAI_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
  .\.venv\Scripts\python.exe -m eval.longmemeval.run_s_full

可选参数：
  --limit N        只跑前 N 题（默认 0 = 全量 500）
  --category TYPE  只跑某一题型（如 knowledge-update）
  --top-k K        recall 取前 K 条（默认 settings.long_term_top_k）
  --answer-model M / --judge-model M  默认都 glm-4.7-flash
  --pace S         每题间隔秒数（限流，默认 1.0）
  --skip-judge     只生成答案不 judge
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# eval/ was pulled out of backend/; make the project root importable so
# `eval.longmemeval.*` resolves, and backend/ so its modules resolve.
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]            # eval/longmemeval -> eval -> CatTogether
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# tqdm 进度条；装在 venv 里（pip install tqdm）
from tqdm import tqdm

# 复用现有 harness 的真实逻辑 —— 不重写 ingest/recall/answer/judge
from eval.longmemeval import run as run_mod
from eval.longmemeval import judge as judge_mod
from eval.longmemeval import download_data

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

S_DATA_PATH = DATA_DIR / "longmemeval_s_cleaned.json"
ANSWER_MODEL_DEFAULT = os.environ.get("CT_ANSWER_MODEL", "glm-4.7-flash")
JUDGE_MODEL_DEFAULT = os.environ.get("CT_JUDGE_MODEL", "glm-4.7-flash")


# ---------------------------------------------------------------------------
# 第 1 步：确保 S 变体数据存在
# ---------------------------------------------------------------------------
def ensure_s_data() -> Path:
    """S 变体没下就下，已存在则跳过。返回数据文件路径。"""
    if S_DATA_PATH.exists():
        print(f"[data] S 变体已存在：{S_DATA_PATH.name} "
              f"({S_DATA_PATH.stat().st_size / 1e6:.1f} MB)")
        return S_DATA_PATH
    print("[data] 下载 S 变体（~几百 MB，含干扰 session）...")
    return download_data.download("s")


# ---------------------------------------------------------------------------
# 第 2 步：逐题生成答案（ingest → recall → answer），带进度条
# ---------------------------------------------------------------------------
async def generate_answers(data_path: Path, answer_model: str, top_k: int,
                           limit: int, category: str, pace: float) -> Path:
    """跑全量题，写 outputs/<stem>.model-<model>.hyp.jsonl。返回该文件路径。"""
    questions = json.loads(data_path.read_text(encoding="utf-8"))
    if category:
        questions = [q for q in questions if q.get("question_type") == category]
    if limit:
        questions = questions[:limit]

    print(f"[run] {len(questions)} 题 | answer={answer_model} | top_k={top_k}")

    # 前置 key 检查：没 key 会让 providers.stream_openai 静默 yield ""，
    # 500 题全空跑 9 分钟白费时间。
    from config import settings
    if not (settings.openai_api_key or os.environ.get("OPENAI_API_KEY")):
        sys.exit("ERROR: OPENAI_API_KEY 未设置。先 $env:OPENAI_API_KEY=\"<智谱 key>\" 再跑。")

    hyp_path = OUT_DIR / f"{data_path.stem}.model-{answer_model}.hyp.jsonl"
    t0 = time.time()

    # tqdm 包住整个循环：显示进度、已用时、速率、预计剩余
    with hyp_path.open("w", encoding="utf-8") as f, \
         tqdm(questions, desc="answering", unit="q",
              dynamic_ncols=True) as bar:
        for q in bar:
            qid = q["question_id"]
            # 每题先清空记忆，保证 per-question 干净、不串题
            await run_mod.reset_memory()
            # 把这题的 haystack 逐 turn 灌进长期记忆（一条 turn 一条 MemoryEntry）
            n_ingested = await run_mod.ingest_haystack(q)
            # recall top_k + 喂 LLM 答题（走真实 CatTogether 检索路径）
            # 包一层 try/except：单题 429/网络错不能崩掉整个 500 题 run，
            # 记成空答案继续下一题（judge 阶段会把它判 False，但不影响其他题）。
            try:
                hyp = await run_mod.answer_question(q, answer_model, top_k)
            except Exception as e:
                hyp = ""
                bar.write(f"[warn] {qid} answer 失败，记空继续: {e!r}")

            f.write(json.dumps(
                {"question_id": qid, "hypothesis": hyp, "n_memories": n_ingested},
                ensure_ascii=False) + "\n")
            f.flush()

            # 进度条后缀显示当前题的题型 + 灌入条数 + 答案预览，方便边跑边看
            bar.set_postfix_str(
                f"{q.get('question_type','')[:18]} mem={n_ingested} "
                f"ans={hyp.replace(chr(10),' ')[:40]!r}")
            await asyncio.sleep(pace)  # 限流 pacing

    print(f"[run] done in {time.time()-t0:.1f}s -> {hyp_path}")
    return hyp_path


# ---------------------------------------------------------------------------
# 第 3 步：逐题 judge 打分，带进度条
# ---------------------------------------------------------------------------
async def judge_answers(hyp_path: Path, ref_path: Path, judge_model: str,
                        pace: float) -> Path:
    """用 GLM-4.7-Flash 对每条假设打 yes/no，写 .judge-<model>.log。"""
    from config import settings
    base_url = settings.openai_base_url
    api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[judge] 跳过：没有 OPENAI_API_KEY", file=sys.stderr)
        return hyp_path

    hypotheses = [json.loads(l) for l in hyp_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    references = json.loads(ref_path.read_text(encoding="utf-8"))
    qid2ref = {r["question_id"]: r for r in references}

    log_path = hyp_path.with_suffix(hyp_path.suffix + f".judge-{judge_model}.log")

    # 续跑：读已有 judge log，跳过已打分的题，追加写新题。
    done_qids: set[str] = set()
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_qids.add(json.loads(line)["question_id"])
    todo = [e for e in hypotheses if e["question_id"] not in done_qids]
    print(f"[judge] {len(hypotheses)} 条假设 | 已 judge {len(done_qids)} | 待 judge {len(todo)} | judge={judge_model}")

    import httpx
    t0 = time.time()
    async with httpx.AsyncClient() as client:
        # 续跑时追加写（a），全新时覆盖写（w）
        mode = "a" if done_qids else "w"
        with log_path.open(mode, encoding="utf-8") as out, \
             tqdm(todo, desc="judging", unit="q",
                  dynamic_ncols=True) as bar:
            for entry in bar:
                qid = entry["question_id"]
                ref = qid2ref.get(qid)
                if ref is None:
                    bar.set_postfix_str(f"skip {qid}")
                    continue
                task = ref["question_type"]
                abstention = "_abs" in qid  # 弃权题：判模型是否说"不知道"
                prompt = judge_mod.get_anscheck_prompt(
                    task, ref["question"], ref["answer"],
                    entry["hypothesis"], abstention=abstention)
                # 单题 429/网络错不崩全程：记 label=False 继续，最后可识别重跑。
                try:
                    raw = await judge_mod.call_judge(
                        client, base_url, api_key, judge_model, prompt)
                except Exception as e:
                    raw = ""
                    bar.write(f"[warn] {qid} judge 失败，记 no 继续: {e!r}")
                label = "yes" in raw.lower()
                entry["autoeval_label"] = {"model": judge_model, "label": label, "raw": raw}
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out.flush()
                bar.set_postfix_str(f"{task[:18]} -> {'yes' if label else 'no'}")
                await asyncio.sleep(pace)

    print(f"[judge] done in {time.time()-t0:.1f}s -> {log_path}")
    return log_path


# ---------------------------------------------------------------------------
# 第 4 步：汇总 accuracy（overall / 按题型 / 弃权）
# ---------------------------------------------------------------------------
def aggregate(log_path: Path, ref_path: Path) -> None:
    """复现官方 print_qa_metrics.py 的拆分，写 .summary.json 并打印。"""
    entries = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    references = json.loads(ref_path.read_text(encoding="utf-8"))
    qid2ref = {r["question_id"]: r for r in references}

    qtypes = sorted({r["question_type"] for r in references})
    per_type: dict[str, list[int]] = {t: [] for t in qtypes}
    abstention: list[int] = []

    for e in entries:
        ref = qid2ref.get(e["question_id"])
        if ref is None:
            continue
        s = 1 if e.get("autoeval_label", {}).get("label", False) else 0
        per_type[ref["question_type"]].append(s)
        if "_abs" in e["question_id"]:
            abstention.append(s)

    def mean(xs): return round(sum(xs) / len(xs), 4) if xs else 0.0

    all_acc = [s for xs in per_type.values() for s in xs]
    task_means = [mean(per_type[t]) for t in qtypes if per_type[t]]

    print("=" * 64)
    print("LongMemEval S — 全量结果")
    print("-" * 64)
    print(judge_mod.JUDGE_NOTE)
    print("-" * 64)
    print(f"Total judged : {len(all_acc)}")
    print(f"Overall acc  : {mean(all_acc)}")
    print(f"Task-avg acc : {mean(task_means)}")
    print("\nPer question_type:")
    for t in qtypes:
        if per_type[t]:
            print(f"  {t:32s} {mean(per_type[t])}  (n={len(per_type[t])})")
    if abstention:
        print(f"\nAbstention acc: {mean(abstention)}  (n={len(abstention)})")
    print("=" * 64)

    summary = {
        "judge_note": judge_mod.JUDGE_NOTE,
        "total": len(all_acc),
        "overall_accuracy": mean(all_acc),
        "task_averaged_accuracy": mean(task_means),
        "per_type": {t: {"accuracy": mean(per_type[t]), "n": len(per_type[t])}
                     for t in qtypes if per_type[t]},
        "abstention_accuracy": mean(abstention),
        "abstention_n": len(abstention),
    }
    sp = log_path.with_suffix(log_path.suffix + ".summary.json")
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary -> {sp}")


# ---------------------------------------------------------------------------
# main：串起四步
# ---------------------------------------------------------------------------
async def amain(args):
    # 1. 数据
    data_path = ensure_s_data()
    # hyp 文件路径（answer 阶段写这里）
    hyp_path = OUT_DIR / f"{data_path.stem}.model-{args.answer_model}.hyp.jsonl"
    if not args.judge_only:
        # 2. 生成答案（进度条 #1）
        hyp_path = await generate_answers(
            data_path, args.answer_model, args.top_k, args.limit, args.category, args.pace)
    else:
        print(f"[judge-only] 跳过 answer，直接用已有 hyp: {hyp_path}")
        if not hyp_path.exists():
            sys.exit(f"ERROR: {hyp_path} 不存在，没法只跑 judge")
    if args.skip_judge:
        print("\n[--skip-judge] 答案已写出，不跑 judge。")
        print(f"  hyp = {hyp_path}")
        return
    # 3. judge（进度条 #2，支持续跑）
    log_path = await judge_answers(hyp_path, data_path, args.judge_model, args.pace)
    # 4. 汇总
    aggregate(log_path, data_path)


def main(argv=None):
    from config import settings
    ap = argparse.ArgumentParser(description="Run LongMemEval S variant (full) with progress bars")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全量）")
    ap.add_argument("--category", default="", help="只跑某一题型")
    ap.add_argument("--top-k", type=int, default=settings.long_term_top_k)
    ap.add_argument("--answer-model", default=ANSWER_MODEL_DEFAULT)
    ap.add_argument("--judge-model", default=JUDGE_MODEL_DEFAULT)
    ap.add_argument("--pace", type=float, default=2.0, help="每题间隔秒（限流，智谱免费档建议≥2）")
    ap.add_argument("--skip-judge", action="store_true", help="只生成答案不 judge")
    ap.add_argument("--judge-only", action="store_true", help="跳过 answer，只续跑 judge（用已有 hyp）")
    args = ap.parse_args(argv)
    asyncio.run(amain(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
