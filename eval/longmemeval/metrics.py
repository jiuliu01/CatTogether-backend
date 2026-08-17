r"""Aggregate LongMemEval judge results into accuracy by question_type.

Reproduces the official print_qa_metrics.py breakdown (overall micro-average,
per question_type, task-averaged macro-average, abstention subset) and tags
the output with the non-official-judge note.

Usage (from backend/):
  .\.venv\Scripts\python.exe -m eval.longmemeval.metrics \
      --log eval/longmemeval/outputs/<...>.judge-glm-4.7-flash.log \
      --ref eval/longmemeval/data/longmemeval_oracle.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

JUDGE_NOTE = (
    "NON-OFFICIAL judge: GLM-4.7-Flash via OpenAI-compatible endpoint. "
    "Scores are NOT directly comparable to the official GPT-4o evaluation "
    "in the LongMemEval paper."
)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Print LongMemEval QA metrics")
    ap.add_argument("--log", required=True)
    ap.add_argument("--ref", required=True)
    args = ap.parse_args(argv)

    log_path = Path(args.log)
    ref_path = Path(args.ref)

    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    references = json.loads(ref_path.read_text(encoding="utf-8"))
    qid2ref = {r["question_id"]: r for r in references}

    qtypes = sorted({r["question_type"] for r in references})
    qtype2acc: dict[str, list[int]] = {t: [] for t in qtypes}
    abstention_acc: list[int] = []

    for entry in entries:
        qid = entry["question_id"]
        ref = qid2ref.get(qid)
        if ref is None:
            continue
        label = entry.get("autoeval_label", {}).get("label", False)
        score = 1 if label else 0
        qtype2acc[ref["question_type"]].append(score)
        if "_abs" in qid:
            abstention_acc.append(score)

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    all_acc = [s for xs in qtype2acc.values() for s in xs]
    task_means = [mean(qtype2acc[t]) for t in qtypes if qtype2acc[t]]

    print("=" * 60)
    print("LongMemEval QA metrics")
    print("-" * 60)
    print(JUDGE_NOTE)
    print("-" * 60)
    print(f"Total judged: {len(all_acc)}")
    print(f"Overall accuracy: {mean(all_acc)}")
    print(f"Task-averaged accuracy: {mean(task_means)}")
    print()
    print("Per question_type:")
    for t in qtypes:
        xs = qtype2acc[t]
        if xs:
            print(f"  {t:32s} {mean(xs)}  (n={len(xs)})")
    if abstention_acc:
        print()
        print(f"Abstention accuracy: {mean(abstention_acc)}  (n={len(abstention_acc)})")
    print("=" * 60)

    # also write a json summary next to the log
    summary = {
        "judge_note": JUDGE_NOTE,
        "total": len(all_acc),
        "overall_accuracy": mean(all_acc),
        "task_averaged_accuracy": mean(task_means),
        "per_type": {t: {"accuracy": mean(qtype2acc[t]), "n": len(qtype2acc[t])} for t in qtypes if qtype2acc[t]},
        "abstention_accuracy": mean(abstention_acc),
        "abstention_n": len(abstention_acc),
    }
    summary_path = log_path.with_suffix(log_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
