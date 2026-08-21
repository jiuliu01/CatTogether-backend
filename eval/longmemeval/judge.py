r"""Judge LongMemEval hypotheses with a GLM-4.7-Flash OpenAI-compatible judge.

This reuses the official LongMemEval prompt templates (evaluate_qa.py) but
points the judge client at a custom OpenAI-compatible base_url + model
(Zhipu GLM-4.7-Flash by default) instead of GPT-4o.

IMPORTANT: GLM-4.7-Flash is NOT the official judge. Scores are not directly
comparable to the GPT-4o numbers in the LongMemEval paper. This is recorded
in the output log under "judge_model" and "judge_note".

Usage (from backend/):
  .\.venv\Scripts\python.exe -m eval.longmemeval.judge \
      --hyp eval/longmemeval/outputs/longmemeval_oracle.model-glm-4.7-flash.hyp.jsonl \
      --ref eval/longmemeval/data/longmemeval_oracle.json \
      --judge-model glm-4.7-flash
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# eval/ was pulled out of backend/; make project root + backend importable so
# `config` (used in settings_openai_base_url) resolves when run as a module.
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import httpx

HERE = Path(__file__).resolve().parent

JUDGE_NOTE = (
    "NON-OFFICIAL judge: GLM-4.7-Flash via OpenAI-compatible endpoint. "
    "Scores are NOT directly comparable to the official GPT-4o evaluation "
    "in the LongMemEval paper."
)


# --- official prompt templates (verbatim from LongMemEval evaluate_qa.py) ----

def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = ("I will give you a question, a correct answer, and a response from a model. "
                        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                        "If the response is equivalent to the correct answer or contains all the intermediate "
                        "steps to get the correct answer, you should also answer yes. If the response only "
                        "contains a subset of the information required by the answer, answer no. \n\n"
                        "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                        "Is the model response correct? Answer yes or no only.")
            prompt = template.format(question, answer, response)
        elif task == "temporal-reasoning":
            template = ("I will give you a question, a correct answer, and a response from a model. "
                        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                        "If the response is equivalent to the correct answer or contains all the intermediate "
                        "steps to get the correct answer, you should also answer yes. If the response only "
                        "contains a subset of the information required by the answer, answer no. In addition, "
                        "do not penalize off-by-one errors for the number of days. If the question asks for the "
                        "number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
                        "predicting 19 days when the answer is 18), the model's response is still correct. \n\n"
                        "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                        "Is the model response correct? Answer yes or no only.")
            prompt = template.format(question, answer, response)
        elif task == "knowledge-update":
            template = ("I will give you a question, a correct answer, and a response from a model. "
                        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                        "If the response contains some previous information along with an updated answer, the "
                        "response should be considered as correct as long as the updated answer is the required answer.\n\n"
                        "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                        "Is the model response correct? Answer yes or no only.")
            prompt = template.format(question, answer, response)
        elif task == "single-session-preference":
            template = ("I will give you a question, a rubric for desired personalized response, and a response "
                        "from a model. Please answer yes if the response satisfies the desired response. Otherwise, "
                        "answer no. The model does not need to reflect all the points in the rubric. The response "
                        "is correct as long as it recalls and utilizes the user's personal information correctly.\n\n"
                        "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
                        "Is the model response correct? Answer yes or no only.")
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError(task)
    else:
        template = ("I will give you an unanswerable question, an explanation, and a response from a model. "
                    "Please answer yes if the model correctly identifies the question as unanswerable. The model "
                    "could say that the information is incomplete, or some other information is given but the asked "
                    "information is not.\n\n"
                    "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
                    "Does the model correctly identify the question as unanswerable? Answer yes or no only.")
        prompt = template.format(question, answer, response)
    return prompt


# --- judge client -----------------------------------------------------------

async def call_judge(client, base_url, api_key, model, prompt):
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        # GLM-4.7-Flash is a reasoning model: it spends ~230-280 reasoning
        # tokens before emitting the final yes/no. A small budget here returns
        # an empty content string and every label silently becomes False.
        # 2048 leaves comfortable headroom for the reasoning trace + answer.
        "max_tokens": 2048,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # 和 run.py 一致的 429 加固：8 次、backoff 封顶 60s。
    # 智谱免费档限流窗口 ~60s，原来 4 次/最长 8s 等不到重置就崩。
    for attempt in range(8):
        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=httpx.Timeout(120.0, connect=10.0))
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            # glm-5.2 is a reasoning model: the final yes/no may land in
            # ``content`` or, if the proxy folds the reasoning trace into the
            # message, in ``reasoning_content``. Prefer ``content``; fall back
            # to ``reasoning_content`` and take its last line so we don't pull
            # the whole chain into the yes/no label.
            text = (msg.get("content") or "").strip()
            if not text:
                rc = (msg.get("reasoning_content") or "").strip()
                text = rc.splitlines()[-1].strip() if rc else ""
            return text
        except Exception as e:
            if attempt == 7:
                raise
            msg = str(e)
            is_rate = "429" in msg or "Too Many Requests" in msg
            wait = min(2 ** attempt, 60) if is_rate else min(2 ** attempt, 30)
            await asyncio.sleep(wait)


async def run(args) -> Path:
    hyp_path = Path(args.hyp)
    ref_path = Path(args.ref)

    hypotheses = [json.loads(line) for line in hyp_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    references = json.loads(ref_path.read_text(encoding="utf-8"))
    qid2ref = {r["question_id"]: r for r in references}

    base_url = args.base_url or settings_openai_base_url()
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: no API key. Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        sys.exit(2)

    log_path = hyp_path.with_suffix(hyp_path.suffix + f".judge-{args.judge_model}.log")
    print(f"Judging {len(hypotheses)} hypotheses with {args.judge_model} @ {base_url}")

    t0 = time.time()
    async with httpx.AsyncClient() as client:
        with log_path.open("w", encoding="utf-8") as out:
            for i, entry in enumerate(hypotheses, 1):
                qid = entry["question_id"]
                ref = qid2ref.get(qid)
                if ref is None:
                    print(f"[{i}] skip {qid}: not in reference")
                    continue
                task = ref["question_type"]
                question = ref["question"]
                answer = ref["answer"]
                hyp = entry["hypothesis"]
                abstention = "_abs" in qid
                prompt = get_anscheck_prompt(task, question, answer, hyp, abstention=abstention)
                raw = await call_judge(client, base_url, api_key, args.judge_model, prompt)
                label = "yes" in raw.lower()
                entry["autoeval_label"] = {"model": args.judge_model, "label": label, "raw": raw}
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out.flush()
                dt = time.time() - t0
                print(f"[{i}/{len(hypotheses)}] {qid} ({task}) -> {label} ({raw!r}) t={dt:.1f}s")
                await asyncio.sleep(args.pace)

    print(f"\nJudge log: {log_path}")
    return log_path


def settings_openai_base_url() -> str:
    from config import settings
    return settings.openai_base_url


def main(argv=None):
    ap = argparse.ArgumentParser(description="Judge LongMemEval hypotheses")
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--judge-model", default=os.environ.get("CT_JUDGE_MODEL", "glm-4.7-flash"))
    ap.add_argument("--base-url", default="")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--pace", type=float, default=2.0, help="seconds between judge calls (rate limit)")
    args = ap.parse_args(argv)
    log_path = asyncio.run(run(args))
    print("\nNext: aggregate metrics with:")
    print(f"  python -m eval.longmemeval.metrics --log {log_path} --ref {args.ref}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
