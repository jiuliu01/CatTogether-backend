# LongMemEval × CatTogether

Offline harness that runs the [LongMemEval](https://github.com/xiaowu0162/LongMemEval)
benchmark against CatTogether's memory + LLM stack. It does **not** start the
FastAPI server or touch Feishu — it reuses `backend/memory/` and
`backend/agents/llm/providers.py` directly so the real CatTogether retrieval
path is exercised.

## Pipeline

For each of the 500 questions:

1. **Reset** the SUT agent's long-term memory (`long_term_memory.clear`).
2. **Ingest** every turn of `haystack_sessions` as one memory entry each
   (`long_term_memory.bulk_load`), tagged with session id + role. User turns
   get importance 0.7, assistant turns 0.5.
3. **Recall** top-k entries via `memory_manager.recall(question)` — this is
   CatTogether's keyword-overlap retrieval (no embeddings).
4. **Answer**: the recalled memories are injected into the system prompt and
   the question is sent to the answer model through `providers.stream_openai`
   (OpenAI-compatible endpoint).
5. **Judge**: a separate pass scores each hypothesis yes/no with a
   GLM-4.7-Flash OpenAI-compatible judge using the official LongMemEval prompt
   templates.
6. **Aggregate**: overall + per-`question_type` + abstention accuracy.

> ⚠️ **Non-official judge.** GLM-4.7-Flash is used instead of GPT-4o. Scores
> are **not directly comparable** to the numbers in the LongMemEval paper.
> This note is recorded in every judge log and metrics summary.

## Dataset variants

| variant | file | size |
|---------|------|------|
| oracle  | `longmemeval_oracle.json`   | evidence sessions only (~15 MB) |
| s       | `longmemeval_s_cleaned.json`| ~115k tokens, ~40 sessions/Q    |
| m       | `longmemeval_m_cleaned.json`| ~500 sessions/Q                 |

## Run

All commands run from `backend/` with the venv python:

```powershell
$env:OPENAI_API_KEY   = "<your zhipu key>"
$env:OPENAI_BASE_URL  = "https://open.bigmodel.cn/api/paas/v4/"
$py = ".\.venv\Scripts\python.exe"

# 1. download data (oracle for smoke test, or 'all')
& $py -m eval.longmemeval.download_data --variant oracle

# 2. generate answers (limit 20 for a smoke test; drop --limit for all 500)
& $py -m eval.longmemeval.run `
    --data eval/longmemeval/data/longmemeval_oracle.json `
    --answer-model glm-4.7-flash `
    --limit 20

# 3. judge the hypotheses
& $py -m eval.longmemeval.judge `
    --hyp eval/longmemeval/outputs/longmemeval_oracle.model-glm-4.7-flash.hyp.jsonl `
    --ref eval/longmemeval/data/longmemeval_oracle.json `
    --judge-model glm-4.7-flash

# 4. aggregate metrics
& $py -m eval.longmemeval.metrics `
    --log eval/longmemeval/outputs/longmemeval_oracle.model-glm-4.7-flash.hyp.jsonl.judge-glm-4.7-flash.log `
    --ref eval/longmemeval/data/longmemeval_oracle.json
```

## Config

| env var | default | meaning |
|---------|---------|---------|
| `OPENAI_API_KEY`  | —     | Zhipu API key (OpenAI-compatible) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Zhipu base URL |
| `CT_ANSWER_MODEL` | `glm-4.7-flash` | answer model override |
| `CT_JUDGE_MODEL`  | `glm-4.7-flash` | judge model override |

`run.py` flags: `--data`, `--answer-model`, `--top-k` (default from
`CT_LONG_TERM_TOP_K`), `--limit`, `--category`.

## Files

```
eval/longmemeval/
  download_data.py   # fetch JSON from HuggingFace
  run.py             # ingest + recall + answer  -> outputs/*.hyp.jsonl
  judge.py           # score hypotheses          -> outputs/*.judge-*.log
  metrics.py         # aggregate accuracy        -> outputs/*.summary.json
  data/              # downloaded datasets
  outputs/           # hypotheses, judge logs, summaries
```
