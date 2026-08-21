# Memory Agent 端到端评测

评测 Memory 2.2 的 memory agent（记忆提取器）在 LongMemEval 上的端到端价值。
完整方案见上级 `eval/评测方案.md`。

## 三组对照

| 组 | 提取 | 存储/检索 | 脚本 |
|----|------|-----------|------|
| A | memory agent 提取 | Qdrant dense+bm25 RRF | `run_e2e.py --group A` |
| B | 不提取，原文直写 | Qdrant dense+bm25 RRF | `run_e2e.py --group B` |
| C | 不提取，原文灌入 | 2.1 SQLite 关键词 | `../longmemeval/run.py`（已有 outputs） |

- A vs B = 提取的净贡献（固定检索，只换提取）
- A vs C = v22 相对旧链路的整体提升

## 运行

```powershell
$py = "C:\Users\20772\miniconda3\envs\Catenv\python.exe"
$env:OPENAI_API_KEY  = "<zhipu key>"
$env:OPENAI_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
$env:CT_MEMORY_V22_ENABLED = "true"

# smoke：oracle 前 20 题，A 组
& $py -m eval.memory_agent.run_e2e --data eval/longmemeval/data/longmemeval_oracle.json --group A --limit 20

# B 组对照
& $py -m eval.memory_agent.run_e2e --data eval/longmemeval/data/longmemeval_oracle.json --group B --limit 20

# judge + metrics（复用 longmemeval）
& $py -m eval.longmemeval.judge --hyp eval/memory_agent/outputs/longmemeval_oracle.A.hyp.jsonl --ref eval/longmemeval/data/longmemeval_oracle.json
& $py -m eval.longmemeval.metrics --log <上面的 judge log> --ref eval/longmemeval/data/longmemeval_oracle.json
```

## 前置

- Qdrant 起在 `127.0.0.1:6333`（`docker run ... qdrant/qdrant`）
- conda `Catenv`（qdrant-client / fastembed / sentence-transformers）
- 智谱 API key（OpenAI 兼容端点）

## 输出

`outputs/{variant}.{group}.hyp.jsonl` — 每题 `{question_id, hypothesis, n_extracted/n_raw_written, n_skipped_dup, n_recalled, ...}`
`outputs/{variant}.{group}.extraction_stats.json` — 全量过程指标

## 提取质量小样本

`extract_eval.py` + `gold_samples.jsonl`（待建）——直接度量提取 precision/recall/拒记/归属/domain，见方案 §7。
