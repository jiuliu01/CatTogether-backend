r"""诊断智谱 429：发 1 个请求，打印状态码、限流响应头、响应体。

在 backend/ 下、设好 OPENAI_API_KEY 的同一个 shell 里跑：
  .\.venv\Scripts\python.exe -m eval.longmemeval.diag_zhipu

用来判断 429 是哪一种：
  - 有 Retry-After 头  → 时间窗口限流，等那么久就行
  - body 里说 quota/额度/今日  → 每日额度用尽，retry 没用，得换 key/升级
  - body 里说 concurrent/并发  → 并发太高，降并发
"""
import os, json, httpx

base = os.environ.get("OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
key = os.environ.get("OPENAI_API_KEY", "")
url = base.rstrip("/") + "/chat/completions"
payload = {"model": "glm-4.7-flash",
           "messages": [{"role": "user", "content": "say hi"}],
           "max_tokens": 16, "stream": False}
headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

print(f"POST {url}")
print(f"key 长度: {len(key)}  base: {base}\n")

r = httpx.post(url, json=payload, headers=headers, timeout=30)
print(f"status: {r.status_code}\n")

print("=== 限流相关响应头 ===")
for k, v in r.headers.items():
    if any(s in k.lower() for s in ["rate", "retry", "x-", "quota", "limit", "remain"]):
        print(f"  {k}: {v}")
print()

print("=== 响应体 ===")
try:
    print(json.dumps(r.json(), ensure_ascii=False, indent=2)[:2000])
except Exception:
    print(r.text[:2000])
