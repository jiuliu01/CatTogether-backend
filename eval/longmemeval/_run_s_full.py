r"""从 config.txt 注入 OPENAI env 到当前进程，然后跑 run_s_full 全量。

用法：
  .\.venv\Scripts\python.exe -m eval.longmemeval._run_s_full [run_s_full 的参数]
"""
import os, re, sys
from pathlib import Path

cfg = Path(__file__).resolve().parents[3] / "config.txt"
pat = re.compile(r'\$env:(\w+)\s*=\s*"([^"]+)"')
for line in cfg.read_text(encoding="utf-8", errors="ignore").splitlines():
    m = pat.search(line)
    if m and m.group(1).startswith("OPENAI"):
        os.environ[m.group(1)] = m.group(2)

k = os.environ.get("OPENAI_API_KEY", "")
print(f"[inject] OPENAI_API_KEY: {'set len=' + str(len(k)) if k else 'NOT set'}")
print(f"[inject] OPENAI_BASE_URL: {os.environ.get('OPENAI_BASE_URL', 'NOT set')}")

# 把剩余 argv 透传给 run_s_full
sys.argv = [sys.argv[0]] + sys.argv[1:]
from eval.longmemeval.run_s_full import main
sys.exit(main())
