r"""从 config.txt 解析 $env:NAME = "value" 行，注入当前进程 env，再跑 diag_zhipu。

匹配 PowerShell 语法：$env:OPENAI_API_KEY  = "..."
"""
import os, re
from pathlib import Path

cfg = Path(__file__).resolve().parents[3] / "config.txt"
pat = re.compile(r'\$env:(\w+)\s*=\s*"([^"]+)"')
for line in cfg.read_text(encoding="utf-8", errors="ignore").splitlines():
    m = pat.search(line)
    if m and m.group(1).startswith("OPENAI"):
        os.environ[m.group(1)] = m.group(2)

k = os.environ.get("OPENAI_API_KEY", "")
print(f"OPENAI_API_KEY: {'set len=' + str(len(k)) if k else 'NOT set'}")
print(f"OPENAI_BASE_URL: {os.environ.get('OPENAI_BASE_URL', 'NOT set')}")

from eval.longmemeval import diag_zhipu  # noqa
