r"""Download LongMemEval dataset JSON files from HuggingFace.

Usage (from backend/):
  .\.venv\Scripts\python.exe -m eval.longmemeval.download_data --variant oracle
  .\.venv\Scripts\python.exe -m eval.longmemeval.download_data --variant all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

VARIANT_FILES = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}
VARIANT_URLS = {
    "oracle": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
    "s": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
    "m": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json",
}


def download(variant: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = VARIANT_URLS[variant]
    dest = DATA_DIR / VARIANT_FILES[variant]
    if dest.exists():
        print(f"[skip] {dest.name} already exists ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    print(f"Downloading {variant}: {url}")
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(600.0, connect=30.0)) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = 0
            with dest.open("wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1 << 20):
                    f.write(chunk)
                    total += len(chunk)
                    print(f"\r  {total/1e6:.1f} MB", end="", flush=True)
            print()
    print(f"[ok] {dest} ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download LongMemEval data")
    ap.add_argument("--variant", choices=list(VARIANT_FILES) + ["all"], default="oracle")
    args = ap.parse_args(argv)
    variants = list(VARIANT_FILES) if args.variant == "all" else [args.variant]
    for v in variants:
        download(v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
