"""LLM providers: streaming OpenAI and Anthropic via httpx.

Auth is strictly API-key (env vars). No interactive login. Both providers are
normalized to yield ('text_delta', {'delta': str}) chunks. Tool-calling is left
to a later milestone; here we stream text only.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from config import settings


async def stream_openai(
    messages: list[dict],
    model: str,
    system: str | None = None,
    temperature: float = 0.7,
) -> AsyncIterator[str]:
    api_key = settings.openai_api_key
    if not api_key:
        yield ""
        return
    payload_messages: list[dict] = []
    if system:
        payload_messages.append({"role": "system", "content": system})
    payload_messages.extend(messages)
    payload = {"model": model, "messages": payload_messages, "temperature": temperature, "stream": True}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj["choices"][0]["delta"].get("content")
                except Exception:
                    continue
                if delta:
                    yield delta


async def stream_anthropic(
    messages: list[dict],
    model: str,
    system: str | None = None,
    temperature: float = 0.7,
) -> AsyncIterator[str]:
    api_key = settings.anthropic_api_key
    if not api_key:
        yield ""
        return
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": temperature,
        "stream": True,
    }
    if system:
        payload["system"] = system
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    url = f"{settings.anthropic_base_url.rstrip('/')}/v1/messages"
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if not data:
                    continue
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                if obj.get("type") == "content_block_delta":
                    delta = obj.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            yield text
