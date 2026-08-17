"""Send and reply to Feishu messages with tenant_access_token caching."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

import httpx

from config import settings
from core.output_safety import sanitize_agent_text


logger = logging.getLogger(__name__)


class FeishuAPIError(RuntimeError):
    pass


class FeishuSender:
    def __init__(self) -> None:
        self._token: str | None = None
        self._token_expires_at = 0.0
        self._token_lock = asyncio.Lock()

    async def _access_token(self) -> str:
        if self._token and time.monotonic() < self._token_expires_at:
            return self._token
        if not settings.feishu_enabled:
            raise FeishuAPIError("FEISHU_APP_ID and FEISHU_APP_SECRET are required")
        async with self._token_lock:
            if self._token and time.monotonic() < self._token_expires_at:
                return self._token
            url = f"{settings.feishu_api_base_url}/open-apis/auth/v3/tenant_access_token/internal"
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    url,
                    json={
                        "app_id": settings.feishu_app_id,
                        "app_secret": settings.feishu_app_secret,
                    },
                )
                response.raise_for_status()
                data = response.json()
            if data.get("code") != 0 or not data.get("tenant_access_token"):
                raise FeishuAPIError(data.get("msg") or "failed to obtain tenant_access_token")
            self._token = data["tenant_access_token"]
            self._token_expires_at = time.monotonic() + max(int(data.get("expire", 7200)) - 300, 60)
            return self._token

    async def _post_with_retry(self, url: str, body: dict) -> dict:
        return await self._request_with_retry("POST", url, body)

    async def _request_with_retry(self, method: str, url: str, body: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                token = await self._access_token()
                async with httpx.AsyncClient(timeout=15) as client:
                    call = getattr(client, method.lower())
                    response = await call(
                        url,
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Content-Type": "application/json; charset=utf-8",
                        },
                        json=body,
                    )
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue
                break

            detail = self._response_detail(response)
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = FeishuAPIError(detail)
                if attempt < 2:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue
                break
            if response.status_code >= 400:
                raise FeishuAPIError(detail)
            try:
                data = response.json()
            except ValueError as exc:
                raise FeishuAPIError(detail) from exc
            if data.get("code") != 0:
                raise FeishuAPIError(detail)
            return data
        raise FeishuAPIError(str(last_error or "unknown Feishu API error"))

    @staticmethod
    def _response_detail(response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            data = {}
        code = data.get("code") if isinstance(data, dict) else None
        message = data.get("msg") if isinstance(data, dict) else None
        headers = getattr(response, "headers", {}) or {}
        request_id = (
            headers.get("x-request-id")
            or headers.get("x-tt-logid")
            or (data.get("request_id") if isinstance(data, dict) else None)
        )
        parts = [f"HTTP {response.status_code}"]
        if code is not None:
            parts.append(f"code={code}")
        if message:
            parts.append(f"msg={message}")
        if request_id:
            parts.append(f"request_id={request_id}")
        return "Feishu API error: " + ", ".join(parts)

    async def reply(
        self,
        message_id: str,
        text: str,
        *,
        reply_in_thread: bool = True,
        request_uuid: str | None = None,
    ) -> list[dict]:
        text = self._redact(text)
        results = []
        chunks = self._split_text(text)
        for index, chunk in enumerate(chunks):
            unique = request_uuid or uuid.uuid4().hex
            if len(chunks) > 1:
                unique = f"{unique[:42]}-{index}"
            url = (
                f"{settings.feishu_api_base_url}/open-apis/im/v1/messages/"
                f"{message_id}/reply"
            )
            results.append(
                await self._post_with_retry(
                    url,
                    {
                        "msg_type": "text",
                        "content": json.dumps({"text": chunk}, ensure_ascii=False),
                        "reply_in_thread": reply_in_thread,
                        "uuid": unique[:50],
                    },
                )
            )
        return results

    async def send_card(self, message_id: str, card: dict, *, reply_in_thread: bool = True) -> str | None:
        """Send an interactive card as a reply; return the new message_id for updates."""
        card = self._sanitize_payload(card)
        self._validate_card_payload(card)
        url = (
            f"{settings.feishu_api_base_url}/open-apis/im/v1/messages/"
            f"{message_id}/reply"
        )
        data = await self._post_with_retry(
            url,
            {
                "msg_type": "interactive",
                "content": json.dumps(card, ensure_ascii=False),
                "reply_in_thread": reply_in_thread,
            },
        )
        msg = data.get("data", {}).get("message_id") if isinstance(data.get("data"), dict) else None
        return msg

    async def update_card(self, card_message_id: str, card: dict) -> bool:
        """Patch an already-sent interactive card with new content."""
        card = self._sanitize_payload(card)
        self._validate_card_payload(card)
        url = (
            f"{settings.feishu_api_base_url}/open-apis/im/v1/messages/"
            f"{card_message_id}"
        )
        try:
            await self._request_with_retry(
                "PATCH",
                url,
                {
                    "content": json.dumps(card, ensure_ascii=False),
                },
            )
            return True
        except (httpx.HTTPError, FeishuAPIError) as exc:
            logger.warning("Feishu card update failed: %s", exc, exc_info=True)
            raise

    @staticmethod
    def _split_text(text: str, limit: int | None = None) -> list[str]:
        limit = max(int(limit or settings.chat_chunk_limit), 1)
        text = text.strip() or "任务已完成。"
        if len(text) <= limit:
            return [text]
        chunks: list[str] = []
        remaining = text
        while remaining:
            if len(remaining) <= limit:
                chunks.append(remaining)
                break
            split_at = remaining.rfind("\n", 0, limit)
            if split_at < limit // 2:
                split_at = limit
            chunks.append(remaining[:split_at].rstrip())
            remaining = remaining[split_at:].lstrip()
        return chunks

    @staticmethod
    def _redact(text: str) -> str:
        return sanitize_agent_text(text).text

    @classmethod
    def _sanitize_payload(cls, value):
        if isinstance(value, str):
            return cls._redact(value)
        if isinstance(value, list):
            return [cls._sanitize_payload(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._sanitize_payload(item) for key, item in value.items()}
        return value

    @staticmethod
    def _validate_card_payload(card: dict) -> None:
        if card.get("schema") != "2.0":
            raise FeishuAPIError("Feishu card must use schema 2.0")

        def walk(value):
            if isinstance(value, dict):
                tag = value.get("tag")
                if tag in {"note", "hr"}:
                    raise FeishuAPIError(
                        f"Feishu card schema 2.0 does not support legacy tag: {tag}"
                    )
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(card)
        encoded = json.dumps(card, ensure_ascii=False).encode("utf-8")
        if len(encoded) > 28_000:
            raise FeishuAPIError(
                f"Feishu card payload exceeds safe 28 KB limit: {len(encoded)} bytes"
            )


feishu_sender = FeishuSender()
