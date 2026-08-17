"""Validate and enqueue Feishu receive-message events."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from config import settings
from core.task_queue import FeishuTaskQueue, feishu_task_queue
from integrations.feishu.mapping_store import FeishuMappingStore, feishu_mapping_store
from integrations.feishu.message_parser import FeishuMessageParser, feishu_message_parser
from integrations.feishu.sender import FeishuSender, feishu_sender

logger = logging.getLogger(__name__)


class FeishuEventHandler:
    def __init__(
        self,
        *,
        parser: FeishuMessageParser | None = None,
        mappings: FeishuMappingStore | None = None,
        tasks: FeishuTaskQueue | None = None,
        sender: FeishuSender | None = None,
    ) -> None:
        self.parser = parser or feishu_message_parser
        self.mappings = mappings or feishu_mapping_store
        self.tasks = tasks or feishu_task_queue
        self.sender = sender or feishu_sender

    async def handle_raw(self, payload: dict[str, Any]) -> str:
        inbound, reason = self.parser.parse(payload)
        if inbound is None:
            if reason in ("unsupported_message_type", "empty_content"):
                message_id = self._message_id(payload)
                if message_id:
                    asyncio.create_task(
                        self._reply_unsupported(message_id, reason)
                    )
            return reason or "ignored"

        if not self._allowed(
            inbound.tenant_key,
            inbound.chat_id,
            inbound.sender_open_id,
        ):
            return "unauthorized"
        if not await self.mappings.mark_processed(inbound.message_id):
            return "duplicate"
        await self.tasks.enqueue(inbound)
        return "queued"

    def submit_threadsafe(
        self,
        payload: dict[str, Any],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        future = asyncio.run_coroutine_threadsafe(self.handle_raw(payload), loop)

        def log_failure(done) -> None:
            try:
                done.result()
            except Exception:
                logger.exception("failed to process Feishu event")

        future.add_done_callback(log_failure)

    @staticmethod
    def _allowed(tenant_key: str, chat_id: str, open_id: str) -> bool:
        checks = (
            (settings.feishu_allowed_tenant_keys, tenant_key),
            (settings.feishu_allowed_chat_ids, chat_id),
            (settings.feishu_allowed_open_ids, open_id),
        )
        return all(not allowed or value in allowed for allowed, value in checks)

    async def _reply_unsupported(self, message_id: str, reason: str) -> None:
        text = (
            "暂时只支持文本任务。"
            if reason == "unsupported_message_type"
            else "请在 @CatTogether 后写明要执行的任务。"
        )
        try:
            await self.sender.reply(message_id, text, reply_in_thread=True)
        except Exception:
            logger.warning("failed to reply to unsupported Feishu message")

    @staticmethod
    def _message_id(payload: dict[str, Any]) -> str | None:
        event = payload.get("event")
        if not isinstance(event, dict):
            return None
        message = event.get("message")
        if not isinstance(message, dict):
            return None
        value = message.get("message_id")
        return str(value) if value else None


feishu_event_handler = FeishuEventHandler()
