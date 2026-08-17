"""Normalize a Feishu message event into CatTogether's inbound message."""
from __future__ import annotations

import json
import re
from typing import Any

from config import settings
from models.schemas import FeishuInboundMessage


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            return {}
    return {}


class FeishuMessageParser:
    def parse(self, payload: dict) -> tuple[FeishuInboundMessage | None, str | None]:
        header = _as_dict(payload.get("header"))
        event = _as_dict(payload.get("event"))
        sender = _as_dict(event.get("sender"))
        sender_id = _as_dict(sender.get("sender_id"))
        message = _as_dict(event.get("message"))

        if header.get("event_type") not in (None, "", "im.message.receive_v1"):
            return None, "unsupported_event"
        if settings.feishu_app_id and header.get("app_id") not in (None, settings.feishu_app_id):
            return None, "wrong_app"
        if sender.get("sender_type") != "user":
            return None, "bot_sender"
        if message.get("message_type") != "text":
            return None, "unsupported_message_type"

        message_id = str(message.get("message_id") or "")
        chat_id = str(message.get("chat_id") or "")
        tenant_key = str(header.get("tenant_key") or sender.get("tenant_key") or "")
        open_id = str(sender_id.get("open_id") or "")
        if not all((message_id, chat_id, tenant_key, open_id)):
            return None, "missing_identity"

        content = _as_dict(message.get("content"))
        text = str(content.get("text") or "").strip()
        mentions = message.get("mentions") or []
        if not isinstance(mentions, list):
            mentions = []

        bot_mentions: list[dict] = []
        for mention in mentions:
            if not isinstance(mention, dict):
                continue
            mention_id = _as_dict(mention.get("id"))
            if settings.feishu_bot_open_id:
                if mention_id.get("open_id") == settings.feishu_bot_open_id:
                    bot_mentions.append(mention)
            else:
                # With group_at_msg permission the event is delivered because
                # the current bot was mentioned. The SDK does not require the
                # bot open_id to be configured for this common case.
                bot_mentions.append(mention)

        chat_type = str(message.get("chat_type") or "group")
        mentioned_bot = bool(bot_mentions)
        if chat_type == "group" and not mentioned_bot:
            return None, "not_mentioned"

        for mention in bot_mentions:
            key = str(mention.get("key") or "")
            name = str(mention.get("name") or "")
            if key:
                text = text.replace(key, " ")
            if name:
                text = re.sub(rf"@{re.escape(name)}\b", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return None, "empty_content"

        return FeishuInboundMessage(
            message_id=message_id,
            tenant_key=tenant_key,
            chat_id=chat_id,
            chat_type=chat_type,
            sender_open_id=open_id,
            sender_type="user",
            root_id=message.get("root_id") or None,
            parent_id=message.get("parent_id") or None,
            content=text,
            mentioned_bot=mentioned_bot,
        ), None


feishu_message_parser = FeishuMessageParser()
