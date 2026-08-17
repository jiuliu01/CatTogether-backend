"""In-memory session/channel manager with optional JSON snapshot persistence.

No database. Channels and messages live in memory; on shutdown/restart we
optionally snapshot to data/sessions/<channel_id>.json and reload on startup.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from config import settings
from core.persistence import atomic_write_text
from models.schemas import Channel, ConversationSummary, Message, Thread


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SessionManager:
    def __init__(self) -> None:
        self._channels: dict[str, Channel] = {}
        self._messages: dict[str, list[Message]] = {}
        self._seq: dict[str, int] = {}
        self._threads: dict[str, Thread] = {}
        self._thread_keys: dict[tuple[str, str, str], str] = {}
        self._summaries: dict[str, dict[str, ConversationSummary]] = {}
        self._lock = asyncio.Lock()

    # --- channels ---
    async def create_channel(self, name: str, agent_ids: list[str] | None = None) -> Channel:
        cid = _uuid()
        ch = Channel(id=cid, name=name, agent_ids=list(agent_ids or []))
        async with self._lock:
            self._channels[cid] = ch
            self._messages[cid] = []
            self._seq[cid] = 0
            self._summaries[cid] = {}
        return ch

    async def get_channel(self, channel_id: str) -> Channel | None:
        return self._channels.get(channel_id)

    async def list_channels(self) -> list[Channel]:
        return list(self._channels.values())

    async def delete_channel(self, channel_id: str) -> bool:
        async with self._lock:
            existed = channel_id in self._channels
            self._channels.pop(channel_id, None)
            self._messages.pop(channel_id, None)
            self._seq.pop(channel_id, None)
            self._summaries.pop(channel_id, None)
            thread_ids = [tid for tid, thread in self._threads.items() if thread.channel_id == channel_id]
            for thread_id in thread_ids:
                thread = self._threads.pop(thread_id)
                self._thread_keys.pop(
                    (thread.channel_id, thread.source, thread.external_root_id), None
                )
        return existed

    async def add_agent(self, channel_id: str, agent_id: str) -> Channel | None:
        async with self._lock:
            ch = self._channels.get(channel_id)
            if ch is None:
                return None
            if agent_id not in ch.agent_ids:
                ch.agent_ids.append(agent_id)
            return ch

    async def remove_agent(self, channel_id: str, agent_id: str) -> Channel | None:
        async with self._lock:
            ch = self._channels.get(channel_id)
            if ch is None:
                return None
            if agent_id in ch.agent_ids:
                ch.agent_ids.remove(agent_id)
            return ch

    # --- messages ---
    async def append_message(
        self,
        channel_id: str,
        role: str,
        content: str,
        agent_id: str | None = None,
        thread_id: str | None = None,
        external_message_id: str | None = None,
    ) -> Message | None:
        async with self._lock:
            if channel_id not in self._channels:
                return None
            self._seq[channel_id] += 1
            msg = Message(
                id=_uuid(),
                channel_id=channel_id,
                role=role,  # type: ignore[arg-type]
                agent_id=agent_id,
                content=content,
                seq=self._seq[channel_id],
                thread_id=thread_id,
                external_message_id=external_message_id,
            )
            self._messages[channel_id].append(msg)
            return msg

    async def get_history(self, channel_id: str, limit: int = 100) -> list[Message]:
        msgs = self._messages.get(channel_id, [])
        return list(msgs[-limit:])

    async def get_agents(self, channel_id: str) -> list[str]:
        ch = self._channels.get(channel_id)
        return list(ch.agent_ids) if ch else []

    # --- threads ---
    async def get_or_create_thread(
        self,
        channel_id: str,
        external_root_id: str,
        source: str = "web",
    ) -> Thread:
        key = (channel_id, source, external_root_id)
        async with self._lock:
            existing_id = self._thread_keys.get(key)
            if existing_id:
                return self._threads[existing_id]
            thread = Thread(
                id=_uuid(),
                channel_id=channel_id,
                external_root_id=external_root_id,
                source=source,  # type: ignore[arg-type]
            )
            self._threads[thread.id] = thread
            self._thread_keys[key] = thread.id
            return thread

    async def get_thread(self, thread_id: str) -> Thread | None:
        return self._threads.get(thread_id)

    async def get_thread_history(self, thread_id: str, limit: int = 100) -> list[Message]:
        messages = [
            message
            for channel_messages in self._messages.values()
            for message in channel_messages
            if message.thread_id == thread_id
        ]
        return messages[-limit:]

    @staticmethod
    def _summary_key(thread_id: str | None) -> str:
        return f"thread:{thread_id}" if thread_id else "channel"

    async def get_scope_history(
        self,
        channel_id: str,
        *,
        thread_id: str | None = None,
        limit: int | None = None,
        after_seq: int | None = None,
    ) -> list[Message]:
        messages = self._messages.get(channel_id, [])
        if thread_id:
            messages = [message for message in messages if message.thread_id == thread_id]
        else:
            messages = [message for message in messages if message.thread_id is None]
        if after_seq is not None:
            messages = [message for message in messages if message.seq > after_seq]
        if limit is not None:
            messages = messages[-limit:]
        return list(messages)

    async def get_summary(
        self,
        channel_id: str,
        thread_id: str | None = None,
    ) -> ConversationSummary | None:
        return self._summaries.get(channel_id, {}).get(self._summary_key(thread_id))

    async def update_summary(
        self,
        channel_id: str,
        summary: ConversationSummary,
        thread_id: str | None = None,
    ) -> None:
        async with self._lock:
            self._summaries.setdefault(channel_id, {})[self._summary_key(thread_id)] = summary

    # --- persistence (JSON snapshot) ---
    def snapshot_dir(self) -> Path:
        d = settings.data_dir / "sessions"
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def snapshot(self) -> None:
        async with self._lock:
            for cid, ch in self._channels.items():
                payload = {
                    "channel": ch.model_dump(mode="json"),
                    "messages": [m.model_dump(mode="json") for m in self._messages.get(cid, [])],
                    "threads": [
                        t.model_dump(mode="json")
                        for t in self._threads.values()
                        if t.channel_id == cid
                    ],
                    "summaries": {
                        key: summary.model_dump(mode="json")
                        for key, summary in self._summaries.get(cid, {}).items()
                    },
                }
                atomic_write_text(
                    self.snapshot_dir() / f"{cid}.json",
                    json.dumps(payload, ensure_ascii=False, indent=2),
                )

    async def restore(self) -> None:
        d = self.snapshot_dir()
        if not d.exists():
            return
        for fp in d.glob("*.json"):
            try:
                payload = json.loads(fp.read_text(encoding="utf-8"))
                ch = Channel.model_validate(payload["channel"])
                msgs = [Message.model_validate(m) for m in payload["messages"]]
                threads = [Thread.model_validate(t) for t in payload.get("threads", [])]
                summaries = {
                    key: ConversationSummary.model_validate(value)
                    for key, value in payload.get("summaries", {}).items()
                }
                async with self._lock:
                    self._channels[ch.id] = ch
                    self._messages[ch.id] = msgs
                    self._seq[ch.id] = max((m.seq for m in msgs), default=0)
                    self._summaries[ch.id] = summaries
                    for thread in threads:
                        self._threads[thread.id] = thread
                        self._thread_keys[
                            (thread.channel_id, thread.source, thread.external_root_id)
                        ] = thread.id
            except Exception:
                continue


session_manager = SessionManager()
