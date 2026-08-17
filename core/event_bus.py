"""In-memory event bus: per-channel pub/sub with monotonic sequence numbers.

Decouples agent output (publishers) from WebSocket clients and the orchestrator
(subscribers). No persistence — on reconnect, clients refetch history via REST
and resume by last_seq.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import AsyncIterator

from models.schemas import AgentEvent


class EventBus:
    def __init__(self) -> None:
        # Per-channel broadcast: one reusable queue per channel, plus a list of
        # per-subscriber queues so slow clients don't block fast ones.
        self._subs: dict[str, list[asyncio.Queue[AgentEvent | None]]] = defaultdict(list)
        self._seqs: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    async def publish(self, channel_id: str, agent_id: str, type: str, data: dict | None = None, agent_name: str = "") -> AgentEvent:
        async with self._lock:
            self._seqs[channel_id] += 1
            seq = self._seqs[channel_id]
        event = AgentEvent(seq=seq, channel_id=channel_id, agent_id=agent_id, agent_name=agent_name, type=type, data=data or {})  # type: ignore[arg-type]
        for q in self._subs.get(channel_id, []):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest to keep stream moving; subscribers refetch history.
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception:
                    pass
        return event

    async def subscribe(self, channel_id: str) -> AsyncIterator[AgentEvent]:
        q: asyncio.Queue[AgentEvent | None] = asyncio.Queue(maxsize=1024)
        async with self._lock:
            self._subs[channel_id].append(q)
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                yield item
        finally:
            async with self._lock:
                if q in self._subs.get(channel_id, []):
                    self._subs[channel_id].remove(q)

    def unsubscribe_all(self, channel_id: str) -> None:
        for q in self._subs.get(channel_id, []):
            try:
                q.put_nowait(None)
            except Exception:
                pass


event_bus = EventBus()
