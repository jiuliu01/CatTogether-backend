"""Live Feishu run card with persisted progress, heartbeat, and fallback."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from core.progress_formatter import format_progress
from core.run_event_store import run_event_store
from integrations.feishu.card_builder import CardBuilder
from integrations.feishu.sender import FeishuSender
from models.schemas import RunProgressEvent


logger = logging.getLogger(__name__)
DeliveryCallback = Callable[[dict], Awaitable[None]]


# --- Structured tool-call trace (developer observability, off by default) ---
_TOOL_ARG_KEYS = (
    "file_path", "path", "pattern", "query", "command",
    "target", "task", "access", "domain", "scope_id", "top_k",
)


def _trace_truncate(value: str, limit: int) -> str:
    value = value or ""
    return value if len(value) <= limit else value[:limit] + "…"


def _structured_tool_event(kind: str, data: dict) -> dict:
    """Pick a few meaningful fields per tool call — never a raw dump."""
    from core.output_safety import sanitize_agent_text

    raw_args = data.get("args") if isinstance(data.get("args"), dict) else {}
    summary_args: dict[str, str] = {}
    for key in _TOOL_ARG_KEYS:
        if key in raw_args:
            summary_args[key] = _trace_truncate(
                sanitize_agent_text(str(raw_args[key])).text, 300
            )
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "tool": str(data.get("tool") or ""),
        "tool_use_id": str(data.get("tool_use_id") or "")[:100],
        "agent_id": str(data.get("agent_id") or ""),
        "agent_name": str(data.get("agent_name") or ""),
        "phase": str(data.get("phase") or ""),
        "args": summary_args,
    }
    if kind == "tool_result":
        rec["result"] = _trace_truncate(
            sanitize_agent_text(str(data.get("result") or "")).text, 800
        )
        rec["is_error"] = bool(data.get("is_error"))
    return rec


def _append_tool_trace(run_id: str, event: dict) -> None:
    """Append one structured tool event to data/feishu/run_tools/<run_id>.jsonl."""
    try:
        d = settings.data_dir / "feishu" / "run_tools"
        d.mkdir(parents=True, exist_ok=True)
        with (d / f"{run_id}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("tool trace append failed", exc_info=True)



class FeishuProgressReporter:
    def __init__(
        self,
        sender: FeishuSender,
        message_id: str,
        run_id: str,
        *,
        update_interval: float | None = None,
        heartbeat_interval: float | None = None,
        entry_agent_name: str = "",
        on_delivery: DeliveryCallback | None = None,
        existing_card_message_id: str | None = None,
        elapsed_seconds: int = 0,
    ) -> None:
        self._sender = sender
        self._message_id = message_id
        self._run_id = run_id
        self._update_interval = max(
            float(update_interval if update_interval is not None else settings.feishu_progress_update_interval),
            0.0,
        )
        self._heartbeat_interval = max(
            float(
                heartbeat_interval
                if heartbeat_interval is not None
                else settings.feishu_progress_heartbeat_interval
            ),
            0.05,
        )
        self._card = CardBuilder(
            run_id,
            timeline_limit=settings.feishu_progress_timeline_limit,
        )
        self._card.set_entry_agent(entry_agent_name)
        self._card_msg_id: str | None = existing_card_message_id
        self._card_disabled_for_run = False
        self._last_update = 0.0
        self._started_at = time.monotonic() - max(int(elapsed_seconds), 0)
        self._last_event_at = self._started_at
        self._last_business_event_at = self._started_at
        self._stalled_reported = False
        self._last_fallback_progress = 0.0
        self._lock = asyncio.Lock()
        self._fallback_used = False
        self._heartbeat_task: asyncio.Task | None = None
        self._entry_agent_name = entry_agent_name
        self._event_count = 0
        self._delivery_state = "card_sent" if existing_card_message_id else "not_started"
        self._delivery_error = ""
        self._on_delivery = on_delivery
        self._last_delivery_notice: tuple[str, str, str | None] | None = None
        self._recent_checkpoints: dict[str, float] = {}

    async def start(self, *, agent_id: str = "", phase: str = "entry") -> None:
        """Immediately send the live card and start silent-run heartbeats."""
        await self.activate()
        await self.on_progress(
            "stage",
            {
                "phase": phase,
                "agent_id": agent_id,
                "agent_name": self._entry_agent_name,
            },
        )

    async def activate(self) -> None:
        """Restart heartbeats without creating another visible stage."""
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(),
                name=f"feishu-progress-heartbeat-{self._run_id}",
            )

    def restore_events(self, events: list[RunProgressEvent]) -> None:
        """Rebuild one existing card after restart without duplicating events."""
        for event in events:
            self._event_count = max(self._event_count, event.seq)
            if event.kind == "stage":
                self._card.begin_stage(event.agent_name, event.phase)
                self._card.latest_status = event.title
            elif event.kind == "heartbeat":
                self._card.latest_status = event.title
            else:
                self._card.record_progress(
                    event.title,
                    detail=event.detail,
                    agent_name=event.agent_name,
                    phase=event.phase,
                    timestamp=event.created_at.astimezone().strftime("%H:%M:%S"),
                )
        self._card.set_elapsed(int(time.monotonic() - self._started_at))

    async def on_progress(self, kind: str, data: dict) -> None:
        """Persist one progress item and coalesce it into the live card."""
        # Developer-only structured tool-call trace. Off by default; a no-op
        # when the flag is unset so production pays nothing.
        if settings.agent_tool_trace and kind in ("tool_call", "tool_result"):
            _append_tool_trace(self._run_id, _structured_tool_event(kind, data))
        try:
            from core.run_store import run_store
            getter = getattr(run_store, "get", None)
            run = await getter(self._run_id) if getter else None
            if run and run.status in ("completed", "failed", "cancelled"):
                return
            formatted = format_progress(kind, data)
            if formatted is None:
                return
            now = time.monotonic()
            if formatted.kind == "checkpoint":
                checkpoint_key = f"{formatted.title}\n{formatted.detail}".strip()
                previous = self._recent_checkpoints.get(checkpoint_key)
                if previous is not None and now - previous < settings.checkpoint_min_interval:
                    return
                self._recent_checkpoints[checkpoint_key] = now
                self._recent_checkpoints = {
                    key: seen for key, seen in self._recent_checkpoints.items()
                    if now - seen < settings.checkpoint_min_interval
                }
            self._last_event_at = now
            if kind not in {"heartbeat", "stalled"}:
                self._last_business_event_at = now
                self._stalled_reported = False
            agent_name = str(data.get("agent_name") or self._entry_agent_name or "")
            agent_id = str(data.get("agent_id") or "")
            phase = str(data.get("phase") or "entry")
            event = RunProgressEvent(
                run_id=self._run_id,
                kind=formatted.kind,  # type: ignore[arg-type]
                agent_id=agent_id,
                agent_name=agent_name,
                phase=phase,
                title=formatted.title,
                detail=formatted.detail,
                child_run_id=str(data.get("child_run_id") or "") or None,
                target=str(data.get("target") or "") or None,
                status=str(data.get("status") or "") or None,
                artifact_id=str(data.get("artifact_id") or "") or None,
                type=str(data.get("artifact_type") or data.get("type") or "") or None,
                url=str(data.get("url") or "") or None,
            )
            try:
                event = await run_event_store.append(event)
            except Exception:
                # Persistence is diagnostic; a disk problem must not silence
                # live progress in Feishu.
                event.seq = self._event_count + 1
                logger.warning("persisting Run progress event failed", exc_info=True)
            self._event_count = event.seq

            async with self._lock:
                self._card.set_elapsed(int(now - self._started_at))
                if formatted.kind == "stage":
                    self._card.begin_stage(agent_name, phase)
                    self._card.latest_status = formatted.title
                elif formatted.kind == "heartbeat":
                    # Heartbeats prove liveness but are not work. Keep them in
                    # Run JSONL and update the visible status without filling
                    # the folded process panel with repeated lines.
                    self._card.latest_status = formatted.title
                else:
                    self._card.record_progress(
                        formatted.title,
                        detail=formatted.detail,
                        agent_name=agent_name,
                        phase=phase,
                        timestamp=event.created_at.astimezone().strftime("%H:%M:%S"),
                    )

                force = formatted.kind in {
                    "stage", "checkpoint", "delegate", "delegate_requested",
                    "delegate_started", "delegate_completed", "delegate_failed",
                    "artifact", "artifact_created", "artifact_failed", "heartbeat",
                    "stalled", "error",
                }
                if force or now - self._last_update >= self._update_interval:
                    delivered = await self._flush_card()
                    if not delivered and self._card_msg_id is None:
                        await self._fallback_progress(formatted.title, formatted.kind)
        except Exception:
            logger.warning("feishu progress handling failed", exc_info=True)

    async def _heartbeat_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval)
                now = time.monotonic()
                business_silence = now - self._last_business_event_at
                if business_silence >= 90 and not self._stalled_reported:
                    self._stalled_reported = True
                    await self.on_progress(
                        "stalled",
                        {
                            "agent_name": self._entry_agent_name,
                            "phase": "entry",
                            "title": "较长时间没有新的阶段进展，Agent 仍在运行",
                            "detail": f"已等待 {int(business_silence)} 秒；达到超时上限后会自动停止。",
                        },
                    )
                elif now - self._last_event_at >= self._heartbeat_interval:
                    await self.on_progress(
                        "heartbeat",
                        {
                            "agent_name": self._entry_agent_name,
                            "phase": "entry",
                            "elapsed_seconds": int(now - self._started_at),
                        },
                    )
        except asyncio.CancelledError:
            return

    async def _flush_card(self) -> bool:
        """Send the card the first time, or update it afterward."""
        if not settings.feishu_card_v2:
            self._delivery_state = "card_disabled"
            self._delivery_error = "CT_FEISHU_CARD_V2 is disabled"
            await self._notify_delivery()
            return False
        if self._card_disabled_for_run and self._card_msg_id is None:
            return False
        card = self._card.build()
        if self._card_msg_id is None:
            try:
                self._card_msg_id = await self._sender.send_card(self._message_id, card)
                if not self._card_msg_id:
                    raise RuntimeError("Feishu card reply returned no message_id")
                self._delivery_state = "card_sent"
                self._delivery_error = ""
            except Exception as exc:
                self._delivery_state = "card_failed"
                self._delivery_error = str(exc)
                logger.warning("feishu send_card failed: %s", exc, exc_info=True)
                self._card_msg_id = None
                self._card_disabled_for_run = True
            self._last_update = time.monotonic()
            await self._notify_delivery()
            return self._card_msg_id is not None
        try:
            ok = await self._sender.update_card(self._card_msg_id, card)
            self._last_update = time.monotonic()
            if ok:
                self._delivery_state = "card_updated"
                self._delivery_error = ""
            else:
                self._delivery_state = "card_update_failed"
                self._delivery_error = "update_card returned False"
                logger.warning("feishu update_card returned False")
            await self._notify_delivery()
            return bool(ok)
        except Exception as exc:
            self._delivery_state = "card_update_failed"
            self._delivery_error = str(exc)
            self._last_update = time.monotonic()
            logger.warning("feishu update_card failed: %s", exc, exc_info=True)
            await self._notify_delivery()
            return False

    async def _fallback_progress(self, title: str, kind: str) -> None:
        # Text fallback is reserved for meaningful milestones. Heartbeats and
        # ordinary tool actions would otherwise create a new Feishu message on
        # every interval when cards are unavailable.
        if kind in {"heartbeat", "action"}:
            return
        now = time.monotonic()
        min_interval = max(settings.checkpoint_min_interval, self._heartbeat_interval)
        if self._last_fallback_progress and now - self._last_fallback_progress < min_interval:
            return
        self._last_fallback_progress = now
        try:
            await self._sender.reply(
                self._message_id,
                f"⏳ [{self._entry_agent_name or 'Agent'}] {title}\nRun ID: {self._run_id}",
                reply_in_thread=True,
                request_uuid=f"{self._run_id}-progress-{self._event_count}"[:50],
            )
            self._delivery_state = "text_progress_fallback"
            await self._notify_delivery()
        except Exception as exc:
            self._delivery_error = str(exc)
            logger.warning("feishu progress fallback reply failed: %s", exc)

    async def finalize(self) -> None:
        """Stop heartbeat and publish the final answer/artifact card."""
        heartbeat, self._heartbeat_task = self._heartbeat_task, None
        if heartbeat is not None:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

        async with self._lock:
            self._card.set_elapsed(int(time.monotonic() - self._started_at))
            self._card.complete_current_stage()
            updated = await self._flush_card()
            if not updated and self._card_msg_id is not None:
                try:
                    replacement_id = await self._sender.send_card(
                        self._message_id,
                        self._card.build(),
                    )
                    if replacement_id:
                        self._card_msg_id = replacement_id
                        self._delivery_state = "replacement_card_sent"
                        self._delivery_error = ""
                        updated = True
                except Exception as exc:
                    self._delivery_error = str(exc)
                    logger.warning("feishu replacement final card failed: %s", exc, exc_info=True)
                if not updated:
                    self._card_msg_id = None
                await self._notify_delivery()

            if not updated and self._card_msg_id is None and not self._fallback_used:
                self._fallback_used = True
                text = self._fallback_text(self._card.result_content())
                try:
                    await self._sender.reply(
                        self._message_id,
                        f"{text}\n\nRun ID: {self._run_id}",
                        reply_in_thread=True,
                        request_uuid=f"{self._run_id}-final",
                    )
                    self._delivery_state = "text_final_fallback"
                    await self._notify_delivery()
                except Exception as exc:
                    self._delivery_error = str(exc)
                    logger.warning("feishu finalize fallback reply failed: %s", exc)

    async def pause(self) -> None:
        """Stop heartbeats while a parent invocation waits for background work."""
        heartbeat, self._heartbeat_task = self._heartbeat_task, None
        if heartbeat is not None:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
        async with self._lock:
            self._card.set_elapsed(int(time.monotonic() - self._started_at))
            await self._flush_card()

    async def _notify_delivery(self) -> None:
        marker = (self._delivery_state, self._delivery_error, self._card_msg_id)
        if marker == self._last_delivery_notice:
            return
        self._last_delivery_notice = marker
        if self._on_delivery:
            try:
                await self._on_delivery(self.delivery_snapshot())
            except Exception:
                logger.debug("persisting Feishu delivery state failed", exc_info=True)

    def delivery_snapshot(self) -> dict:
        return {
            "state": self._delivery_state,
            "card_message_id": self._card_msg_id,
            "last_error": self._delivery_error,
            "event_count": self._event_count,
        }

    def process_snapshot(self) -> list[dict[str, str]]:
        return self._card.process_snapshot()

    @staticmethod
    def _fallback_text(text: str) -> str:
        # FeishuSender.reply performs lossless chunking. Do not truncate the
        # terminal answer here, otherwise card failure would silently lose it.
        return text
