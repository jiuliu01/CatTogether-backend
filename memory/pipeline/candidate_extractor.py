"""Stage 2: Candidate Extraction — Event → MemoryCandidate[] (§7.1).

Extracts memory candidates from the event payload using rule-based analysis.
This module is self-contained — it does not depend on the deleted
``memory.analyzer`` module.  The rule-based extraction logic was moved here
from the old analyzer and adapted to work with the v4 ``Event`` model.

Failure handling: no candidates → event is ``skipped``.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from config import settings
from memory.models import Event
from models.schemas import MemoryCandidate, MemoryWriteEvent


logger = logging.getLogger(__name__)


_SENTENCE_RE = re.compile(r"(?<=[。！？!?\n])")


def _sentences(text: str) -> list[str]:
    return [part.strip(" \r\n-*#") for part in _SENTENCE_RE.split(text) if part.strip(" \r\n-*#")]


def _event_to_write_event(event: Event) -> MemoryWriteEvent:
    """Adapt a v4 ``Event`` to the v1 ``MemoryWriteEvent`` interface.

    The ``Event.payload`` dict carries the rich fields (user_text, final_text,
    mutation_paths, etc.) that the rule-based analyzer needs.
    """
    p = event.payload or {}
    actor = event.actor
    return MemoryWriteEvent(
        event_id=event.event_id,
        event_type=event.event_type,
        user_id=p.get("user_id", "default"),
        workspace_id=p.get("workspace_id"),
        channel_id=p.get("channel_id") or "default",
        thread_id=p.get("thread_id"),
        run_id=p.get("run_id"),
        agent_id=actor.id if actor.kind == "agent" else p.get("agent_id", "system") or "system",
        agent_role=p.get("agent_role", "custom"),
        user_text=p.get("user_text", ""),
        final_text=p.get("final_text", ""),
        mutation_paths=p.get("mutation_paths", []),
        allowed_domains=list(event.allowed_domains),
        source_message_ids=p.get("source_message_ids", []),
        succeeded=p.get("succeeded", True),
        tenant_id=event.tenant_id,
        task_id=p.get("task_id"),
    )


# ---------------------------------------------------------------------------
# Rule-based candidate extraction (moved from memory.analyzer)
# ---------------------------------------------------------------------------

class _RuleBasedAnalyzer:
    """Conservative baseline analyzer.

    Extracts user preferences/corrections, workspace changes, and agent
    practices from the event context.  This is the only extraction strategy
    in the final system — LLM extraction was removed to keep the pipeline
    deterministic and dependency-free.
    """

    async def analyze(self, event: MemoryWriteEvent) -> list[MemoryCandidate]:
        # A failed task may still have changed project files. Preserve that
        # partial state so the next run does not assume a clean workspace.
        if not event.succeeded and not event.mutation_paths:
            return []
        allowed = set(event.allowed_domains)
        candidates: list[MemoryCandidate] = []

        if event.succeeded and "user" in allowed:
            candidates.extend(self._user_candidates(event))
        if "workspace" in allowed and event.workspace_id:
            candidates.extend(self._workspace_candidates(event))
        if "project" in allowed and event.workspace_id:
            candidates.extend(self._workspace_candidates(event))
        if event.succeeded and "agent" in allowed and settings.memory_agent_auto_write:
            candidates.extend(self._agent_candidates(event))
        return candidates

    def _user_candidates(self, event: MemoryWriteEvent) -> list[MemoryCandidate]:
        text = event.user_text.strip()
        if not text or any(marker in text for marker in ("这次", "本次", "当前任务")):
            return []
        preference_markers = ("我喜欢", "我习惯", "我希望以后", "以后请", "以后都", "请记住", "不要再")
        correction_markers = ("更正", "纠正", "改为", "不再", "之前说错", "不是")
        result: list[MemoryCandidate] = []
        for sentence in _sentences(text):
            is_correction = any(marker in sentence for marker in correction_markers)
            is_preference = any(marker in sentence for marker in preference_markers)
            if not (is_correction or is_preference):
                continue
            result.append(MemoryCandidate(
                domain="user",
                kind="correction" if is_correction else "preference",
                text=sentence[:500],
                confidence=0.82 if is_correction else 0.84,
                importance=0.75,
                source_type="user_statement",
                source_ids=list(event.source_message_ids),
                tags=["correction" if is_correction else "preference"],
            ))
        return result

    def _workspace_candidates(self, event: MemoryWriteEvent) -> list[MemoryCandidate]:
        evidence = bool(event.mutation_paths)
        explicit_confirmation = any(
            marker in event.user_text for marker in ("确认采用", "决定采用", "就按", "确定使用")
        )
        if not event.succeeded and not evidence:
            return []
        if not evidence and not explicit_confirmation:
            return []

        if evidence:
            paths = "、".join(event.mutation_paths[:20])
            task = next(iter(_sentences(event.user_text)), "")[:240]
            if event.succeeded:
                prefix = "项目修改已完成"
                tags = ["completed", "workspace-change"]
            else:
                prefix = "任务未完成，但项目文件已发生变化"
                tags = ["partial", "workspace-change"]
            details = [prefix]
            if task:
                details.append(f"任务：{task}")
            details.append(f"变更文件：{paths}")
            return [MemoryCandidate(
                domain="workspace",
                kind="progress",
                text="；".join(details)[:1200],
                confidence=0.82,
                importance=0.75,
                source_type="workspace_change",
                source_ids=list(event.source_message_ids) + ([event.run_id] if event.run_id else []),
                tags=tags,
            )]

        source = event.final_text if evidence else event.user_text
        sentences = _sentences(source)
        selected = next(
            (
                sentence for sentence in sentences
                if any(marker in sentence for marker in ("已完成", "已新增", "已修改", "采用", "决定", "使用"))
            ),
            "",
        )
        if not selected:
            return []
        return [MemoryCandidate(
            domain="workspace",
            kind="progress" if evidence else "decision",
            text=selected[:600],
            confidence=0.78 if evidence else 0.75,
            importance=0.7,
            source_type="workspace_change" if evidence else "user_statement",
            source_ids=list(event.source_message_ids) + ([event.run_id] if event.run_id else []),
            tags=["completed" if evidence else "decision"],
        )]

    def _agent_candidates(self, event: MemoryWriteEvent) -> list[MemoryCandidate]:
        if event.event_type != "agent_invocation" or event.mutation_paths:
            return []
        for sentence in _sentences(event.final_text):
            if any(marker in sentence for marker in ("通用做法", "应当", "建议始终", "最佳实践")):
                return [MemoryCandidate(
                    domain="agent",
                    kind="practice",
                    text=sentence[:500],
                    confidence=0.86,
                    importance=0.65,
                    source_type="agent_result",
                    source_ids=list(event.source_message_ids),
                    tags=[event.agent_role, "practice"],
                )]
        return []


# Singleton analyzer instance.
_rule_analyzer = _RuleBasedAnalyzer()


class CandidateExtractor:
    """Stage 2: extract memory candidates from an event.

    Uses the built-in rule-based analyzer.  Any extraction failure returns
    an empty list (event will be marked ``skipped`` by the pipeline).
    """

    def __init__(self, analyzer: Any = None) -> None:
        self._analyzer = analyzer or _rule_analyzer

    async def extract(self, event: Event) -> list[MemoryCandidate]:
        """Extract candidates from *event*.

        Returns an empty list if no candidates are found (event will be
        marked ``skipped`` by the orchestrator).
        """
        write_event = _event_to_write_event(event)
        try:
            candidates = await self._analyzer.analyze(write_event)
        except Exception:
            logger.exception("candidate extraction failed for event %s", event.event_id)
            return []
        return candidates
