"""Context Builder (§9) — assembles four layers into a MemoryContext.

Assembly order (§9.2):
    1. Authenticate Actor and tenant.
    2. Generate AllowedScopes.
    3. Read Hot (fixed budget).
    4. Retrieve Fact (main budget).
    5. Only for backtracking intents: read History.
    6. Only when procedural guidance needed: load Skill.
    7. Add source and trust labels.
    8. Prompt injection isolation.
    9. Truncate to token_budget.

Default budget (§9.3):
    Hot     — fixed retention, never fully displaced
    Fact    — main retrieval budget
    History — default 0, enabled by intent
    Skill   — default 0, enabled by task need

History and Skill content are data references, never system instructions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from config import settings
from memory.db import MemoryDB, memory_db
from memory.layers.fact_store import FactStore
from memory.layers.history_store import HistoryStoreWrapper
from memory.layers.hot_store import HotStore
from memory.layers.skill_store import SkillStoreWrapper
from memory.models import (
    AllowedScope as _AllowedScopeModel,
    HotMemoryItem, MemoryIntent, RetrievalQuery, RetrievalResult,
    SessionMessage, SkillMetadata,
)
from memory.permissions import ActorContext, compile_allowed_scopes
from memory.scope import MemoryScope
from memory.token_counter import ApproximateTokenCounter


logger = logging.getLogger(__name__)
_token_counter = ApproximateTokenCounter()


# Intents that trigger History loading.
_HISTORY_INTENTS: set[str] = {"continue_task", "reflect", "audit"}

# Intents that trigger Skill loading.
_SKILL_INTENTS: set[str] = {"start_task", "continue_task"}


def infer_memory_intent(query: str, requested: MemoryIntent = "recall") -> MemoryIntent:
    """Infer a retrieval intent when the caller only supplied ``recall``.

    The orchestrator does not run a separate LLM classifier before context
    assembly.  A small deterministic router keeps obvious backtracking
    requests ("what did we do before", "continue last task", etc.) from being
    treated as ordinary factual lookups.
    """
    if requested != "recall":
        return requested
    text = (query or "").strip().lower()
    if any(token in text for token in (
        "继续", "接着", "接上", "恢复", "上次做到", "continue", "resume",
    )):
        return "continue_task"
    if any(token in text for token in (
        "回想", "回顾", "之前", "以前", "上次", "做过", "干了", "历史",
        "remember", "recall", "previously", "last time", "what did",
    )):
        return "reflect"
    if any(token in text for token in (
        "审计", "追溯", "谁改", "变更记录", "audit", "trace",
    )):
        return "audit"
    return requested


@dataclass
class MemoryContext:
    """The assembled context from all four layers.

    ``rendered`` is the final text to inject into the prompt.  The
    individual layer results are also available for inspection.
    """
    rendered: str = ""
    hot_items: list[HotMemoryItem] = field(default_factory=list)
    fact_results: list[RetrievalResult] = field(default_factory=list)
    history_messages: list[SessionMessage] = field(default_factory=list)
    skills: list[SkillMetadata] = field(default_factory=list)
    token_count: int = 0
    budget_used: dict[str, int] = field(default_factory=dict)


def _trust_label(item: Any) -> str:
    """Generate a trust label for a fact/hot item."""
    prov = getattr(item, "provenance", None)
    if prov is not None:
        extractor = getattr(prov, "extractor", "rule")
        confidence = getattr(prov, "extraction_confidence", 0.5)
        if extractor == "manual":
            return "[trusted:manual]"
        if extractor == "llm" and confidence >= 0.8:
            return "[trusted:llm-high]"
        if confidence < 0.5:
            return "[unverified]"
    return ""


def _isolate_injection(text: str) -> str:
    """Prompt injection isolation (§9.2 step 8).

    Wraps memory content in markers and sanitises control sequences.
    This is a defence-in-depth measure — the LLM should treat content
    between markers as data, not instructions.
    """
    if not text:
        return ""
    # Remove null bytes and other control characters (except newlines/tabs).
    cleaned = "".join(
        c for c in text
        if c == "\n" or c == "\t" or ord(c) >= 0x20
    )
    return cleaned


class ContextBuilder:
    """Assembles four layers into a MemoryContext (§9).

    Usage::

        builder = ContextBuilder()
        ctx = await builder.build_memory_context(
            actor=actor,
            scope=scope,
            query="what database do we use",
            intent="recall",
            token_budget=2000,
        )
        prompt += ctx.rendered
    """

    def __init__(
        self,
        db: MemoryDB | None = None,
        *,
        hot_store: HotStore | None = None,
        fact_store: FactStore | None = None,
        history_store: HistoryStoreWrapper | None = None,
        skill_store: SkillStoreWrapper | None = None,
    ) -> None:
        self._db = db or memory_db
        self._hot = hot_store or HotStore(self._db)
        self._facts = fact_store or FactStore(self._db)
        self._history = history_store or HistoryStoreWrapper(self._db)
        self._skills = skill_store or SkillStoreWrapper(self._db)

    async def build_memory_context(
        self,
        actor: ActorContext,
        scope: MemoryScope,
        query: str,
        intent: MemoryIntent = "recall",
        token_budget: int = 2000,
        *,
        channel_id: str | None = None,
        thread_id: str | None = None,
    ) -> MemoryContext:
        """Build the memory context from all four layers.

        Parameters
        ----------
        actor
            Authenticated actor (tenant_id, role, actor_id).
        scope
            The memory scope (domain, scope_id, tenant_id).
        query
            The retrieval query text.
        intent
            Retrieval intent — controls which layers are loaded.
        token_budget
            Total token budget for the rendered context.

        Returns
        -------
        MemoryContext
            The assembled context with ``rendered`` text and layer results.
        """
        resolved_intent = infer_memory_intent(query, intent)
        budget_used: dict[str, int] = {}
        sections: list[tuple[str, str]] = []

        # --- Step 1-2: Actor + AllowedScopes ---
        perm_scopes = compile_allowed_scopes(
            actor, actor.tenant_id,
            domain=scope.domain, scope_id=scope.scope_id,
        )
        allowed_scopes = [
            _AllowedScopeModel(tenant_id=actor.tenant_id, domain=s.domain, scope_id=s.scope_id)
            for s in perm_scopes
        ]

        # --- Step 3: Hot (fixed budget) ---
        hot_budget = min(200, token_budget // 4)  # 25% or 200 tokens, whichever is smaller
        hot_items = self._hot.get_for_scope(
            tenant_id=actor.tenant_id,
            domain=scope.domain,
            scope_id=scope.scope_id,
        )
        hot_text = self._render_hot(hot_items)
        hot_text = _token_counter.truncate(hot_text, hot_budget)
        if hot_text:
            sections.append(("Hot Memory", hot_text))
        budget_used["hot"] = _token_counter.count(hot_text)

        # --- Step 4: Fact (main budget) ---
        remaining_budget = token_budget - budget_used["hot"]
        fact_budget = int(remaining_budget * 0.7)  # 70% of remaining
        fact_results: list[RetrievalResult] = []

        # Memory 2.2: when the switch is on, serve facts from Qdrant (flat
        # Memory records, domain as a filter tag) instead of the v2.1 SQLite
        # FactStore. Hot/History/Skill layers keep their v2.1 behavior.
        from config import settings
        v22_facts: list = []
        if settings.memory_v22_enabled and query.strip():
            try:
                from memory.v22 import get_memory_store
                v22_store = get_memory_store()
                if v22_store is not None:
                    v22_hits = v22_store.search(
                        query,
                        domains=[scope.domain] if scope.domain else None,
                        top_k=settings.memory22_top_k,
                        weight_vector=settings.memory22_rrf_weight_vector,
                        weight_bm25=settings.memory22_rrf_weight_bm25,
                    )
                    v22_facts = [h.memory for h in v22_hits]
            except Exception:
                logger.exception("context builder: v22 fact search failed")

        if v22_facts:
            # Render the flat 2.2 memories as text for the working agent.
            fact_text = self._render_v22_memories(v22_facts)
            fact_text = _token_counter.truncate(fact_text, fact_budget)
            if fact_text:
                sections.append(("Facts", fact_text))
            budget_used["facts"] = _token_counter.count(fact_text)
        else:
            # v2.1 fallback: SQLite FactStore search.
            if query.strip():
                rq = RetrievalQuery(
                    text=query,
                    tenant_id=actor.tenant_id,
                    allowed_scopes=list(allowed_scopes),
                    intent=resolved_intent,
                    domain=scope.domain,
                    scope_id=scope.scope_id,
                    top_k=20,
                )
                try:
                    fact_results = self._facts.search(rq)
                except Exception:
                    logger.exception("context builder: fact search failed")
            # Vague backtracking questions often have no lexical overlap with the
            # stored work summary.  When semantic retrieval is unavailable, fall
            # back to the most recent facts in the same project/agent scope.
            if not fact_results and resolved_intent in _HISTORY_INTENTS:
                recent_facts = self._facts.list_by_scope(
                    actor.tenant_id, scope.domain, scope.scope_id, limit=8,
                )
                fact_results = [
                    RetrievalResult(
                        fact=fact,
                        score=0.0,
                        explanation={"fallback": "recent_scope"},
                    )
                    for fact in recent_facts
                ]
            fact_text = self._render_facts(fact_results)
            fact_text = _token_counter.truncate(fact_text, fact_budget)
            if fact_text:
                sections.append(("Facts", fact_text))
            budget_used["facts"] = _token_counter.count(fact_text)

        # --- Step 5: History (by intent) ---
        history_messages: list[SessionMessage] = []
        if resolved_intent in _HISTORY_INTENTS and channel_id:
            history_budget = int(remaining_budget * 0.15)  # 15% of remaining
            try:
                targets: list[tuple[str, str | None]] = [(channel_id, thread_id)]
                if scope.domain == "project" and scope.agent_id:
                    legacy_target = (scope.scope_id, f"legacy-agent:{scope.agent_id}")
                    if legacy_target not in targets:
                        targets.append(legacy_target)

                seen: set[str] = set()
                for target_channel, target_thread in targets:
                    found = self._history.search(
                        tenant_id=actor.tenant_id,
                        query=query,
                        channel_id=target_channel,
                        thread_id=target_thread,
                        limit=10,
                    )
                    # Natural-language requests such as "what did we do
                    # before" may share no keywords with the answer.  In a
                    # backtracking intent, recent messages are the safe and
                    # useful fallback.
                    if not found:
                        found = self._history.list_messages(
                            tenant_id=actor.tenant_id,
                            channel_id=target_channel,
                            thread_id=target_thread,
                            limit=10,
                        )
                    for message in found:
                        if message.message_id not in seen:
                            seen.add(message.message_id)
                            history_messages.append(message)
                history_messages = history_messages[:10]
            except Exception:
                logger.exception("context builder: history search failed")
            history_text = self._render_history(history_messages)
            history_text = _token_counter.truncate(history_text, history_budget)
            if history_text:
                sections.append(("History", history_text))
            budget_used["history"] = _token_counter.count(history_text)
        else:
            budget_used["history"] = 0

        # --- Step 6: Skill (by intent) ---
        skills: list[SkillMetadata] = []
        if resolved_intent in _SKILL_INTENTS:
            skill_budget = int(remaining_budget * 0.15)  # 15% of remaining
            try:
                skills = self._skills.list(
                    tenant_id=actor.tenant_id,
                    domain=scope.domain,
                    scope_id=scope.scope_id,
                    limit=5,
                )
            except Exception:
                logger.exception("context builder: skill list failed")
            skill_text = self._render_skills(skills)
            skill_text = _token_counter.truncate(skill_text, skill_budget)
            if skill_text:
                sections.append(("Skills", skill_text))
            budget_used["skills"] = _token_counter.count(skill_text)
        else:
            budget_used["skills"] = 0

        # --- Steps 7-8: Trust labels + injection isolation ---
        rendered = self._render_sections(sections)
        rendered = _isolate_injection(rendered)

        # --- Step 9: Final truncation to token_budget ---
        rendered = _token_counter.truncate(rendered, token_budget)

        return MemoryContext(
            rendered=rendered,
            hot_items=hot_items,
            fact_results=fact_results,
            history_messages=history_messages,
            skills=skills,
            token_count=_token_counter.count(rendered),
            budget_used=budget_used,
        )

    # ----- Rendering helpers -----

    def _render_hot(self, items: list[HotMemoryItem]) -> str:
        lines: list[str] = []
        for item in items:
            label = _trust_label(item)
            text = _isolate_injection(item.text)
            lines.append(f"- [priority={item.priority}] {label} {text}".rstrip())
        return "\n".join(lines)

    def _render_facts(self, results: list[RetrievalResult]) -> str:
        lines: list[str] = []
        for r in results:
            label = _trust_label(r.fact)
            text = _isolate_injection(r.fact.text)
            signals = ",".join(r.signals) if r.signals else "none"
            lines.append(f"- [{r.fact.kind}|{signals}] {label} {text}".rstrip())
        return "\n".join(lines)

    def _render_v22_memories(self, memories: list) -> str:
        """Render flat 2.2 Memory records as text for a working agent.

        Delegates to ``MemoryRenderer.for_prompt`` (§6.2) so the rendering
        contract lives in one place.
        """
        from memory.v22.render import MemoryRenderer
        from memory.v22.models import MemorySearchResult
        # context_builder receives raw Memory objects; wrap as results with
        # a neutral score/signals so the renderer's signature stays uniform.
        wrapped = [
            MemorySearchResult(memory=m, score=0.0, signals=[])
            for m in memories
        ]
        return MemoryRenderer.for_prompt(wrapped)

    def _render_history(self, messages: list[SessionMessage]) -> str:
        lines: list[str] = []
        for msg in messages:
            text = _isolate_injection(msg.content)
            lines.append(f"- [{msg.role}] {text}".rstrip())
        return "\n".join(lines)

    def _render_skills(self, skills: list[SkillMetadata]) -> str:
        lines: list[str] = []
        for skill in skills:
            text = _isolate_injection(skill.body)
            lines.append(f"- [skill:{skill.name}|v{skill.version}] {text}".rstrip())
        return "\n".join(lines)

    def _render_sections(self, sections: list[tuple[str, str]]) -> str:
        parts: list[str] = []
        for title, content in sections:
            if content.strip():
                parts.append(f"## {title}\n{content}")
        return "\n\n".join(parts)
