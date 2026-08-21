"""Memory 2.2 extractor (§5) — the bridge between the memory agent and the Qdrant store.

Given a work context (a finished agent invocation / conversation), the
extractor:

  1. derives a domain tag (project / user / agent; task reserved, not fired)
  2. assembles three context blocks for the memory agent:
       - recent conversation  ← session_messages (2.1 History layer, reused)
       - related old memories  ← MemoryStore.search (payload form)
       - the new message        ← from the trigger context directly
  3. invokes the memory agent (a ClaudeCodeAgent with the §2.4 system prompt)
     and parses the JSONL it emits
  4. post-processes each emitted memory:
       - sanitize_agent_text(text)
       - validate linked_memory_ids against the uuids we actually handed the
         agent (LLMs hallucinate ids; drop any we never offered)
       - mint the real UUID, fill domain / run_id / agent_id / ...
  5. dedups by (domain, hash) and writes survivors to Qdrant via MemoryStore

Failures never propagate: a memory-agent timeout / parse error / Qdrant outage
is logged and returns an empty result so the orchestrator keeps going.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from core.output_safety import sanitize_agent_text
from memory.db import MemoryDB, memory_db
from memory.history_store import HistoryStore
from memory.scope import MemoryDomain, tenant_id_from_user_id
from memory.v22.memory_store import MemoryStore
from memory.v22.models import Memory, memory_hash


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input / output types
# ---------------------------------------------------------------------------

@dataclass
class ExtractionContext:
    """Everything the extractor needs from a finished invocation."""
    event_type: str = "agent_invocation"          # agent_invocation / conversation_completed
    agent_id: str = "system"
    agent_name: str = ""
    user_text: str = ""
    final_text: str = ""
    mutation_paths: list[str] = field(default_factory=list)
    succeeded: bool = True
    run_id: str | None = None
    channel_id: str | None = None
    thread_id: str | None = None
    project_id: str | None = None
    user_id: str = "default"
    task_id: str | None = None                    # reserved — never populated today


@dataclass
class ExtractionResult:
    written: list[Memory] = field(default_factory=list)
    skipped_dup: int = 0
    skipped_invalid: int = 0
    failed: str | None = None                     # set when the whole extraction failed


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------

# Greedy-first brace match for the first JSON object in the agent output. The
# memory agent is instructed to output only JSON, but defensively tolerate
# leading/trailing prose or markdown fences.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_first_json(text: str) -> dict | None:
    """Pull the first ``{ ... }`` JSON object out of *text*.

    The memory agent is told to emit only JSON, but defensively tolerate:
    - leading/trailing prose
    - markdown fences
    - missing outer braces (some models emit ``"memory": [...]`` without the
      surrounding ``{`` ``}``)

    Strategy: try the balanced-brace scan first; if the result lacks the
    expected ``memory`` key (i.e. we grabbed an inner object by mistake),
    fall back to wrapping the whole text in ``{`` ``}``.
    """
    if not text:
        return None
    m = _JSON_FENCE_RE.search(text)
    if m:
        cand = m.group(1)
        try:
            return json.loads(cand)
        except ValueError:
            pass
    # Balanced-brace scan from the first ``{``.
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = text[start:i + 1]
                    try:
                        obj = json.loads(cand)
                        if isinstance(obj, dict) and "memory" in obj:
                            return obj
                    except ValueError:
                        pass
                    break  # fall through to wrap recovery
    # Recovery 1: the agent emitted the object body without surrounding
    # braces AND without the leading quote on the first key. Reconstruct by
    # prepending ``{"`` and appending ``}`` → ``{"memory":[...]}``.
    for suffix in ("}", "]}"):
        try:
            obj = json.loads('{"' + text + suffix)
            if isinstance(obj, dict) and "memory" in obj:
                return obj
        except ValueError:
            pass
    # Recovery 2: the output is truncated mid-value (CLI stream cut off).
    # Autocomplete common truncation points: an open array value, an open
    # object, an open outer object. Try progressively more closing tokens.
    for suffix in ("[]}]}", "\"]}]}", "[]}", "\"]}", "}", "]}"):
        try:
            obj = json.loads('{"' + text + suffix)
            if isinstance(obj, dict) and "memory" in obj:
                return obj
        except ValueError:
            pass
    # Recovery 3: plain brace wrap (handles cases where the leading key is
    # already quoted, e.g. ``"memory":[...]``).
    for suffix in ("}", "]}"):
        try:
            obj = json.loads("{" + text + suffix)
            if isinstance(obj, dict) and "memory" in obj:
                return obj
        except ValueError:
            pass
    return None


def _parse_memory_output(raw: str) -> list[dict]:
    """Parse the memory agent's JSONL contract into a list of raw records.

    Expected shape: ``{"memory": [{"id","text","attributed_to","linked_memory_ids"}, ...]}``.
    Missing / empty / malformed → empty list.
    """
    obj = _extract_first_json(raw)
    if not isinstance(obj, dict):
        return []
    memories = obj.get("memory")
    if not isinstance(memories, list):
        return []
    out: list[dict] = []
    for rec in memories:
        if not isinstance(rec, dict):
            continue
        text = str(rec.get("text") or "").strip()
        if not text:
            continue
        attributed = str(rec.get("attributed_to") or "assistant").strip().lower()
        if attributed not in ("user", "assistant"):
            attributed = "assistant"
        linked = rec.get("linked_memory_ids") or []
        if not isinstance(linked, list):
            linked = []
        out.append({
            "text": text,
            "attributed_to": attributed,  # type: ignore[arg-type]
            "linked_memory_ids": [str(x) for x in linked if isinstance(x, (str, int))],
        })
    return out


# ---------------------------------------------------------------------------
# Domain derivation (system decides, never the memory agent)
# ---------------------------------------------------------------------------

def _derive_domain(ctx: ExtractionContext, attributed_to: str) -> MemoryDomain:
    """Pick the domain tag for one emitted memory.

    ``attributed_to`` is the primary discriminator: a user statement is a user
    memory regardless of project context; an assistant statement is project-scoped
    if it landed in a project with work, otherwise agent-scoped.

      - task: reserved — the system has no task entity today (AgentRun has no
        task_id), so this branch is left in place but never fires.
      - user: attributed_to=user (a user preference / correction / statement).
      - project: attributed_to=assistant AND there's a project_id AND the
        invocation either succeeded or touched files.
      - agent: attributed_to=assistant without qualifying project work.
    """
    # task: reserved (never fires today — no task entity upstream).
    if ctx.task_id:
        return "task"  # type: ignore[return-value]
    if attributed_to == "user":
        return "user"
    if ctx.project_id and (ctx.succeeded or ctx.mutation_paths):
        return "project"
    return "agent"


# ---------------------------------------------------------------------------
# Context assembly for the memory agent
# ---------------------------------------------------------------------------

def _format_recent_turns(rows: list) -> str:
    """Render recent session_messages (oldest → newest) as a transcript block."""
    if not rows:
        return "（无最近对话）"
    lines: list[str] = []
    for r in rows:
        role = getattr(r, "role", "")
        who = getattr(r, "agent_id", "") or ("user" if role == "user" else role)
        content = (getattr(r, "content", "") or "").strip().replace("\n", " ")
        if len(content) > 600:
            content = content[:600] + "…"
        lines.append(f"[{who}] {content}")
    return "\n".join(lines)


def _format_related_memories(results) -> str:
    """Render related-old-memory hits in PAYLOAD form for the memory agent.

    Delegates to ``MemoryRenderer.for_extraction`` (§6.2) so the rendering
    contract lives in one place.
    """
    from memory.v22.render import MemoryRenderer
    return MemoryRenderer.for_extraction(results)


def _build_prompt(
    ctx: ExtractionContext,
    recent_turns_text: str,
    related_block: str,
) -> str:
    """Assemble the user-message portion of the memory-agent prompt.

    The agent's system_prompt (§2.4 JSONL contract) is supplied by the
    ClaudeCodeAgent executor from its spec; this is the per-call payload.
    """
    return (
        "请把下面这段 agent 工作上下文提纯成记忆，按你的输出格式输出 JSON。\n\n"
        "【最近对话】\n"
        f"{recent_turns_text}\n\n"
        "【已有相关 memory（供引用，linked_memory_ids 只能引用这里出现的 uuid）】\n"
        f"{related_block}\n\n"
        "【待提纯上下文】\n"
        f"触发事件：{ctx.event_type}\n"
        f"来源 agent：{ctx.agent_id}"
        f"{f'（{ctx.agent_name}）' if ctx.agent_name else ''}\n"
        f"用户消息：{ctx.user_text}\n"
        f"agent 最终回复：{ctx.final_text}\n"
        f"文件改动：{', '.join(ctx.mutation_paths) if ctx.mutation_paths else '无'}\n"
        f"是否成功：{ctx.succeeded}\n"
        f"run_id：{ctx.run_id or '-'}\n"
    )


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class MemoryExtractor:
    """Bridges the memory agent and the Qdrant store.

    Construct once per process; ``extract`` is called for each finished
    invocation. Heavy collaborators (store, history store, the memory agent
    instance) are injected so tests can stub them.
    """

    def __init__(
        self,
        store: MemoryStore,
        memory_agent: Any,                       # BaseAgent (memory)
        history: HistoryStore | None = None,
        db: MemoryDB | None = None,
        *,
        recent_turns: int = 10,
        related_top_k: int = 5,
    ) -> None:
        self._store = store
        self._agent = memory_agent
        self._history = history or HistoryStore(db or memory_db)
        self._db = db or memory_db
        self._recent_turns = recent_turns
        self._related_top_k = related_top_k
        self._last_usage: dict | None = None  # set by extract() for token accounting

    async def extract(self, ctx: ExtractionContext) -> ExtractionResult:
        result = ExtractionResult()
        logger.info(
            "[v22-trigger] event=%s agent=%s run=%s channel=%s | user=%s final=%s",
            ctx.event_type, ctx.agent_id, ctx.run_id, ctx.channel_id,
            (ctx.user_text or "")[:80], (ctx.final_text or "")[:80],
        )
        try:
            # 1. Pull recent conversation turns from the 2.1 History layer.
            tenant_id = tenant_id_from_user_id(ctx.user_id)
            recent: list = []
            if ctx.channel_id:
                try:
                    recent = self._history.list_messages(
                        tenant_id=tenant_id,
                        channel_id=ctx.channel_id,
                        thread_id=ctx.thread_id,
                        limit=self._recent_turns,
                    )
                    # list_messages returns newest-first (ORDER BY seq DESC);
                    # show oldest→newest so the agent reads in order.
                    recent = list(reversed(recent))
                except Exception:
                    logger.exception("recent-turns fetch failed; continuing without them")
                    recent = []
            recent_text = _format_recent_turns(recent)

            # 2. Search related old memories in the project/user/agent scope.
            query_text = (ctx.user_text + "\n" + ctx.final_text).strip() or ctx.user_text
            related: list = []
            offered_uuids: set[str] = set()
            try:
                related = self._store.search(
                    query_text,
                    domains=["project", "user", "agent"],
                    top_k=self._related_top_k,
                )
                offered_uuids = {h.memory.id for h in related}
            except Exception:
                logger.exception("related-memory search failed; continuing without them")
            related_block = _format_related_memories(related)

            # 3. Invoke the memory agent.
            raw_output, usage = await self._invoke_memory_agent(ctx, recent_text, related_block)
            # Surface the latest usage for callers that track token costs (run_e2e).
            # Not stored on ExtractionResult per the eval design — the caller reads
            # this attribute right after extract() returns.
            self._last_usage = usage
            if raw_output is None:
                result.failed = "memory_agent_invocation_failed"
                return result

            # 4. Parse the JSONL.
            records = _parse_memory_output(raw_output)
            if not records:
                # Empty output is a valid "nothing worth remembering".
                return result

            # 5. Post-process + dedup + write.
            for rec in records:
                safe = sanitize_agent_text(rec["text"]).text.strip()
                if not safe:
                    result.skipped_invalid += 1
                    continue
                domain = _derive_domain(ctx, rec["attributed_to"])
                # Validate linked ids against what we actually offered.
                clean_linked = [u for u in rec["linked_memory_ids"] if u in offered_uuids]

                # Dedup by (domain, text).
                try:
                    if self._store.exists_by_hash(domain, safe):
                        result.skipped_dup += 1
                        continue
                except Exception:
                    logger.exception("dedup check failed; writing anyway")

                mem = Memory(
                    id=uuid.uuid4().hex,
                    text=safe,
                    domain=domain,
                    attributed_to=rec["attributed_to"],  # type: ignore[arg-type]
                    linked_memory_ids=clean_linked,
                    run_id=ctx.run_id,
                    channel_id=ctx.channel_id,
                    thread_id=ctx.thread_id,
                    agent_id=ctx.agent_id,
                    event_type=ctx.event_type,
                    hash=memory_hash(safe),
                )
                try:
                    self._store.upsert(mem)
                    result.written.append(mem)
                    logger.info(
                        "[v22-written] id=%s domain=%s attr=%s run=%s | %s",
                        mem.id, mem.domain, mem.attributed_to, mem.run_id,
                        (mem.text or "")[:120],
                    )
                except Exception:
                    logger.exception("memory upsert failed for one record; skipping")
                    result.skipped_invalid += 1
        except Exception:
            logger.exception("memory extraction failed (non-fatal)")
            result.failed = result.failed or "extraction_error"
        logger.info(
            "[v22-done] written=%d dup=%d invalid=%d failed=%s | run=%s agent=%s",
            len(result.written), result.skipped_dup, result.skipped_invalid,
            result.failed, ctx.run_id, ctx.agent_id,
        )
        return result

    async def _invoke_memory_agent(
        self,
        ctx: ExtractionContext,
        recent_text: str,
        related_block: str,
    ) -> tuple[str | None, dict | None]:
        """Run the memory agent and collect its full final output as one string.

        The agent is driven by its spec (ApiMemoryAgent for the API backend, or
        ClaudeCodeAgent for the legacy CLI backend). We feed the assembled prompt
        via InvokeContext and consume the ``final_text`` event; any narration
        before it is discarded. The ``done`` event (emitted by ApiMemoryAgent)
        carries a ``usage`` block for token accounting — we return it alongside
        the text so the caller can accumulate token usage without storing it on
        ExtractionResult (per the eval design, token stats live in run_e2e).
        """
        from agents.base import InvokeContext

        prompt = _build_prompt(ctx, recent_text, related_block)
        agent_ctx = InvokeContext(
            channel_id=ctx.channel_id or "",
            user_message=prompt,
            user_id=ctx.user_id,
            workspace_id=ctx.project_id,
            thread_id=ctx.thread_id,
            agent_role="memory",
            memory="",                       # no injected context — we feed it as the message
            workspace_dir="",                # read-only; no workspace for the extractor
            workspace_access="read",
            spec=getattr(self._agent, "_spec", None),
            project_id=ctx.project_id,
            timeout=getattr(getattr(self._agent, "_spec", None), "timeout", 120),
        )
        final_text: str | None = None
        usage: dict | None = None
        parts: list[str] = []
        try:
            async for etype, data in self._agent.invoke(agent_ctx):
                if etype == "final_text":
                    final_text = str(data.get("text") or "").strip()
                elif etype == "text_delta":
                    # Some flows never emit final_text; accumulate narration as
                    # a fallback so a JSON buried in deltas is still parseable.
                    parts.append(str(data.get("delta") or ""))
                elif etype == "done":
                    # ApiMemoryAgent surfaces token usage here (CLI backend has none).
                    usage = data.get("usage") or None
                elif etype == "error":
                    logger.warning("memory agent error: %s", data.get("message", ""))
                    return None, None
        except Exception:
            logger.exception("memory agent invocation raised")
            return None, None
        return final_text or "".join(parts).strip() or None, usage
