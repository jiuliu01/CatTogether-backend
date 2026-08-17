"""Memory 2.2 renderers (§6.2) — the two-form output contract.

The same ``MemoryStore.search`` result is rendered differently depending on
who consumes it:

- **extraction form (payload)** — for the memory agent. Carries uuid +
  text + domain + attributed_to + linked_memory_ids so the agent can cite
  existing uuids in its ``linked_memory_ids`` output.
- **prompt form (text)** — for working agents via ``context_builder``.
  Surfaces only text + light domain/attributed labels; uuids and link graphs
  are internal bookkeeping, not prompt material.
- **tool form (dict)** — for the ``memory_search`` MCP tool result. Returns
  a list of dicts with text as the body and uuid/links as metadata so a
  working agent calling the tool gets structured data without being flooded.

Centralizing these here keeps the three call sites (extractor,
context_builder, /internal/memory/search) from drifting apart and lets us
unit-test the contract directly.
"""
from __future__ import annotations

from memory.v22.models import MemorySearchResult


class MemoryRenderer:
    """Centralized rendering of v22 search results.

    All three forms consume the same ``list[MemorySearchResult]`` and never
    mutate it. Text is sanitised upstream; renderers only format.
    """

    # ------------------------------------------------------------------
    # extraction form — payload, for the memory agent
    # ------------------------------------------------------------------
    @staticmethod
    def for_extraction(results: list[MemorySearchResult]) -> str:
        """Payload form: uuid + text + domain + attributed_to + linked ids.

        The memory agent reads this block and may cite any uuid that appears
        here in its ``linked_memory_ids`` output. ``_format_related_memories``
        in the extractor delegates here.
        """
        if not results:
            return "（无相关旧 memory）"
        lines: list[str] = []
        for h in results:
            m = h.memory
            lines.append(
                f"- uuid={m.id} | domain={m.domain} | "
                f"attributed_to={m.attributed_to} | "
                f"linked={m.linked_memory_ids} | text={m.text[:200]}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # prompt form — text, for working agents (context_builder)
    # ------------------------------------------------------------------
    @staticmethod
    def for_prompt(results: list[MemorySearchResult]) -> str:
        """Text form for direct prompt injection.

        Only ``text`` plus light ``[domain|attributed_to]`` labels are
        surfaced; uuids and the link graph are internal. ``_render_v22_memories``
        in the context builder delegates here.
        """
        lines: list[str] = []
        for h in results:
            m = h.memory
            text = (m.text or "").rstrip()
            if not text:
                continue
            lines.append(f"- [{m.domain}|{m.attributed_to}] {text}".rstrip())
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # tool form — dict, for the memory_search MCP tool
    # ------------------------------------------------------------------
    @staticmethod
    def for_tool(results: list[MemorySearchResult]) -> list[dict]:
        """Tool-result form for ``/internal/memory/search``.

        Each dict carries text as the body and uuid/links/score as metadata
        so a working agent calling ``memory_search`` gets structured data it
        can reason over without a payload flood.
        """
        out: list[dict] = []
        for h in results:
            m = h.memory
            out.append({
                "id": m.id,
                "text": m.text,
                "domain": m.domain,
                "attributed_to": m.attributed_to,
                "linked_memory_ids": list(m.linked_memory_ids),
                "run_id": m.run_id,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "score": round(h.score, 4),
                "signals": list(h.signals),
            })
        return out
