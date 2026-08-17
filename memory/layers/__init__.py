"""Four-layer data stores (§5.8-5.10, §9).

Layers:
    Hot     — small, stable, always-injected (§5.8)
    Fact    — long-term factual knowledge, three-signal retrieval (§5.2)
    History — session messages, backtracking (§5.9)
    Skill   — reusable procedures, versioned (§5.10)

The context builder (§9) assembles these layers into a MemoryContext
with budget allocation: Hot (fixed) → Fact (main) → History/Skill (by intent).
"""
from __future__ import annotations

from memory.layers.fact_store import FactStore
from memory.layers.history_store import HistoryStore
from memory.layers.hot_store import HotStore
from memory.layers.skill_store import SkillStore

__all__ = ["HotStore", "FactStore", "HistoryStore", "SkillStore"]
