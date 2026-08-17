"""Custom user-defined agent: loads a Python function at runtime.

Convention: the module path "pkg.mod:run" exposes
    async def run(ctx: InvokeContext) -> AsyncIterator[RawEvent]
Used for non-LLM agents (rules, scrapers, calculators, ...).
"""
from __future__ import annotations

import importlib
from typing import AsyncIterator

from agents.base import BaseAgent, InvokeContext, RawEvent


class CustomAgent(BaseAgent):
    def __init__(self, agent_id: str, name: str, module_path: str, description: str = "") -> None:
        super().__init__(agent_id, name, "custom", description)
        self.module_path = module_path
        self._run = None

    def _load(self):
        if self._run is not None:
            return self._run
        mod_name, _, func_name = self.module_path.partition(":")
        if not func_name:
            func_name = "run"
        mod = importlib.import_module(mod_name)
        self._run = getattr(mod, func_name)
        return self._run

    async def invoke(self, ctx: InvokeContext) -> AsyncIterator[RawEvent]:
        try:
            run = self._load()
            async for ev in run(ctx):
                yield ev
        except Exception as e:
            yield ("error", {"message": str(e)})
