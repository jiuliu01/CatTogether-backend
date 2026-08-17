"""LLM-backed agent: assembles memory + history + user message and streams text."""
from __future__ import annotations

from typing import AsyncIterator

from agents.base import BaseAgent, InvokeContext, RawEvent
from agents.llm import providers


class LLMAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        name: str,
        provider: str,            # "openai" | "anthropic"
        model: str,
        system_prompt: str = "",
        description: str = "",
    ) -> None:
        super().__init__(agent_id, name, "llm", description)
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt

    async def health(self) -> bool:
        from config import settings
        if self.provider == "openai":
            return bool(settings.openai_api_key)
        if self.provider == "anthropic":
            return bool(settings.anthropic_api_key)
        return False

    def _build_messages(self, ctx: InvokeContext) -> list[dict]:
        msgs: list[dict] = []
        for m in ctx.history:
            if m.role == "user":
                msgs.append({"role": "user", "content": m.content})
            elif m.role == "agent":
                msgs.append({"role": "assistant", "content": m.content})
        # ensure last is the current user message
        if not msgs or msgs[-1]["content"] != ctx.user_message:
            msgs.append({"role": "user", "content": ctx.user_message})
        return msgs

    async def invoke(self, ctx: InvokeContext) -> AsyncIterator[RawEvent]:
        system = self.system_prompt
        if ctx.memory:
            system = f"{system}\n\n{ctx.memory}" if system else ctx.memory
        messages = self._build_messages(ctx)
        try:
            if self.provider == "openai":
                stream = providers.stream_openai(messages, self.model, system=system)
            else:
                stream = providers.stream_anthropic(messages, self.model, system=system)
            got_any = False
            async for delta in stream:
                if not delta:
                    continue
                got_any = True
                yield ("text_delta", {"delta": delta})
            if not got_any:
                yield ("error", {"message": f"no API key configured for {self.provider}"})
                return
            yield ("done", {})
        except Exception as e:
            yield ("error", {"message": str(e)})
