"""Agent abstraction. All agents (CLI / LLM / custom) implement BaseAgent.

The unified contract is a single async generator that yields AgentEvent-like
dicts. The orchestrator stamps seq/channel_id and publishes to the event bus.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal

from models.schemas import AgentInfo, Message


@dataclass
class InvokeContext:
    channel_id: str
    user_message: str
    user_id: str = "default"
    workspace_id: str | None = None
    thread_id: str | None = None
    agent_role: str = "custom"
    history: list[Message] = field(default_factory=list)
    memory: str = ""           # pre-built memory prompt string
    workspace_dir: str = ""
    workspace_access: Literal["read", "write"] = "write"
    mentioned_agents: list[str] = field(default_factory=list)
    cancel_event: object | None = None  # asyncio.Event
    timeout: float | None = None  # override settings.cli_timeout for this invoke
    spec: object | None = None  # AgentSpec; when set, drives the CLI executor
    project_id: str | None = None
    delegate_token: str | None = None  # token issued for this agent run (MCP)
    mcp_config_path: str | None = None  # path to per-run .mcp.json
    roster: str = ""  # 同事名单 injected into the prompt
    mcp_capabilities: list[str] = field(default_factory=list)
    output_intent: dict = field(default_factory=dict)
    root_run_id: str | None = None
    invocation_id: str | None = None
    parent_invocation_id: str | None = None


# A raw event yielded by agents: (type, data). The orchestrator wraps it.
# `final_text` is an internal collection hint used by CLI adapters that can
# distinguish the final answer from tool-call narration. The orchestrator
# consumes it without publishing it on the public WebSocket event stream.
RawEvent = tuple[
    Literal[
        "text_delta",
        "final_text",
        "tool_call",
        "tool_result",
        "status",
        "done",
        "error",
        "recoverable_error",
    ],
    dict,
]


class BaseAgent(ABC):
    def __init__(self, agent_id: str, name: str, kind: str, description: str = "") -> None:
        self.agent_id = agent_id
        self.name = name
        self.kind = kind  # type: ignore[assignment]
        self.description = description
        self.status: str = "idle"

    def info(self) -> AgentInfo:
        return AgentInfo(
            id=self.agent_id, name=self.name, kind=self.kind,  # type: ignore[arg-type]
            status=self.status, description=self.description,  # type: ignore[arg-type]
        )

    @abstractmethod
    async def invoke(self, ctx: InvokeContext) -> AsyncIterator[RawEvent]:
        """Yield raw events. Implementations MUST yield ('done', {}) on success
        or ('error', {...}) on failure as the terminal event."""
        ...
        yield  # pragma: no cover  # make it a generator for type checkers

    async def health(self) -> bool:
        return True
