"""AgentSpec: the executable specification for one agent.

All agents share a single execution backend (a Claude Code CLI subprocess).
Differences between agents are captured entirely in this spec: persona via
system_prompt, capabilities via allowed_tools/sandbox, and behavior via
auto_trigger/delegatable/trigger_keywords.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field


def now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryConfig(BaseModel):
    enabled: bool = True
    top_k: int = 5
    carry_on_transplant: bool = True


class AgentSpec(BaseModel):
    agent_id: str
    name: str
    project_id: str
    role: str = "custom"
    system_prompt: str = ""
    # `builtin_tools` controls which Claude built-in tools actually exist via
    # `--tools`. `allowed_tools` is kept for backward-compatible project JSON
    # and as the source for automatic migration when builtin_tools is absent.
    allowed_tools: list[str] = Field(default_factory=lambda: ["Read", "Glob", "Grep"])
    builtin_tools: list[str] | None = None
    mcp_capabilities: list[str] = Field(default_factory=lambda: [
        "delegate", "report_progress",
        "wiki_read", "wiki_list", "wiki_write", "wiki_append",
        "memory_search",
    ])
    disallowed_tools: list[str] = Field(default_factory=lambda: ["Bash"])
    sandbox: Literal["read", "workspace-write"] = "read"
    auto_trigger: bool = True
    delegatable: bool = True
    max_turns: int = 30
    timeout: float = 600
    trigger_keywords: list[str] = Field(default_factory=list)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    internal: bool = False
    """Internal agents (e.g. the memory extractor) are excluded from the roster,
    cannot be @-mentioned, and cannot be pulled via the delegate tool. They run
    only as system-triggered hooks."""
    platform: Literal["claude-cli"] = "claude-cli"
    """Execution backend selector. ``claude-cli`` reuses the ClaudeCodeAgent
    subprocess adapter (the only backend today); future backends slot in here."""
    origin: Literal["generated", "transplanted", "template"] = "generated"
    created_at: datetime = Field(default_factory=now)

    @property
    def effective_builtin_tools(self) -> list[str]:
        """Role-scoped Claude tools, compatible with pre-migration specs."""
        return list(self.builtin_tools if self.builtin_tools is not None else self.allowed_tools)


class AgentSpecUpdateRequest(BaseModel):
    """Partial update for the 制定 (specify) operation."""
    name: str | None = None
    role: str | None = None
    system_prompt: str | None = None
    allowed_tools: list[str] | None = None
    builtin_tools: list[str] | None = None
    mcp_capabilities: list[str] | None = None
    disallowed_tools: list[str] | None = None
    sandbox: Literal["read", "workspace-write"] | None = None
    auto_trigger: bool | None = None
    delegatable: bool | None = None
    max_turns: int | None = None
    timeout: float | None = None
    trigger_keywords: list[str] | None = None
