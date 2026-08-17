"""Bootstrap: register the enabled built-in agent at startup.

Claude Code is the only built-in agent enabled by default. The other adapter
implementations remain available in the source tree for a future opt-in.

Memory 2.2 also registers the internal memory-extraction agent here so it is
available for the post-invocation hook. It is excluded from the roster.
"""
from __future__ import annotations

from core.registry import registry
from agents.cli.claude_code import ClaudeCodeAgent
from agents.memory_agent import register_memory_agent


def register_builtins() -> None:
    # Remove built-ins that older versions registered in the same process.
    for agent_id in ("codex", "openai-default", "anthropic-default"):
        registry.unregister(agent_id)
    registry.register(ClaudeCodeAgent())
    register_memory_agent()
