"""Memory extraction agent (Memory 2.2, §2).

A special internal agent that distills work context into structured memory
records. Unlike the four "working" cats (coordinator/researcher/coder/reviewer),
this agent:

- is NOT in the roster — users cannot @ it, other agents cannot delegate to it
- is NOT user-invokable — it runs only as an automatic post-invocation hook
- has read-only file tools (Read/Glob/Grep) so it can inspect referenced paths
- has NO memory_search tool — it produces memory, never consumes it (avoids the
  self-triggering loop: memory agent writing memory would itself trigger a
  memory agent invocation)
- defaults to the Claude Code CLI platform (reuses ClaudeCodeAgent execution),
  but carries its own system_prompt so its output contract is the JSONL shape
  defined in §2.3 of the 2.2 plan

This module only defines the spec + system_prompt + registration helpers.
The actual invocation wiring (orchestrator hook) lands in stage 4.
"""
from __future__ import annotations

from agents.cli.claude_code import ClaudeCodeAgent
from core.registry import registry
from models.agent_spec import AgentSpec


MEMORY_AGENT_ID = "memory"
MEMORY_AGENT_NAME = "记忆提取器"


# Output contract: a single JSON object { "memory": [ {id,text,attributed_to,
# linked_memory_ids}, ... ] }. ``id`` is a within-batch ordinal ("0","1",...),
# NOT the final UUID — the system mints the UUID when it writes the Qdrant Point.
SYSTEM_PROMPT = """你是记忆提取器。你的工作是把一段 agent 工作上下文提纯成结构化记忆，只输出一段 JSON，不要输出任何解释、前后缀、Markdown 代码围栏。

JSON 格式：
{
  "memory": [
    {"id": "0", "text": "...", "attributed_to": "user", "linked_memory_ids": ["..."]}
  ]
}

字段规则：
- id：必填字符串，从 "0" 开始依次编号，表示本次提取的第几条。
- text：必填字符串，提纯后的事实陈述。
- attributed_to：必填字符串，"user" 或 "assistant"，表示这条话的来源。
- linked_memory_ids：可选字符串数组，指向上下文里给你的已有 memory 的 uuid；没有关联就传 []。

提取原则：
- 只记值得长期保留的事实：项目决策、技术选型、用户偏好、已完成的变更、约定、纠正。
- 不记一次性指令（"这次用 Python"）、过程性叙述、未确认的猜测、寒暄。
- text 是提纯后的事实陈述，不是原文复制；去掉寒暄、冗余、上下文依赖词，让这条话脱离上下文也能看懂。
- attributed_to 标记来源：用户说的话填 "user"，agent 说的结论填 "assistant"。
- 如果某条新记忆和「已有相关 memory」里的某条相关，把那条的 uuid 填进 linked_memory_ids。
- 没有值得记的就输出 {"memory": []}。

上下文里会给你「已有相关 memory」的 uuid 和内容，用 linked_memory_ids 引用它们。只能引用上下文里实际出现的 uuid，不要编造。
"""


def memory_agent_spec() -> AgentSpec:
    """Build the AgentSpec for the memory extraction agent.

    Internal: ``internal=True`` keeps it out of the roster and out of
    delegate/entry resolution. Read-only sandbox, no memory_search capability.
    Platform defaults to the Claude Code CLI (the only execution backend today;
    ``platform`` is a forward-compat field for swapping in other backends).
    """
    return AgentSpec(
        agent_id=MEMORY_AGENT_ID,
        name=MEMORY_AGENT_NAME,
        project_id="",                 # not project-scoped; global internal agent
        role="memory",
        system_prompt=SYSTEM_PROMPT,
        # No file tools: extraction is pure text→JSON from the prompt context,
        # which already carries user_text/final_text/mutation_paths. Giving it
        # Read/Glob/Grep tempts the model to actually read the mutated files and
        # burns turns → wall-clock timeout under the 300s deadline. Pure prompt
        # extraction completes in a single model turn.
        allowed_tools=[],
        builtin_tools=[],
        mcp_capabilities=["report_progress"],  # NO delegate, NO memory_search
        disallowed_tools=["Bash", "Edit", "Write", "Read", "Glob", "Grep"],
        sandbox="read",
        auto_trigger=False,            # never auto-routed as an entry agent
        delegatable=False,             # cannot be pulled via delegate tool
        max_turns=3,                   # safety net — single-turn JSON expected
        timeout=300,
        internal=True,                 # new field: excluded from roster/delegate
        platform="claude-cli",         # new field: execution backend selector
        origin="generated",
    )


def register_memory_agent() -> None:
    """Register the memory agent in the global registry as a built-in.

    Idempotent: re-registering replaces the existing instance. The spec is
    attached as ``_spec`` so the Claude Code executor drives the CLI from it
    (same pattern as ``core.registry.materialize``).
    """
    spec = memory_agent_spec()
    agent = ClaudeCodeAgent(
        agent_id=spec.agent_id,
        name=spec.name,
        description="Memory extraction agent (internal, 2.2)",
    )
    agent._spec = spec  # type: ignore[attr-defined]
    registry.register(agent)


def is_internal(agent_id: str) -> bool:
    """True if the given agent_id refers to an internal (non-working) agent.

    Used by roster construction and delegate/entry resolution to exclude the
    memory agent from user-facing surfaces.
    """
    agent = registry.get(agent_id)
    if agent is None:
        return False
    spec = getattr(agent, "_spec", None)
    return bool(spec and getattr(spec, "internal", False))
