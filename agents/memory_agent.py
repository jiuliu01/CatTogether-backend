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

import logging
from typing import AsyncIterator

import httpx

from agents.base import BaseAgent, InvokeContext, RawEvent
from agents.cli.claude_code import ClaudeCodeAgent
from config import settings
from core.registry import registry
from models.agent_spec import AgentSpec


logger = logging.getLogger(__name__)


MEMORY_AGENT_ID = "memory"
MEMORY_AGENT_NAME = "记忆提取器"


# Output contract: a single JSON object { "memory": [ {id,text,attributed_to,
# linked_memory_ids}, ... ] }. ``id`` is a within-batch ordinal ("0","1",...),
# NOT the final UUID — the system mints the UUID when it writes the Qdrant Point.
SYSTEM_PROMPT = """你是记忆提取器。你的工作是把一段工作上下文提纯成结构化记忆，只输出一段 JSON，不要输出任何解释、前后缀、Markdown 代码围栏。

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

【值得记的内容】
提取这类信息，无论它来自工程、项目、生活还是闲聊：
- 事件与经历：发生了什么事、何时发生、参与了什么活动。
- 偏好与习惯：用户长期稳定的喜好、做法、要求。
- 关系与归属：人物之间、人物与组织/物品/地点之间的关系。
- 决策与约定：已经做出的选择、确定的方案、双方达成的约定。
- 事实与状态：已经确认的事实、当前状态、已完成的进度。
- 时间与顺序：日期、时间点、先后顺序、因果链条。
- 纠正与更新：对旧信息的修正、补充、否定、作废。

【不记的内容】
- 一次性指令（如"这次用 Python""刚才那条删掉"）。
- 寒暄、客套、情绪宣泄、纯过程性叙述。
- 未确认的猜测、假设、设想中的方案。
- 与长期事实无关的临时上下文。

【提纯规则】
- text 是提纯后的事实陈述，不是原文复制；去掉寒暄、冗余、重复。
- 必须保留：具体时间、日期、数量、先后顺序、因果关系、归属关系。这些是事实的一部分，不是"上下文依赖词"，砍掉它们会让记忆失效。
- 让陈述尽量脱离上下文也能看懂，但前提是不丢掉上述必须保留的信息；宁可多保留一点细节，也不要为了简洁而丢掉时间或顺序。
- 一条 Memory 承载一个“可独立检索、可独立更新、可独立失效”的最小语义事实单元；多个相互独立的事实必须拆分，强关联且不可无损拆分的信息允许保留在同一 Memory 中。
- 时间和顺序信息要写进 text 本身，而不是依赖读者去回看原文。

【来源标记】
- 用户说的话填 "user"；agent 说的结论填 "assistant"。

【关联】
- 如果某条新记忆和「已有相关 memory」里的某条相关，把那条的 uuid 填进 linked_memory_ids。
- 只能引用上下文里实际出现的 uuid，不要编造。

【空结果】
- 没有值得记的就输出 {"memory": []}。但注意：事件、时间、顺序、偏好、关系都属于值得记的内容，判断标准要放宽，不要轻易输出空。
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


class ApiMemoryAgent(BaseAgent):
    """Memory extraction agent that calls an OpenAI-compatible API directly.

    Replaces the Claude Code CLI subprocess backend for the memory agent. The
    CLI backend spawns a subprocess per extraction (slow + asyncio subprocess
    pipe races on Windows under concurrency); this backend is a single httpx
    non-streaming POST per extraction — no subprocess, no pipe races, and the
    response carries a ``usage`` block we surface via the ``done`` event for
    token accounting.

    The agent is driven by its ``_spec`` (set in register_memory_agent) for
    system_prompt / timeout / max_turns. Only a single model turn is expected
    (extraction is pure text→JSON), so max_turns is effectively 1 here.
    """

    def __init__(self, agent_id: str, name: str, description: str = "") -> None:
        super().__init__(agent_id=agent_id, name=name, kind="llm", description=description)

    async def invoke(self, ctx: InvokeContext) -> AsyncIterator[RawEvent]:
        spec = getattr(self, "_spec", None)
        system_prompt = getattr(spec, "system_prompt", "") if spec else ""
        timeout = getattr(spec, "timeout", 120) if spec else 120
        api_key = settings.memory_agent_api_key
        if not api_key:
            yield ("error", {"message": "memory agent api key not configured"})
            return
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": ctx.user_message})
        payload = {
            "model": settings.memory_agent_model,
            "messages": messages,
            "temperature": settings.memory_agent_temperature,
            "max_tokens": settings.memory_agent_max_tokens,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        url = f"{settings.memory_agent_base_url.rstrip('/')}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(float(timeout), connect=15.0)) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.exception("ApiMemoryAgent request failed")
            yield ("error", {"message": str(e)})
            return
        # glm-5.2 is a reasoning model: content may be empty with the answer
        # folded into reasoning_content. Prefer content; fall back to the last
        # line of reasoning_content.
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError):
            yield ("error", {"message": f"unexpected response shape: {str(data)[:200]}"})
            return
        text = (msg.get("content") or "").strip()
        if not text:
            rc = (msg.get("reasoning_content") or "").strip()
            text = rc.splitlines()[-1].strip() if rc else ""
        usage = data.get("usage") or {}
        yield ("final_text", {"text": text})
        yield ("done", {"usage": usage})


def register_memory_agent() -> None:
    """Register the memory agent in the global registry as a built-in.

    Idempotent: re-registering replaces the existing instance. The spec is
    attached as ``_spec`` so the executor (CLI or API) drives from it.

    Backend selection: ``CT_MEMORY_AGENT_BACKEND`` (default ``api``). ``api``
    uses ApiMemoryAgent (direct httpx, no subprocess). ``cli`` falls back to
    the legacy ClaudeCodeAgent subprocess backend.
    """
    spec = memory_agent_spec()
    if settings.memory_agent_backend == "cli":
        agent = ClaudeCodeAgent(
            agent_id=spec.agent_id,
            name=spec.name,
            description="Memory extraction agent (internal, 2.2, CLI backend)",
        )
    else:
        agent = ApiMemoryAgent(
            agent_id=spec.agent_id,
            name=spec.name,
            description="Memory extraction agent (internal, 2.2, API backend)",
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
