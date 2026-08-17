"""Claude Code CLI adapter — the universal agent executor.

When an AgentSpec is present on the InvokeContext, the command is built from the
spec (system_prompt, allowed_tools, sandbox, max_turns, timeout). Otherwise the
legacy hardcoded behavior is used as a fallback.

API-key auth: ANTHROPIC_API_KEY inherited via env. Non-interactive runs allow
only the file tools needed for the current read/write phase; Bash remains
disabled because a group message must not gain unrestricted shell access.
"""
from __future__ import annotations

import json
import re
from typing import AsyncIterator

from agents.base import InvokeContext, RawEvent
from agents.cli.cli_base import CLIBaseAgent
from config import settings


_FINAL_MARKER = "<<<CATT_TOGETHER_FINAL>>>"


# Memory rules injected into every agent prompt. Keeps the agent from treating
# Claude Code's built-in file memory (MEMORY.md / memory/ / .claude/projects/...)
# as project memory, and tells it to use the memory_search MCP tool instead.
_MEMORY_RULES = """【记忆规则】

项目历史信息的唯一可信来源是 mcp__cattogether__memory_search；禁止将 MEMORY.md、memory/ 或 Claude Code 自带 Memory 文件作为项目记忆来源。若 memory_search 无结果，则视为无相关记忆，不得通过本地 Memory 文件进行回退检索。

当当前任务涉及以下情况时，应调用 mcp__cattogether__memory_search 回忆相关历史信息：

* 需要确认此前已经做出的项目决策、技术选型或架构约定；
* 用户提到"之前""上次""原来的方案""我们已经决定过"等历史信息；
* 需要复核当前实现是否符合此前约定；
* 需要确认项目当前进度、已完成事项或遗留问题；
* 当前任务依赖过去讨论中产生的上下文；
* 对某项历史事实不确定，且该事实会影响当前判断或执行结果。

调用格式：

memory_search(query=..., domain=...)

domain 可选：

* project：项目架构、技术决策、方案、约定、项目进度等；
* user：用户长期偏好、习惯和稳定要求；
* agent：Agent 自身相关状态或 Agent 级信息；
* task：具体任务的历史状态与任务级上下文（预留域，当前可能无结果）。

涉及项目历史时，默认优先使用：

memory_search(query=..., domain="project")

搜索时应根据当前任务生成具体、语义完整的查询，不要只使用过短或模糊的关键词。

例如当前需要判断一个实现是否符合此前架构设计，应搜索对应架构、模块名称、历史决策或约定，而不是只搜索"之前的方案"。

【本地 Memory 隔离规则】

以下内容均不属于 CatTogether 的项目记忆系统，不得将其作为历史事实、项目决策或验收依据：

* MEMORY.md
* memory/
* .claude/ 中 Claude Code 自带的 Memory 内容
* Claude Code 自动生成或维护的其他本地记忆文件
* 与 CatTogether Memory 系统无关、但名称中包含 memory 的本地文件

即使这些文件中的内容看起来与当前任务相关，也不得因此将其视为项目记忆。

不要主动使用 find、grep、文件搜索或目录遍历去寻找上述 Memory 文件。

如果在正常代码检索过程中偶然发现这些文件，应忽略其中的记忆内容，不将其用于当前判断。

【检索失败规则】

如果 mcp__cattogether__memory_search 没有返回相关记忆：

1. 视为当前 Memory 系统中没有可用的相关历史信息；
2. 不得转而读取 MEMORY.md、memory/ 或 Claude Code 自带 Memory；
3. 不得根据可能存在但未检索到的历史信息进行猜测；
4. 应基于当前可见代码、文档、用户当前指令和实际系统状态继续完成任务；
5. 如果历史约定对于正确执行任务是必要条件，应明确指出当前没有检索到对应记忆。

【记忆与当前事实的关系】

Memory 用于恢复历史上下文，但不能覆盖当前实际状态。

如果 Memory 中的旧信息与当前代码、配置、接口或用户最新指令发生冲突：

* 用户当前明确指令优先；
* 当前实际代码和系统状态优先于过时 Memory；
* 应将 Memory 视为历史记录，而不是强制覆盖现实状态；
* 必要时指出历史约定与当前实现之间存在差异。

【协作 Agent】

需要额外分析、代码检查、并行探索或独立复核时，可以使用 delegate 请求其他 Agent 协助。

被委派的 Agent 同样具备 memory_search，可自行遵循本规则查询项目历史。主 Agent 也可在 task 描述中补充关键历史结论，减少子 Agent 的重复检索。

最终主 Agent 应结合当前任务、实际代码状态以及有效的 CatTogether Memory 结果做出判断。"""


def split_final_response(text: str) -> tuple[str, str]:
    """Split process narration from the user-facing final answer.

    New prompts require an explicit marker. The heuristic fallback handles
    older/custom agent prompts that return a short work-log preamble followed
    by a separator or Markdown report heading.
    """
    value = (text or "").strip()
    if not value:
        return "", ""

    # The marker is a protocol boundary only when it occupies a whole line.
    # Use the first boundary: a formal report may legitimately quote the
    # marker later while documenting this adapter.
    marker_line = re.search(
        rf"(?m)^[ \t]*{re.escape(_FINAL_MARKER)}[ \t]*\r?$",
        value,
    )
    if marker_line:
        process = value[:marker_line.start()]
        answer = value[marker_line.end():]
        return process.strip(), answer.strip()

    def looks_like_process(prefix: str) -> bool:
        if not prefix or len(prefix) > 1600:
            return False
        action_cues = (
            "我先",
            "我已经",
            "我已",
            "自己上手",
            "读完",
            "检查完",
            "定位完",
            "分析完",
            "源码",
            "相关文件",
        )
        handoff_cues = (
            "下面是",
            "以下是",
            "完整分析",
            "完整报告",
            "分析报告",
            "结果如下",
            "总结如下",
        )
        return (
            any(cue in prefix for cue in action_cues)
            and any(cue in prefix for cue in handoff_cues)
        )

    # Common Claude pattern:
    # "I inspected ... Below is the report.\n\n---\n\n# Report"
    separator = re.search(r"\n{2,}---\s*\n{2,}", value)
    if separator:
        prefix = value[:separator.start()].strip()
        answer = value[separator.end():].strip()
        if looks_like_process(prefix) and answer:
            return prefix, answer

    # Same pattern without a horizontal rule.
    heading = re.search(r"(?m)^#{1,6}\s+\S", value)
    if heading and heading.start() > 0:
        prefix = value[:heading.start()].strip()
        if looks_like_process(prefix):
            return prefix, value[heading.start():].strip()

    return "", value


class ClaudeCodeAgent(CLIBaseAgent):
    def __init__(self, agent_id: str = "claude-code", name: str = "Claude Code", description: str = "Anthropic Claude Code CLI agent") -> None:
        super().__init__(agent_id, name, settings.claude_bin, description)
        self._pending_text: dict[int, list[str]] = {}

    def _bin_name(self) -> str:
        return "claude"

    def build_command(self, ctx: InvokeContext) -> list[str]:
        bin = self.resolve_bin() or "claude"
        spec = ctx.spec
        command = [
            bin, "-p", "-",  # read prompt from stdin
            "--output-format", "stream-json",
            "--verbose",
        ]
        # Attach the per-run MCP config so the agent can call the delegate tool.
        if ctx.mcp_config_path:
            command.extend(["--mcp-config", ctx.mcp_config_path])
            command.append("--strict-mcp-config")
        if spec is not None:
            max_turns = getattr(spec, "max_turns", 30) or 30
            command.extend(["--max-turns", str(max_turns)])
            sandbox = getattr(spec, "sandbox", "read")
            effective = getattr(spec, "effective_builtin_tools", None)
            allowed = list(effective if effective is not None else getattr(spec, "allowed_tools", ["Read", "Glob", "Grep"]))
            disallowed = list(getattr(spec, "disallowed_tools", ["Bash"]))
            # Read/write is controlled purely by the tool whitelist, NOT by
            # permission-mode. We never use "plan" mode: in non-interactive -p
            # mode it traps the agent in a plan/ExitPlanMode loop. acceptEdits
            # avoids interactive prompts; disallowedTools enforces the boundary.
            if ctx.workspace_access != "write" or sandbox != "workspace-write":
                allowed = [t for t in allowed if t not in ("Edit", "Write")]
            known_builtin = {"Read", "Glob", "Grep", "Edit", "Write", "WebSearch", "WebFetch"}
            allowed = [
                t for t in dict.fromkeys(allowed)
                if t in known_builtin and t.lower() not in {"agent", "bash"}
            ]
            command.extend(["--permission-mode", "acceptEdits"])
            if settings.strict_role_tools:
                command.extend(["--tools", *(allowed or [""])])
            preapproved = list(allowed)
            for capability in ctx.mcp_capabilities:
                if capability in {"delegate", "report_progress"}:
                    preapproved.append(f"mcp__cattogether__{capability}")
                elif capability.startswith("wiki_"):
                    preapproved.append(f"mcp__cattogether-wiki__{capability}")
                elif capability == "memory_search":
                    preapproved.append("mcp__cattogether__memory_search")
            if preapproved:
                command.extend(["--allowedTools", *dict.fromkeys(preapproved)])
            disallowed = list(dict.fromkeys(disallowed + ["Agent", "Bash"]))
            command.extend(["--disallowedTools", *disallowed])
            return command

        # Legacy fallback (no spec): hardcoded tool envelope by access phase.
        command.extend(["--max-turns", "30"])
        if ctx.workspace_access == "write":
            command.extend([
                "--permission-mode", "acceptEdits",
                "--tools", "Read", "Glob", "Grep", "Edit", "Write",
                "--allowedTools", "Read", "Glob", "Grep", "Edit", "Write",
                "--disallowedTools", "Agent", "Bash",
            ])
        else:
            command.extend([
                "--permission-mode", "acceptEdits",
                "--tools", "Read", "Glob", "Grep",
                "--allowedTools", "Read", "Glob", "Grep",
                "--disallowedTools", "Agent", "Bash", "Edit", "Write",
            ])
        return command

    def build_prompt(self, ctx: InvokeContext) -> str:
        spec = ctx.spec
        parts: list[str] = []
        system_prompt = getattr(spec, "system_prompt", "") if spec else ""
        if system_prompt:
            parts.append(system_prompt)
        parts.append(_MEMORY_RULES)
        if ctx.memory:
            parts.append(ctx.memory)
        if ctx.roster:
            parts.append(ctx.roster)
        if ctx.workspace_dir:
            parts.append(
                "当前项目根目录（系统已绑定）："
                f"{ctx.workspace_dir}\n"
                "所有相对路径都以此目录为准。不要把 CatTogether 后端目录、隔离目录或其他相邻目录"
                "误认为当前项目；除非用户明确要求，不要向此根目录之外搜索或修改文件。"
            )
        if ctx.output_intent:
            mode = ctx.output_intent.get("mode", "chat")
            wiki_caps = ", ".join(ctx.output_intent.get("wiki_capabilities", [])) or "无"
            parts.append(
                f"本次交付方式：{mode}。可用知识库能力：{wiki_caps}。"
                "chat 表示直接在飞书回答；wiki/both 表示用户明确要求文档，文档正文写入知识库，"
                "聊天中的最终答复只保留简短结论和文档链接。不要因为答案较长就自行改成文档。"
            )
        if settings.structured_run_result:
            parts.append(
                "输出约定：Read/Glob/Grep/Edit/Write/delegate 等工具动作会由系统自动记录。"
                "发现重要阶段结论时，用 report_progress 工具发一条简短、可向用户复述的 checkpoint；"
                "不要报告私有推理、原始工具结果、密钥或完整配置。terminal result 只放真正给用户看的"
                "最终答复，不要重复工作日志，也不要输出内部边界标记。若任务明确要求文档，先调用"
                "允许的 wiki_* 工具创建或更新文档，最终答复只说明结论、完成情况和链接。"
            )
        else:
            parts.append(
                "输出约定：工具调用前的工作说明、检查进度属于工作过程。真正给用户看的最终答复"
                f"必须以单独一行 {_FINAL_MARKER} 开始；标记后只放最终结论或正式报告。"
            )
        parts.append(f"User message:\n{ctx.user_message}")
        return "\n\n---\n\n".join(parts)

    async def parse_chunk(self, raw: str, ctx: InvokeContext) -> AsyncIterator[RawEvent]:
        line = raw.strip()
        if not line:
            return
        try:
            obj = json.loads(line)
        except Exception:
            yield ("text_delta", {"delta": raw})
            return
        t = obj.get("type")
        if t == "assistant":
            # stream-json: assistant message with content blocks
            blocks = obj.get("message", {}).get("content", [])
            has_tool_use = any(
                isinstance(block, dict) and block.get("type") == "tool_use"
                for block in blocks
            )
            context_key = id(ctx)
            if has_tool_use:
                for pending in self._pending_text.pop(context_key, []):
                    if pending:
                        yield ("text_delta", {"delta": pending})
            for block in blocks:
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if not text:
                        continue
                    if has_tool_use:
                        # Narration immediately before a tool call belongs in
                        # the collapsed progress panel, not in the final result.
                        yield ("text_delta", {"delta": text})
                    else:
                        # Buffer a text-only assistant message. If a later
                        # assistant message calls a tool it becomes process;
                        # otherwise the terminal result deduplicates it.
                        self._pending_text.setdefault(context_key, []).append(text)
                elif block.get("type") == "tool_use":
                    tool_name = block.get("name", "tool")
                    tool_input = block.get("input", {})
                    tool_id = block.get("id", "")
                    # Surface delegate tool calls so the orchestrator can act.
                    if tool_name == "delegate":
                        yield ("tool_call", {"tool": "delegate", "args": tool_input, "tool_use_id": tool_id})
                    else:
                        yield ("tool_call", {"tool": tool_name, "args": tool_input, "tool_use_id": tool_id})
        elif t == "tool_result":
            content = obj.get("content", "")
            if isinstance(content, list):
                content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
            yield ("tool_result", {"tool": obj.get("tool_use_id", "tool"), "result": content})
        elif t == "result":
            pending = "".join(self._pending_text.pop(id(ctx), [])).strip()
            if obj.get("is_error") or obj.get("subtype") in ("error", "failed"):
                yield ("error", {"message": obj.get("result") or str(obj)})
            else:
                # Canonical answer. Move a work-log preamble into progress and
                # keep only the formal response as final_text.
                result = obj.get("result")
                if isinstance(result, str) and result:
                    from core.output_safety import sanitize_agent_text

                    process, answer = split_final_response(result) if settings.legacy_final_marker else ("", result.strip())
                    process = sanitize_agent_text(process).text
                    answer = sanitize_agent_text(answer).text
                    if pending and pending not in {answer, result.strip()} and pending not in process:
                        process = f"{pending}\n{process}".strip()
                    if process:
                        yield ("text_delta", {"delta": process})
                    if answer:
                        yield ("final_text", {"text": answer})
        elif t == "error":
            self._pending_text.pop(id(ctx), None)
            yield ("error", {"message": obj.get("message", str(obj))})
        else:
            text = obj.get("text") or obj.get("content")
            if text and isinstance(text, str):
                yield ("text_delta", {"delta": text})
