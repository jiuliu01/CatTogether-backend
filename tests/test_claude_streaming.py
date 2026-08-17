import asyncio
import json

from agents.base import InvokeContext
from agents.cli.claude_code import ClaudeCodeAgent, split_final_response


async def _parse(payload: dict) -> list[tuple[str, dict]]:
    agent = ClaudeCodeAgent()
    ctx = InvokeContext(channel_id="ch", user_message="test")
    return [
        event
        async for event in agent.parse_chunk(
            json.dumps(payload, ensure_ascii=False),
            ctx,
        )
    ]


def test_tool_narration_is_progress_text():
    events = asyncio.run(_parse({
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": "我先读取文件。"},
                {"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}},
            ],
        },
    }))

    assert events[0] == ("text_delta", {"delta": "我先读取文件。"})
    assert events[1][0] == "tool_call"


def test_text_only_assistant_message_waits_for_terminal_result():
    events = asyncio.run(_parse({
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": "这是最终答案。"},
            ],
        },
    }))

    assert events == []


def test_buffered_text_becomes_progress_when_next_message_calls_tool():
    async def run():
        agent = ClaudeCodeAgent()
        ctx = InvokeContext(channel_id="ch", user_message="test")
        first = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "我先确认入口。"}]},
        }
        second = {
            "type": "assistant",
            "message": {"content": [
                {"type": "text", "text": "开始读取。"},
                {"type": "tool_use", "name": "Read", "input": {"file_path": "main.py"}},
            ]},
        }
        events = []
        async for event in agent.parse_chunk(json.dumps(first), ctx):
            events.append(event)
        async for event in agent.parse_chunk(json.dumps(second), ctx):
            events.append(event)
        return events

    events = asyncio.run(run())
    assert events[0] == ("text_delta", {"delta": "我先确认入口。"})
    assert events[1] == ("text_delta", {"delta": "开始读取。"})
    assert events[2][0] == "tool_call"


def test_result_event_is_canonical_final_text():
    events = asyncio.run(_parse({
        "type": "result",
        "subtype": "success",
        "result": "这是最终答案。",
    }))

    assert events == [("final_text", {"text": "这是最终答案。"})]


def test_result_splits_work_log_preamble_into_progress():
    result = (
        "喵～这是只读分析任务，我自己上手。我已经把 MCP 相关的全部源码读完了。"
        "下面是完整分析报告。\n\n---\n\n"
        "# CatTogether 项目 MCP 实现完整分析报告\n\n正式正文。"
    )

    events = asyncio.run(_parse({
        "type": "result",
        "subtype": "success",
        "result": result,
    }))

    assert events[0][0] == "text_delta"
    assert "自己上手" in events[0][1]["delta"]
    assert events[1] == (
        "final_text",
        {
            "text": (
                "# CatTogether 项目 MCP 实现完整分析报告\n\n正式正文。"
            )
        },
    )


def test_explicit_final_marker_is_preferred():
    process, answer = split_final_response(
        "我先检查文件。\n<<<CATT_TOGETHER_FINAL>>>\n最终结论。"
    )

    assert process == "我先检查文件。"
    assert answer == "最终结论。"


def test_marker_quoted_inside_formal_report_does_not_cut_answer_again():
    process, answer = split_final_response(
        "我先检查文件。\n"
        "<<<CATT_TOGETHER_FINAL>>>\n"
        "# 正式报告\n\n"
        "解析器使用 `<<<CATT_TOGETHER_FINAL>>>` 标记。\n\n"
        "```\n<<<CATT_TOGETHER_FINAL>>>\n```\n"
        "报告结束。"
    )

    assert process == "我先检查文件。"
    assert answer.startswith("# 正式报告")
    assert answer.endswith("报告结束。")
    assert answer.count("<<<CATT_TOGETHER_FINAL>>>") == 2


def test_inline_marker_is_plain_answer_text_not_a_protocol_boundary():
    process, answer = split_final_response(
        "最终说明：<<<CATT_TOGETHER_FINAL>>> 只是旧协议名称。"
    )

    assert process == ""
    assert answer == "最终说明：<<<CATT_TOGETHER_FINAL>>> 只是旧协议名称。"


def test_build_prompt_uses_structured_terminal_result():
    agent = ClaudeCodeAgent()
    prompt = agent.build_prompt(
        InvokeContext(channel_id="ch", user_message="检查项目")
    )

    assert "terminal result" in prompt
    assert "report_progress" in prompt
    assert "不要输出内部边界标记" in prompt
