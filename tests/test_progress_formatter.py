from core.progress_formatter import format_progress


def test_formats_file_tools_as_user_visible_actions():
    read = format_progress("tool_call", {
        "tool": "Read",
        "args": {"file_path": "backend/core/orchestrator.py"},
    })
    grep = format_progress("tool_call", {
        "tool": "Grep",
        "args": {"pattern": "run_entry"},
    })

    assert read and "正在读取" in read.title
    assert "orchestrator.py" in read.title
    assert grep and "正在检索" in grep.title


def test_formats_delegate_and_hides_raw_tool_results():
    delegated = format_progress("tool_call", {
        "tool": "mcp__cattogether__delegate",
        "args": {"target": "狸花", "task": "检查架构"},
    })

    assert delegated and delegated.kind == "delegate"
    assert "狸花" in delegated.title
    assert format_progress("tool_result", {"result": "very large source"}) is None


def test_formats_heartbeat_elapsed_time():
    event = format_progress("heartbeat", {"elapsed_seconds": 45})
    assert event and "45 秒" in event.title
