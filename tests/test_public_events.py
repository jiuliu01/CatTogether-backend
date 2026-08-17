from core.orchestrator import _public_event_data


def test_tool_result_is_not_exposed_to_public_event_stream():
    data = _public_event_data(
        "tool_result",
        {"tool": "Read", "result": "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz"},
    )

    assert data["result"] == "工具已完成；原始结果不直接展示。"
    assert "sk-" not in str(data)


def test_tool_call_keeps_only_safe_progress_fields():
    data = _public_event_data(
        "tool_call",
        {
            "tool": "Write",
            "tool_use_id": "tool-1",
            "args": {
                "file_path": "backend/main.py",
                "content": "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz",
            },
        },
    )

    assert data["args"] == {"file_path": "backend/main.py"}
