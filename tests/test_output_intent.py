from core.output_intent import compact_artifact_final_message, resolve_output_intent


def test_long_chat_request_does_not_grant_wiki_write():
    intent = resolve_output_intent("详细分析一下这个项目，回答不少于 3000 字")

    assert intent.mode == "chat"
    assert intent.wiki_capabilities == ()
    assert intent.artifact_requested is False


def test_explicit_document_request_grants_only_write():
    intent = resolve_output_intent("整理成一份飞书文档，并在聊天里告诉我结论")

    assert intent.mode == "both"
    assert intent.wiki_capabilities == ("wiki_write",)
    assert intent.artifact_requested is True


def test_explicit_document_update_grants_append_and_read():
    intent = resolve_output_intent("给飞书文档补充一节风险说明")

    assert intent.mode == "wiki"
    assert set(intent.wiki_capabilities) == {"wiki_read", "wiki_list", "wiki_append"}
    assert intent.artifact_requested is True


def test_document_body_is_not_duplicated_into_chat(monkeypatch):
    monkeypatch.setattr("core.output_intent.settings.artifact_final_message_limit", 100)
    message, compacted = compact_artifact_final_message(
        "文档正文" * 100,
        artifact_requested=True,
        artifacts=[{"artifact_id": "node1"}],
    )

    assert compacted is True
    assert message == "文档已按要求生成或更新，完整内容请查看下方交付物。"


def test_missing_artifact_keeps_full_chat_answer(monkeypatch):
    monkeypatch.setattr("core.output_intent.settings.artifact_final_message_limit", 100)
    original = "文档正文" * 100
    message, compacted = compact_artifact_final_message(
        original,
        artifact_requested=True,
        artifacts=[],
    )

    assert compacted is False
    assert message == original
