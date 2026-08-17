import json

from integrations.feishu.message_parser import FeishuMessageParser


def message_event(
    *,
    message_id: str = "om_test_1",
    message_type: str = "text",
    text: str = "@_user_1 帮我规划这个项目",
    sender_type: str = "user",
    root_id: str | None = None,
) -> dict:
    return {
        "schema": "2.0",
        "header": {
            "event_id": "evt_1",
            "event_type": "im.message.receive_v1",
            "tenant_key": "tenant_1",
        },
        "event": {
            "sender": {
                "sender_id": {"open_id": "ou_user"},
                "sender_type": sender_type,
                "tenant_key": "tenant_1",
            },
            "message": {
                "message_id": message_id,
                "root_id": root_id,
                "parent_id": None,
                "chat_id": "oc_group",
                "chat_type": "group",
                "message_type": message_type,
                "content": json.dumps({"text": text}, ensure_ascii=False),
                "mentions": [
                    {
                        "key": "@_user_1",
                        "id": {"open_id": "ou_bot"},
                        "name": "CatTogether",
                    }
                ],
            },
        },
    }


def test_parse_group_mention_and_remove_bot_token():
    inbound, reason = FeishuMessageParser().parse(message_event(root_id="om_root"))

    assert reason is None
    assert inbound is not None
    assert inbound.content == "帮我规划这个项目"
    assert inbound.chat_id == "oc_group"
    assert inbound.external_root_id == "om_root"
    assert inbound.mentioned_bot is True


def test_ignore_bot_sender():
    inbound, reason = FeishuMessageParser().parse(
        message_event(sender_type="bot")
    )

    assert inbound is None
    assert reason == "bot_sender"


def test_reject_non_text_message():
    inbound, reason = FeishuMessageParser().parse(
        message_event(message_type="image")
    )

    assert inbound is None
    assert reason == "unsupported_message_type"
