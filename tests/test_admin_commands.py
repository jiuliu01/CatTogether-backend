import pytest

from integrations.feishu.admin_commands import parse_admin_command


def test_parse_generate():
    cmd = parse_admin_command("生成 agent sre role=custom name=SRE from=custom")
    assert cmd is not None
    assert cmd.op == "generate"
    assert cmd.agent_id == "sre"
    assert cmd.role == "custom"
    assert cmd.name == "SRE"
    assert cmd.from_template == "custom"


def test_parse_eliminate():
    cmd = parse_admin_command("消除 reviewer")
    assert cmd is not None
    assert cmd.op == "eliminate"
    assert cmd.agent_id == "reviewer"


def test_parse_specify():
    cmd = parse_admin_command("制定 coder：max_turns=5 sandbox=workspace-write")
    assert cmd is not None
    assert cmd.op == "specify"
    assert cmd.agent_id == "coder"
    assert cmd.fields == {"max_turns": "5", "sandbox": "workspace-write"}


def test_parse_transplant():
    cmd = parse_admin_command("把 chatA 的 coder 移植到 chatB mode=move")
    assert cmd is not None
    assert cmd.op == "transplant"
    assert cmd.src_chat == "chatA"
    assert cmd.agent_id == "coder"
    assert cmd.dst_chat == "chatB"
    assert cmd.mode == "move"


def test_parse_list():
    cmd = parse_admin_command("列出 agent")
    assert cmd is not None
    assert cmd.op == "list"


def test_non_admin_message_returns_none():
    assert parse_admin_command("让 Coder 重构 auth") is None
    assert parse_admin_command("帮我规划这个项目") is None
