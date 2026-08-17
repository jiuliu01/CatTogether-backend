import pytest
from pathlib import Path

from agents.base import InvokeContext
from agents.cli.claude_code import ClaudeCodeAgent
from agents.cli.codex import CodexAgent
from models.agent_spec import AgentSpec


def test_codex_uses_workspace_sandbox_for_writes(monkeypatch):
    agent = CodexAgent()
    monkeypatch.setattr(agent, "resolve_bin", lambda: "codex")
    command = agent.build_command(
        InvokeContext(
            channel_id="ch",
            user_message="change code",
            workspace_access="write",
        )
    )

    assert "--dangerously-bypass-approvals-and-sandbox" not in command
    assert command[command.index("--sandbox") + 1] == "workspace-write"
    assert command[command.index("--ask-for-approval") + 1] == "never"


def test_codex_prefers_desktop_cli_over_windowsapps(monkeypatch, tmp_path):
    local_app_data = tmp_path / "Local"
    desktop_cli = local_app_data / "OpenAI" / "Codex" / "bin" / "version" / "codex.exe"
    desktop_cli.parent.mkdir(parents=True)
    desktop_cli.write_bytes(b"test")
    agent = CodexAgent()
    monkeypatch.setattr(agent, "bin_path", None)
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    assert Path(agent.resolve_bin()) == desktop_cli


def test_codex_uses_read_only_for_review(monkeypatch):
    agent = CodexAgent()
    monkeypatch.setattr(agent, "resolve_bin", lambda: "codex")
    command = agent.build_command(
        InvokeContext(
            channel_id="ch",
            user_message="review",
            workspace_access="read",
        )
    )

    assert command[command.index("--sandbox") + 1] == "read-only"


def test_claude_never_bypasses_permissions_or_allows_bash(monkeypatch):
    agent = ClaudeCodeAgent()
    monkeypatch.setattr(agent, "resolve_bin", lambda: "claude")
    command = agent.build_command(
        InvokeContext(
            channel_id="ch",
            user_message="change code",
            workspace_access="write",
        )
    )

    assert "--dangerously-skip-permissions" not in command
    assert "acceptEdits" in command
    assert "--disallowedTools" in command
    assert "Bash" in command


def test_claude_prompt_declares_bound_workspace_root():
    agent = ClaudeCodeAgent("coordinator", "橘长")
    prompt = agent.build_prompt(
        InvokeContext(
            channel_id="channel",
            user_message="怎么预览小程序？",
            workspace_dir=r"D:\Project\life-note-miniapp",
        )
    )

    assert "当前项目根目录（系统已绑定）" in prompt
    assert r"D:\Project\life-note-miniapp" in prompt
    assert "不要把 CatTogether 后端目录" in prompt


def test_claude_role_tools_are_real_availability_boundary(monkeypatch):
    agent = ClaudeCodeAgent()
    monkeypatch.setattr(agent, "resolve_bin", lambda: "claude")
    spec = AgentSpec(
        agent_id="coordinator",
        name="橘长",
        project_id="p1",
        builtin_tools=["Read", "Glob", "Grep", "Agent", "Bash"],
        mcp_capabilities=["delegate", "report_progress"],
    )
    command = agent.build_command(
        InvokeContext(
            channel_id="ch",
            user_message="inspect",
            workspace_access="read",
            spec=spec,
            mcp_config_path="run.mcp.json",
            mcp_capabilities=["delegate", "report_progress"],
        )
    )

    tools_at = command.index("--tools")
    allowed_at = command.index("--allowedTools")
    available = command[tools_at + 1:allowed_at]
    assert available == ["Read", "Glob", "Grep"]
    assert "Agent" not in available
    assert "Bash" not in available
    assert "--strict-mcp-config" in command
    assert "mcp__cattogether__delegate" in command
    assert "mcp__cattogether__report_progress" in command


@pytest.mark.parametrize("access,expected", [("write", "workspace-write"), ("read", "read-only")])
def test_codex_access_modes(access, expected, monkeypatch):
    agent = CodexAgent()
    monkeypatch.setattr(agent, "resolve_bin", lambda: "codex")
    command = agent.build_command(
        InvokeContext(channel_id="ch", user_message="task", workspace_access=access)
    )
    assert command[command.index("--sandbox") + 1] == expected
