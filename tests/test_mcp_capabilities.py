import json
from pathlib import Path

from mcp_tools.mcp_config import remove_mcp_config, write_mcp_config


def test_chat_run_does_not_mount_wiki_server(monkeypatch, tmp_path):
    monkeypatch.setattr("mcp_tools.mcp_config.settings.mcp_config_dir", tmp_path)
    path = write_mcp_config("chat-token", ["delegate", "report_progress"])
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        assert set(config["mcpServers"]) == {"cattogether"}
    finally:
        remove_mcp_config(path)


def test_document_run_mounts_only_granted_wiki_tools(monkeypatch, tmp_path):
    monkeypatch.setattr("mcp_tools.mcp_config.settings.mcp_config_dir", tmp_path)
    path = write_mcp_config("wiki-token", ["delegate", "report_progress", "wiki_write"])
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        assert set(config["mcpServers"]) == {"cattogether", "cattogether-wiki"}
        wiki_env = config["mcpServers"]["cattogether-wiki"]["env"]
        assert wiki_env["CT_MCP_CAPABILITIES"] == "wiki_write"
    finally:
        remove_mcp_config(path)


def test_explicit_empty_capabilities_mounts_no_servers(monkeypatch, tmp_path):
    monkeypatch.setattr("mcp_tools.mcp_config.settings.mcp_config_dir", tmp_path)
    path = write_mcp_config("no-tools-token", [])
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        assert config["mcpServers"] == {}
    finally:
        remove_mcp_config(path)
