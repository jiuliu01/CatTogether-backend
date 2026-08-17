import asyncio
import json

from config import settings
from integrations.feishu import mapping_store as mapping_module
from integrations.feishu.mapping_store import FeishuMappingStore


def test_rebinding_existing_group_updates_binding_and_project(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    old_workspace = tmp_path / "old"
    new_workspace = tmp_path / "life-note-miniapp"
    old_workspace.mkdir()
    new_workspace.mkdir()
    (data_dir / "feishu").mkdir(parents=True)
    (data_dir / "workspace_roots.json").write_text(
        json.dumps([str(tmp_path)]), encoding="utf-8"
    )
    (data_dir / "feishu" / "bindings.json").write_text(
        json.dumps([
            {
                "id": "binding-1",
                "tenant_key": "tenant",
                "chat_id": "chat",
                "channel_id": "channel",
                "workspace_dir": str(old_workspace),
                "project_id": "project-1",
            }
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "data_dir", data_dir)

    class Projects:
        def __init__(self):
            self.updated = None

        async def update(self, project_id, **fields):
            self.updated = (project_id, fields)

    projects = Projects()
    monkeypatch.setattr(mapping_module, "project_store", projects)

    async def scenario():
        store = FeishuMappingStore()
        binding = await store.bind("tenant", "chat", str(new_workspace))
        saved = json.loads(
            (data_dir / "feishu" / "bindings.json").read_text(encoding="utf-8")
        )[0]
        return binding, saved

    binding, saved = asyncio.run(scenario())
    assert binding.workspace_dir == str(new_workspace.resolve())
    assert saved["workspace_dir"] == str(new_workspace.resolve())
    assert projects.updated == (
        "project-1",
        {"workspace_dir": str(new_workspace.resolve())},
    )
