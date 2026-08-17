import asyncio

from core.agent_store import agent_store
from core.project_store import project_store


def _run(coro):
    return asyncio.run(coro)


def test_create_project_and_seed_defaults(monkeypatch):
    async def scenario(tmp):
        # isolate data dir so we don't touch real projects
        project = await project_store.create("p-seed", workspace_dir=str(tmp))
        specs = await agent_store.seed_defaults(project.id)
        names = {s.agent_id for s in specs}
        assert {"coordinator", "researcher", "coder", "reviewer"} <= names
        project = await project_store.get(project.id)
        assert set(project.agent_ids) >= names
        assert project.default_agent_id == "coordinator"
        # persisted to disk
        loaded = await agent_store.list(project.id)
        assert {s.agent_id for s in loaded} >= names

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        _run(scenario(tmp))


def test_generate_specify_eliminate(monkeypatch):
    async def scenario(tmp):
        project = await project_store.create("p-ops", workspace_dir=str(tmp))
        spec = await agent_store.create(
            project.id, "sre", role="custom", name="SRE", from_template="custom",
        )
        assert spec.origin == "generated"
        assert spec.name == "SRE"

        from models.agent_spec import AgentSpecUpdateRequest
        updated = await agent_store.update(
            project.id, "sre",
            AgentSpecUpdateRequest(
                system_prompt="you are SRE",
                max_turns=5,
                builtin_tools=["Read", "Agent", "Bash", "UnknownTool"],
                mcp_capabilities=["delegate", "untrusted_tool"],
            ),
        )
        assert updated.system_prompt == "you are SRE"
        assert updated.max_turns == 5
        # safety: Bash must stay disallowed
        assert "Bash" in updated.disallowed_tools
        assert updated.builtin_tools == ["Read"]
        assert updated.mcp_capabilities == ["delegate"]

        ok = await agent_store.delete(project.id, "sre", keep_memory=True)
        assert ok
        assert await agent_store.get(project.id, "sre") is None
        project = await project_store.get(project.id)
        assert "sre" not in project.agent_ids

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        _run(scenario(tmp))


def test_transplant_does_not_copy_project_memory(monkeypatch):
    async def scenario(tmp):
        proj_a = await project_store.create("pa", workspace_dir=str(tmp / "a"))
        proj_b = await project_store.create("pb", workspace_dir=str(tmp / "b"))
        await agent_store.create(proj_a.id, "coder", role="coding", name="Coder")
        # write a project-domain fact via the v2.1 API; project memory belongs
        # to the project and must not follow an Agent transplant.
        from memory.memory_api import MemoryAPI, WriteProposal
        from memory.permissions import ActorContext

        api = MemoryAPI()
        actor = ActorContext(
            tenant_id="default",
            role="system",
            actor_id="coder",
            display_name="Coder",
        )
        proposal = WriteProposal(
            text=f"data/projects/{proj_a.id}/workspace 的结构",
            domain="project",
            scope_hint=proj_a.id,
            kind="project_fact",
            tags=["pa"],
            importance=0.7,
            confidence=0.9,
            source_type="agent_result",
            actor_id="coder",
            # Rich context so the rule-based candidate extractor produces a
            # fact: workspace_id + mutation_paths trigger the workspace/project
            # candidate path.
            workspace_id=proj_a.id,
            mutation_paths=[f"data/projects/{proj_a.id}/README.md"],
            user_text="项目修改已完成",
            run_id="test-run-pa",
        )
        await api.write_propose(proposal, actor=actor)

        dst = await agent_store.transplant(proj_a.id, "coder", proj_b.id, mode="copy")
        assert dst.project_id == proj_b.id
        assert dst.origin == "transplanted"

        # V2 project memory belongs to the project and must not follow an
        # Agent transplant. Search the destination project — it should be empty.
        from memory.models import RetrievalQuery

        query = RetrievalQuery(
            text="workspace 结构",
            tenant_id="default",
            domain="project",
            scope_id=proj_b.id,
            intent="recall",
            top_k=5,
        )
        dst_results = await api.search(query, actor=actor)
        assert dst_results == []

        src_query = RetrievalQuery(
            text="workspace 结构",
            tenant_id="default",
            domain="project",
            scope_id=proj_a.id,
            intent="recall",
            top_k=5,
        )
        src_results = await api.search(src_query, actor=actor)
        assert src_results

        # source still present (copy mode)
        assert await agent_store.get(proj_a.id, "coder") is not None

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        _run(scenario(Path(tmp)))


def test_transplant_move_removes_source(monkeypatch):
    async def scenario(tmp):
        proj_a = await project_store.create("pa2", workspace_dir=str(tmp / "a"))
        proj_b = await project_store.create("pb2", workspace_dir=str(tmp / "b"))
        await agent_store.create(proj_a.id, "coder", role="coding", name="Coder")

        await agent_store.transplant(proj_a.id, "coder", proj_b.id, mode="move")
        assert await agent_store.get(proj_a.id, "coder") is None
        assert await agent_store.get(proj_b.id, "coder") is not None

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        _run(scenario(Path(tmp)))
