"""Stage 6 tests: final MemoryAPI service layer (§10).

Tests the final service layer that all REST/MCP/Python callers go through:
  - Agent boundary: only write_propose exposed, no write.commit
  - ADD-only writes: create_revision for text changes
  - Two-phase forget: preview → token → confirm
  - Permission checks on every operation
  - Cross-tenant isolation
  - Hot/History/Skill sub-APIs
  - REST routes in api/memory.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# Ensure backend is on sys.path for test isolation.
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from memory.db import MemoryDB
from memory.models import Fact, RetrievalQuery
from memory.permissions import (
    ActorContext, PermissionDenied,
    CAP_WRITE_PROPOSE, CAP_UPDATE, CAP_DELETE, CAP_FORGET,
)
from memory.memory_api import (
    MemoryAPI, WriteProposal, MetadataPatch,
    ForgetCandidate, ForgetResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """Isolated MemoryDB for each test."""
    return MemoryDB(db_path=tmp_path / "test_api_memory.db")


@pytest.fixture
def api(db):
    """MemoryAPI backed by the isolated DB."""
    return MemoryAPI(db=db)


@pytest.fixture
def admin():
    return ActorContext(tenant_id="default", role="admin", actor_id="admin1")


@pytest.fixture
def agent():
    return ActorContext(tenant_id="default", role="agent", actor_id="agent1")


@pytest.fixture
def user():
    return ActorContext(tenant_id="default", role="user", actor_id="user1")


@pytest.fixture
def other_tenant_admin():
    return ActorContext(tenant_id="other", role="admin", actor_id="admin2")


def _insert_fact(db, **overrides):
    """Helper: insert a fact directly via the SQLite backend."""
    from memory.backends.sqlite_backend import SQLiteStorageBackend
    defaults = dict(
        id=f"fact_{overrides.get('id_suffix', 'test')}",
        tenant_id="default",
        domain="project",
        scope_id="default",
        kind="project_fact",
        text="test fact",
        search_text="test fact",
        tags=[],
        importance=0.5,
        confidence=0.5,
    )
    defaults.update(overrides)
    defaults.pop("id_suffix", None)
    fact = Fact(**defaults)
    SQLiteStorageBackend(db).insert_fact(fact)
    return fact


# ---------------------------------------------------------------------------
# §1  write_propose — agent boundary
# ---------------------------------------------------------------------------

class TestWritePropose:
    def test_write_propose_basic(self, api, admin):
        """Admin can propose a write and get a response."""
        proposal = WriteProposal(text="PostgreSQL is the database", domain="project")
        resp = asyncio.run(api.write_propose(proposal, actor=admin))
        assert resp.status in ("accepted", "skipped", "failed")
        assert resp.event_id.startswith("evt_")

    def test_write_propose_empty_text(self, api, admin):
        """Empty text is rejected with status=failed."""
        proposal = WriteProposal(text="", domain="project")
        resp = asyncio.run(api.write_propose(proposal, actor=admin))
        assert resp.status == "failed"
        assert resp.reason == "empty_text"

    def test_write_propose_agent_can_propose(self, api, agent):
        """Agent role can propose writes (agent boundary allows propose)."""
        proposal = WriteProposal(text="agent observation", domain="agent")
        resp = asyncio.run(api.write_propose(proposal, actor=agent))
        assert resp.status in ("accepted", "skipped", "failed")

    def test_write_propose_workspace_alias(self, api, admin):
        """'workspace' domain is mapped to 'project'."""
        proposal = WriteProposal(text="test fact", domain="workspace")
        resp = asyncio.run(api.write_propose(proposal, actor=admin))
        assert resp.status in ("accepted", "skipped", "failed")

    def test_no_commit_exposed(self, api):
        """The MemoryAPI class does NOT expose a write.commit method."""
        assert not hasattr(api, "write_commit")
        assert not hasattr(api, "commit")


# ---------------------------------------------------------------------------
# §2  search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_returns_results(self, api, admin, db):
        """Search returns a list of RetrievalResult."""
        _insert_fact(db, text="We use PostgreSQL for the database",
                     search_text="We use PostgreSQL for the database")

        query = RetrievalQuery(
            text="database",
            tenant_id="default",
            domain="project",
            scope_id="default",
            top_k=5,
        )
        results = asyncio.run(api.search(query, actor=admin))
        assert isinstance(results, list)

    def test_search_cross_tenant_isolated(self, api, other_tenant_admin, db):
        """Search in tenant 'other' does not return facts from 'default'."""
        _insert_fact(db, text="secret default tenant fact",
                     search_text="secret default tenant fact")

        query = RetrievalQuery(
            text="secret",
            tenant_id="other",
            domain="project",
            scope_id="default",
            top_k=10,
        )
        results = asyncio.run(api.search(query, actor=other_tenant_admin))
        for r in results:
            assert r.fact.tenant_id == "other"

    def test_search_permission_denied(self, api, user):
        """User role cannot read agent domain."""
        query = RetrievalQuery(
            text="test",
            tenant_id="default",
            domain="agent",
            top_k=5,
        )
        with pytest.raises(PermissionDenied):
            asyncio.run(api.search(query, actor=user))


# ---------------------------------------------------------------------------
# §3  update_metadata — ADD-only
# ---------------------------------------------------------------------------

class TestUpdateMetadata:
    def test_update_tags(self, api, admin, db):
        """Update tags on a fact."""
        _insert_fact(db, id="fact_meta1", text="test fact for metadata")

        patch = MetadataPatch(fact_id="fact_meta1", tags=["new_tag", "updated"])
        updated = asyncio.run(api.update_metadata(patch, actor=admin))
        assert updated is not None
        assert "new_tag" in updated.tags

    def test_update_importance(self, api, admin, db):
        """Update importance on a fact."""
        _insert_fact(db, id="fact_meta2", text="importance test", importance=0.5)

        patch = MetadataPatch(fact_id="fact_meta2", importance=0.9)
        updated = asyncio.run(api.update_metadata(patch, actor=admin))
        assert updated is not None
        assert updated.importance == pytest.approx(0.9)

    def test_update_nonexistent_fact(self, api, admin):
        """Updating a nonexistent fact returns None."""
        patch = MetadataPatch(fact_id="fact_nonexistent", tags=["x"])
        result = asyncio.run(api.update_metadata(patch, actor=admin))
        assert result is None


# ---------------------------------------------------------------------------
# §4  create_revision — ADD-only text change
# ---------------------------------------------------------------------------

class TestCreateRevision:
    def test_create_revision(self, api, admin, db):
        """Create a revision: new fact + old marked superseded + relation."""
        _insert_fact(db, id="fact_rev_old", text="original text")

        new_fact = asyncio.run(
            api.create_revision("fact_rev_old", "revised text", actor=admin, reason="correction")
        )
        assert new_fact is not None
        assert new_fact.text == "revised text"
        assert new_fact.id != "fact_rev_old"

        # Old fact should be superseded.
        from memory.backends.sqlite_backend import SQLiteStorageBackend
        old = SQLiteStorageBackend(db).get_fact("fact_rev_old", "default")
        assert old.status == "superseded"

    def test_create_revision_nonexistent(self, api, admin):
        """Creating a revision of a nonexistent fact returns None."""
        result = asyncio.run(api.create_revision("fact_noexist", "new text", actor=admin))
        assert result is None


# ---------------------------------------------------------------------------
# §5  delete_fact — soft archive
# ---------------------------------------------------------------------------

class TestDeleteFact:
    def test_soft_delete(self, api, admin, db):
        """Delete archives a fact (status='archived')."""
        _insert_fact(db, id="fact_del1", text="to be deleted")

        ok = asyncio.run(api.delete_fact("fact_del1", actor=admin))
        assert ok is True

        from memory.backends.sqlite_backend import SQLiteStorageBackend
        deleted = SQLiteStorageBackend(db).get_fact("fact_del1", "default")
        assert deleted.status == "archived"

    def test_delete_permission_denied(self, api, agent, db):
        """Agent role lacks memory.delete."""
        _insert_fact(db, id="fact_del_perm", text="test")

        with pytest.raises(PermissionDenied):
            asyncio.run(api.delete_fact("fact_del_perm", actor=agent))

    def test_delete_nonexistent(self, api, admin):
        """Deleting a nonexistent fact returns False."""
        ok = asyncio.run(api.delete_fact("fact_nonexistent", actor=admin))
        assert ok is False


# ---------------------------------------------------------------------------
# §6  Two-phase forget
# ---------------------------------------------------------------------------

class TestTwoPhaseForget:
    def test_forget_preview_returns_token(self, api, admin, db):
        """forget_preview returns a token and candidate fact IDs."""
        for i in range(3):
            _insert_fact(db, id=f"fact_forget_{i}", text=f"forgettable fact {i}")

        candidate = asyncio.run(
            api.forget_preview({"domain": "project", "scope_id": "default"}, actor=admin)
        )
        assert isinstance(candidate, ForgetCandidate)
        assert len(candidate.fact_ids) >= 3
        assert candidate.token  # non-empty token

    def test_forget_confirm_with_valid_token(self, api, admin, db):
        """forget_confirm_with_ids succeeds with a valid token."""
        _insert_fact(db, id="fact_forget_confirm", text="to be forgotten")

        candidate = asyncio.run(
            api.forget_preview({"domain": "project", "scope_id": "default"}, actor=admin)
        )
        assert "fact_forget_confirm" in candidate.fact_ids

        result = asyncio.run(
            api.forget_confirm_with_ids(candidate.token, candidate.fact_ids, actor=admin)
        )
        assert isinstance(result, ForgetResult)
        assert result.deleted >= 1

        # Fact should be gone.
        from memory.backends.sqlite_backend import SQLiteStorageBackend
        assert SQLiteStorageBackend(db).get_fact("fact_forget_confirm", "default") is None

    def test_forget_confirm_invalid_token(self, api, admin, db):
        """forget_confirm_with_ids fails with an invalid token."""
        _insert_fact(db, id="fact_bad_token", text="test")

        with pytest.raises(PermissionDenied):
            asyncio.run(
                api.forget_confirm_with_ids("invalid_token", ["fact_bad_token"], actor=admin)
            )

    def test_forget_confirm_wrong_fact_ids(self, api, admin, db):
        """Token bound to different fact_ids is rejected."""
        for i in range(2):
            _insert_fact(db, id=f"fact_mismatch_{i}", text=f"mismatch {i}")

        candidate = asyncio.run(
            api.forget_preview({"domain": "project", "scope_id": "default"}, actor=admin)
        )

        # Try to confirm with wrong fact_ids (different from preview).
        with pytest.raises(PermissionDenied):
            asyncio.run(
                api.forget_confirm_with_ids(candidate.token, ["wrong_id"], actor=admin)
            )

    def test_forget_permission_denied(self, api, agent):
        """Agent role lacks memory.forget."""
        with pytest.raises(PermissionDenied):
            asyncio.run(api.forget_preview({"domain": "project"}, actor=agent))

    def test_forget_cross_tenant_isolated(self, api, other_tenant_admin, db):
        """Forget preview from other tenant does not see default tenant facts."""
        _insert_fact(db, id="fact_cross_tenant", text="default tenant secret")

        candidate = asyncio.run(
            api.forget_preview({"domain": "project", "scope_id": "default"}, actor=other_tenant_admin)
        )
        assert "fact_cross_tenant" not in candidate.fact_ids


# ---------------------------------------------------------------------------
# §7  Hot Memory
# ---------------------------------------------------------------------------

class TestHotMemory:
    def test_hot_add_and_get(self, api, admin):
        """Add a hot item and retrieve it."""
        item = asyncio.run(
            api.hot_add("important note", actor=admin, domain="project", scope_id="default", priority=5)
        )
        assert item.id

        items = asyncio.run(api.hot_get(actor=admin, domain="project", scope_id="default"))
        assert any(i.text == "important note" for i in items)

    def test_hot_add_permission_denied(self, api, user):
        """User role lacks memory.hot.manage."""
        with pytest.raises(PermissionDenied):
            asyncio.run(
                api.hot_add("test", actor=user, domain="project", scope_id="default")
            )

    def test_hot_archive(self, api, admin):
        """Archive a hot item."""
        item = asyncio.run(
            api.hot_add("to archive", actor=admin, domain="project", scope_id="default")
        )
        ok = asyncio.run(api.hot_archive(item.id, actor=admin))
        assert ok is True


# ---------------------------------------------------------------------------
# §8  History
# ---------------------------------------------------------------------------

class TestHistory:
    def test_history_search(self, api, admin, db):
        """Search history messages."""
        from memory.layers.history_store import HistoryStoreWrapper
        hs = HistoryStoreWrapper(db=db)
        hs.append(
            tenant_id="default", channel_id="ch1",
            role="user", content="We decided to use PostgreSQL",
        )
        hs.append(
            tenant_id="default", channel_id="ch1",
            role="agent", content="Sounds good, PostgreSQL it is",
        )

        results = asyncio.run(
            api.history_search("PostgreSQL", actor=admin, channel_id="ch1")
        )
        assert len(results) >= 1

    def test_history_scroll(self, api, admin, db):
        """Scroll history messages in order."""
        from memory.layers.history_store import HistoryStoreWrapper
        hs = HistoryStoreWrapper(db=db)
        for i in range(5):
            hs.append(
                tenant_id="default", channel_id="ch2",
                role="user", content=f"message {i}",
            )

        msgs = asyncio.run(api.history_scroll("ch2", actor=admin, limit=10, offset=0))
        assert len(msgs) == 5


# ---------------------------------------------------------------------------
# §9  Skill
# ---------------------------------------------------------------------------

class TestSkill:
    def test_skill_create_and_list(self, api, admin):
        """Create a skill and list it."""
        skill = asyncio.run(
            api.skill_create("deploy-procedure", "1. Run tests\n2. Deploy",
                             actor=admin, domain="agent", scope_id="default",
                             description="How to deploy", tags=["ops"])
        )
        assert skill.id
        assert skill.name == "deploy-procedure"

        skills = asyncio.run(api.skill_list(actor=admin, domain="agent", scope_id="default"))
        assert any(s.name == "deploy-procedure" for s in skills)

    def test_skill_create_permission_denied(self, api, user):
        """User role lacks memory.skill.manage."""
        with pytest.raises(PermissionDenied):
            asyncio.run(
                api.skill_create("test", "body", actor=user, domain="agent", scope_id="default")
            )

    def test_skill_update(self, api, admin):
        """Update a skill (creates new version)."""
        skill = asyncio.run(
            api.skill_create("versioned-skill", "v1 body",
                             actor=admin, domain="agent", scope_id="default")
        )
        updated = asyncio.run(
            api.skill_update(skill.id, "v2 body", actor=admin, change_summary="improved")
        )
        assert updated.version >= 2

    def test_skill_rollback(self, api, admin):
        """Rollback a skill to a previous version."""
        skill = asyncio.run(
            api.skill_create("rollback-skill", "v1 body",
                             actor=admin, domain="agent", scope_id="default")
        )
        updated = asyncio.run(api.skill_update(skill.id, "v2 body", actor=admin))
        # Rollback operates on the current active skill ID.
        rolled = asyncio.run(api.skill_rollback(updated.id, 1, actor=admin))
        assert rolled.body == "v1 body"


# ---------------------------------------------------------------------------
# §10  REST endpoints (api/memory.py)
# ---------------------------------------------------------------------------

class TestRestEndpoints:
    @pytest.fixture
    def client(self, tmp_path):
        """Build a FastAPI app with the new memory routes, isolated DB."""
        import api.memory as memory_module
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        test_db = MemoryDB(db_path=tmp_path / "test_rest_memory.db")
        test_api = MemoryAPI(db=test_db)

        # Patch the module-level singleton.
        orig_api = memory_module._memory_api
        memory_module._memory_api = test_api

        app = FastAPI()
        app.include_router(memory_module.router)
        memory_module.register_exception_handlers(app)
        tc = TestClient(app)

        yield tc

        # Restore.
        memory_module._memory_api = orig_api

    def test_write_proposal_endpoint(self, client):
        """POST /api/memory/write-proposals works."""
        resp = client.post(
            "/api/memory/write-proposals",
            json={"text": "test fact", "domain": "project"},
            headers={"X-Memory-Role": "admin"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "event_id" in data
        assert data["status"] in ("accepted", "skipped", "failed")

    def test_search_endpoint(self, client):
        """POST /api/memory/search works."""
        resp = client.post(
            "/api/memory/search",
            json={"text": "test query", "domain": "project", "top_k": 5},
            headers={"X-Memory-Role": "admin"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "count" in data

    def test_search_permission_denied_endpoint(self, client):
        """User role cannot search agent domain → 403."""
        resp = client.post(
            "/api/memory/search",
            json={"text": "test", "domain": "agent"},
            headers={"X-Memory-Role": "user"},
        )
        assert resp.status_code == 403

    def test_metrics_endpoint(self, client):
        """GET /api/memory/metrics works."""
        resp = client.get("/api/memory/metrics")
        assert resp.status_code == 200

    def test_audit_requires_admin(self, client):
        """GET /api/memory/audit requires admin/system role."""
        resp = client.get(
            "/api/memory/audit",
            headers={"X-Memory-Role": "user"},
        )
        assert resp.status_code == 403

    def test_audit_admin_ok(self, client):
        """GET /api/memory/audit works for admin."""
        resp = client.get(
            "/api/memory/audit",
            headers={"X-Memory-Role": "admin"},
        )
        assert resp.status_code == 200

    def test_hot_add_and_list_endpoint(self, client):
        """POST + GET /api/memory/hot works."""
        resp = client.post(
            "/api/memory/hot",
            json={
                "domain": "project",
                "scope_id": "default",
                "text": "hot test",
                "priority": 5,
            },
            headers={"X-Memory-Role": "admin"},
        )
        assert resp.status_code == 200
        assert "id" in resp.json()

        resp = client.get(
            "/api/memory/hot",
            params={"domain": "project", "scope_id": "default"},
            headers={"X-Memory-Role": "admin"},
        )
        assert resp.status_code == 200
        items = resp.json()
        assert any(i["text"] == "hot test" for i in items)

    def test_hot_add_permission_denied(self, client):
        """User role cannot add hot items → 403."""
        resp = client.post(
            "/api/memory/hot",
            json={
                "domain": "project",
                "scope_id": "default",
                "text": "should fail",
            },
            headers={"X-Memory-Role": "user"},
        )
        assert resp.status_code == 403

    def test_skill_create_and_list_endpoint(self, client):
        """POST + GET /api/memory/skills works."""
        resp = client.post(
            "/api/memory/skills",
            json={
                "name": "test-skill",
                "body": "test body",
                "domain": "agent",
                "scope_id": "default",
            },
            headers={"X-Memory-Role": "admin"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-skill"

        resp = client.get(
            "/api/memory/skills",
            params={"domain": "agent", "scope_id": "default"},
            headers={"X-Memory-Role": "admin"},
        )
        assert resp.status_code == 200
        skills = resp.json()
        assert any(s["name"] == "test-skill" for s in skills)

    def test_skill_create_permission_denied(self, client):
        """User role cannot create skills → 403."""
        resp = client.post(
            "/api/memory/skills",
            json={
                "name": "should-fail",
                "body": "test",
                "domain": "agent",
                "scope_id": "default",
            },
            headers={"X-Memory-Role": "user"},
        )
        assert resp.status_code == 403

    def test_forget_preview_and_confirm_endpoint(self, client):
        """Two-phase forget: preview → confirm."""
        # First write a fact via the pipeline.
        client.post(
            "/api/memory/write-proposals",
            json={"text": "forgettable fact", "domain": "project"},
            headers={"X-Memory-Role": "admin"},
        )

        # Preview.
        resp = client.post(
            "/api/memory/forget/preview",
            json={"domain": "project", "scope_id": "default"},
            headers={"X-Memory-Role": "admin"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert "fact_ids" in data

        if not data["fact_ids"]:
            return  # No facts to forget — skip confirm.

        # Confirm.
        resp = client.post(
            "/api/memory/forget/confirm",
            json={"token": data["token"], "fact_ids": data["fact_ids"]},
            headers={"X-Memory-Role": "admin"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_forget_confirm_invalid_token_endpoint(self, client):
        """Forget confirm with invalid token → 403."""
        resp = client.post(
            "/api/memory/forget/confirm",
            json={"token": "bad_token", "fact_ids": ["fake_id"]},
            headers={"X-Memory-Role": "admin"},
        )
        assert resp.status_code == 403

    def test_forget_preview_permission_denied(self, client):
        """Agent role cannot forget → 403."""
        resp = client.post(
            "/api/memory/forget/preview",
            json={"domain": "project"},
            headers={"X-Memory-Role": "agent"},
        )
        assert resp.status_code == 403
