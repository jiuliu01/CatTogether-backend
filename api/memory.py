"""Memory 2.1 REST API — final routes (§10).

Replaces the v3 memory routes that were in ``api/rest.py``.  All routes go
through the single ``MemoryAPI`` service layer, which enforces the agent
boundary, ADD-only writes, two-phase forget, and permission checks.

Route map (§10):
    POST   /api/memory/write-proposals
    POST   /api/memory/search
    PATCH  /api/memory/facts/{fact_id}/metadata
    POST   /api/memory/facts/{fact_id}/revisions
    DELETE /api/memory/facts/{fact_id}
    POST   /api/memory/forget/preview
    POST   /api/memory/forget/confirm
    GET    /api/memory/hot
    POST   /api/memory/hot
    POST   /api/memory/hot/{item_id}/approve
    DELETE /api/memory/hot/{item_id}
    POST   /api/memory/history/search
    GET    /api/memory/history/scroll
    GET    /api/memory/skills
    POST   /api/memory/skills
    GET    /api/memory/skills/{skill_id}
    PATCH  /api/memory/skills/{skill_id}
    POST   /api/memory/skills/{skill_id}/approve
    POST   /api/memory/skills/{skill_id}/rollback
    GET    /api/memory/metrics
    GET    /api/memory/audit
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from memory.memory_api import (
    MemoryAPI,
    WriteProposal,
    MetadataPatch,
)
from memory.models import RetrievalQuery
from memory.permissions import ActorContext, PermissionDenied
from memory.tenant_auth import get_actor_context


router = APIRouter(prefix="/api/memory")


# Single service-layer instance.
_memory_api = MemoryAPI()


def register_exception_handlers(app) -> None:
    """Register Memory-specific exception handlers on the FastAPI app.

    Call this from main.py after including the router.  Converts
    ``PermissionDenied`` → HTTP 403 Forbidden.
    """
    from fastapi.responses import JSONResponse

    @app.exception_handler(PermissionDenied)
    async def _permission_denied_handler(request, exc: PermissionDenied):
        return JSONResponse(
            status_code=403,
            content={"detail": str(exc)},
        )


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class WriteProposalBody(BaseModel):
    text: str
    domain: str | None = None
    scope_hint: str | None = None
    kind: str | None = None
    tags: list[str] = []
    importance: float = 0.5
    confidence: float = 0.5
    source_type: str = "agent_result"
    source_ids: list[str] = []


class SearchBody(BaseModel):
    text: str
    domain: str | None = None
    scope_id: str | None = None
    top_k: int = 10
    intent: str = "recall"


class MetadataBody(BaseModel):
    tags: list[str] | None = None
    importance: float | None = None
    confidence: float | None = None


class RevisionBody(BaseModel):
    new_text: str
    reason: str = ""


class ForgetPreviewBody(BaseModel):
    domain: str | None = None
    scope_id: str | None = None
    kind: str | None = None
    older_than_days: int | None = None


class ForgetConfirmBody(BaseModel):
    token: str
    fact_ids: list[str]


class HotAddBody(BaseModel):
    domain: str
    scope_id: str
    text: str
    priority: int = 0


class HistorySearchBody(BaseModel):
    query: str
    channel_id: str | None = None
    limit: int = 20


class SkillCreateBody(BaseModel):
    name: str
    body: str
    domain: str = "agent"
    scope_id: str = "default"
    description: str = ""
    tags: list[str] = []


class SkillUpdateBody(BaseModel):
    body: str
    change_summary: str = ""


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _fact_to_dict(fact: Any) -> dict[str, Any]:
    return {
        "id": fact.id,
        "tenant_id": fact.tenant_id,
        "domain": fact.domain,
        "scope_id": fact.scope_id,
        "kind": fact.kind,
        "text": fact.text,
        "tags": list(fact.tags),
        "importance": fact.importance,
        "confidence": fact.confidence,
        "status": fact.status,
        "created_at": fact.created_at.isoformat() if fact.created_at else None,
        "updated_at": fact.updated_at.isoformat() if fact.updated_at else None,
    }


def _result_to_dict(r: Any) -> dict[str, Any]:
    return {
        "fact": _fact_to_dict(r.fact),
        "score": r.score,
        "signals": list(r.signals),
    }


def _skill_to_dict(s: Any) -> dict[str, Any]:
    return {
        "id": s.id,
        "domain": s.domain,
        "scope_id": s.scope_id,
        "name": s.name,
        "description": s.description,
        "body": getattr(s, "body", ""),
        "version": s.version,
        "tags": list(s.tags),
        "status": s.status,
    }


def _hot_to_dict(item: Any) -> dict[str, Any]:
    return {
        "id": item.id,
        "domain": item.domain,
        "scope_id": item.scope_id,
        "text": item.text,
        "priority": getattr(item, "priority", 0),
        "status": item.status,
        "source_fact_ids": list(getattr(item, "source_fact_ids", []) or []),
    }


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

@router.post("/write-proposals")
async def write_proposal(
    body: WriteProposalBody,
    actor: ActorContext = Depends(get_actor_context),
):
    """Submit a write proposal through the nine-stage pipeline (§7).

    This is the *only* external write entry point.  Agents can propose;
    the pipeline decides final status.
    """
    proposal = WriteProposal(
        text=body.text,
        domain=body.domain,
        scope_hint=body.scope_hint,
        kind=body.kind,
        tags=body.tags,
        importance=body.importance,
        confidence=body.confidence,
        source_type=body.source_type,
        source_ids=body.source_ids,
        actor_id=actor.actor_id,
    )
    resp = await _memory_api.write_propose(proposal, actor=actor)
    return {
        "event_id": resp.event_id,
        "fact_ids": list(resp.fact_ids),
        "status": resp.status,
        "reason": resp.reason,
    }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@router.post("/search")
async def search(
    body: SearchBody,
    actor: ActorContext = Depends(get_actor_context),
):
    """Three-signal hybrid retrieval (§8)."""
    domain = body.domain
    if domain == "workspace":
        domain = "project"
    query = RetrievalQuery(
        text=body.text,
        tenant_id=actor.tenant_id,
        domain=domain,  # type: ignore[arg-type]
        scope_id=body.scope_id,
        intent=body.intent,  # type: ignore[arg-type]
        top_k=body.top_k,
    )
    results = await _memory_api.search(query, actor=actor)
    return {"results": [_result_to_dict(r) for r in results], "count": len(results)}


# ---------------------------------------------------------------------------
# Fact metadata / revision / delete
# ---------------------------------------------------------------------------

@router.patch("/facts/{fact_id}/metadata")
async def update_metadata(
    fact_id: str,
    body: MetadataBody,
    actor: ActorContext = Depends(get_actor_context),
):
    """Update fact metadata (tags/importance/confidence only — ADD-only, R4)."""
    patch = MetadataPatch(
        fact_id=fact_id,
        tags=body.tags,
        importance=body.importance,
        confidence=body.confidence,
    )
    updated = await _memory_api.update_metadata(patch, actor=actor)
    if updated is None:
        raise HTTPException(404, "fact not found or permission denied")
    return _fact_to_dict(updated)


@router.post("/facts/{fact_id}/revisions")
async def create_revision(
    fact_id: str,
    body: RevisionBody,
    actor: ActorContext = Depends(get_actor_context),
):
    """Create a new fact revision (ADD-only text change, R4)."""
    new_fact = await _memory_api.create_revision(fact_id, body.new_text, actor=actor, reason=body.reason)
    if new_fact is None:
        raise HTTPException(404, "fact not found")
    return _fact_to_dict(new_fact)


@router.delete("/facts/{fact_id}")
async def delete_fact(
    fact_id: str,
    actor: ActorContext = Depends(get_actor_context),
):
    """Soft delete (archive) a fact. Requires memory.delete."""
    ok = await _memory_api.delete_fact(fact_id, actor=actor)
    if not ok:
        raise HTTPException(404, "fact not found")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Two-phase forget (GDPR hard delete)
# ---------------------------------------------------------------------------

@router.post("/forget/preview")
async def forget_preview(
    body: ForgetPreviewBody,
    actor: ActorContext = Depends(get_actor_context),
):
    """Preview facts that would be forgotten. Returns a short-lived token."""
    criteria: dict[str, Any] = {}
    if body.domain is not None:
        criteria["domain"] = body.domain
    if body.scope_id is not None:
        criteria["scope_id"] = body.scope_id
    if body.kind is not None:
        criteria["kind"] = body.kind
    if body.older_than_days is not None:
        criteria["older_than_days"] = body.older_than_days
    candidate = await _memory_api.forget_preview(criteria, actor=actor)
    return {
        "fact_ids": candidate.fact_ids,
        "count": len(candidate.fact_ids),
        "token": candidate.token,
    }


@router.post("/forget/confirm")
async def forget_confirm(
    body: ForgetConfirmBody,
    actor: ActorContext = Depends(get_actor_context),
):
    """Confirm a forget operation using the token from preview."""
    result = await _memory_api.forget_confirm_with_ids(
        body.token, body.fact_ids, actor=actor,
    )
    return {"deleted": result.deleted, "ok": True}


# ---------------------------------------------------------------------------
# Hot Memory
# ---------------------------------------------------------------------------

@router.get("/hot")
async def hot_list(
    actor: ActorContext = Depends(get_actor_context),
    domain: str | None = None,
    scope_id: str | None = None,
):
    """List hot memory items for a scope."""
    d = domain or "project"
    if d == "workspace":
        d = "project"
    sid = scope_id or "default"
    items = await _memory_api.hot_get(actor=actor, domain=d, scope_id=sid)
    return [_hot_to_dict(it) for it in items]


@router.post("/hot")
async def hot_add(
    body: HotAddBody,
    actor: ActorContext = Depends(get_actor_context),
):
    """Add a hot memory item. Requires memory.hot.manage."""
    d = body.domain
    if d == "workspace":
        d = "project"
    item = await _memory_api.hot_add(
        body.text, actor=actor, domain=d, scope_id=body.scope_id,
        priority=body.priority,
    )
    return {"id": item.id, "status": item.status}


@router.post("/hot/{item_id}/approve")
async def hot_approve(
    item_id: str,
    actor: ActorContext = Depends(get_actor_context),
):
    """Approve a pending hot item. Requires memory.hot.approve."""
    ok = await _memory_api.hot_approve(item_id, actor=actor)
    if not ok:
        raise HTTPException(404, "pending hot item not found")
    return {"ok": True}


@router.delete("/hot/{item_id}")
async def hot_archive(
    item_id: str,
    actor: ActorContext = Depends(get_actor_context),
):
    """Archive a hot item. Requires memory.hot.manage."""
    ok = await _memory_api.hot_archive(item_id, actor=actor)
    if not ok:
        raise HTTPException(404, "hot item not found")
    return {"ok": True}


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@router.post("/history/search")
async def history_search(
    body: HistorySearchBody,
    actor: ActorContext = Depends(get_actor_context),
):
    """FTS5 search over session messages."""
    msgs = await _memory_api.history_search(
        body.query, actor=actor, channel_id=body.channel_id, limit=body.limit,
    )
    return [
        {
            "id": m.id,
            "channel_id": m.channel_id,
            "role": m.role,
            "content": m.content,
            "thread_id": getattr(m, "thread_id", None),
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in msgs
    ]


@router.get("/history/scroll")
async def history_scroll(
    channel_id: str,
    actor: ActorContext = Depends(get_actor_context),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Scroll history messages in chronological order."""
    msgs = await _memory_api.history_scroll(
        channel_id, actor=actor, limit=limit, offset=offset,
    )
    return [
        {
            "id": m.id,
            "channel_id": m.channel_id,
            "role": m.role,
            "content": m.content,
            "thread_id": getattr(m, "thread_id", None),
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in msgs
    ]


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------

@router.get("/skills")
async def skill_list(
    actor: ActorContext = Depends(get_actor_context),
    domain: str | None = None,
    scope_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    """List skills."""
    d = domain
    if d == "workspace":
        d = "project"
    skills = await _memory_api.skill_list(actor=actor, domain=d, scope_id=scope_id, limit=limit)
    return [_skill_to_dict(s) for s in skills]


@router.post("/skills")
async def skill_create(
    body: SkillCreateBody,
    actor: ActorContext = Depends(get_actor_context),
):
    """Create a skill. Requires memory.skill.manage."""
    d = body.domain
    if d == "workspace":
        d = "project"
    try:
        skill = await _memory_api.skill_create(
            body.name, body.body, actor=actor,
            domain=d, scope_id=body.scope_id,
            description=body.description, tags=body.tags,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"id": skill.id, "name": skill.name, "version": skill.version}


@router.get("/skills/{skill_id}")
async def skill_get(
    skill_id: str,
    actor: ActorContext = Depends(get_actor_context),
):
    """Get a skill by ID."""
    from memory.skill_store import SkillStore as _SkillStoreImpl
    from memory.db import memory_db
    skill = _SkillStoreImpl(memory_db).get_skill_by_id(skill_id)
    if skill is None or skill.tenant_id != actor.tenant_id:
        raise HTTPException(404, "skill not found")
    return _skill_to_dict(skill)


@router.patch("/skills/{skill_id}")
async def skill_update(
    skill_id: str,
    body: SkillUpdateBody,
    actor: ActorContext = Depends(get_actor_context),
):
    """Update a skill (creates a new version). Requires memory.skill.manage."""
    try:
        skill = await _memory_api.skill_update(
            skill_id, body.body, actor=actor, change_summary=body.change_summary,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"id": skill.id, "name": skill.name, "version": skill.version}


@router.post("/skills/{skill_id}/approve")
async def skill_approve(
    skill_id: str,
    actor: ActorContext = Depends(get_actor_context),
):
    """Approve a pending skill. Requires memory.skill.manage."""
    skill = await _memory_api.skill_approve(skill_id, actor=actor)
    if skill is None:
        raise HTTPException(404, "skill not found")
    return {"id": skill.id, "name": skill.name, "status": skill.status}


@router.post("/skills/{skill_id}/rollback")
async def skill_rollback(
    skill_id: str,
    to_version: int,
    actor: ActorContext = Depends(get_actor_context),
):
    """Rollback a skill to a prior version. Requires memory.skill.manage."""
    try:
        skill = await _memory_api.skill_rollback(skill_id, to_version, actor=actor)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"id": skill.id, "name": skill.name, "version": skill.version}


# ---------------------------------------------------------------------------
# Metrics & Audit
# ---------------------------------------------------------------------------

@router.get("/metrics")
async def metrics():
    """Get memory system metrics snapshot."""
    from memory.monitoring import memory_metrics
    return memory_metrics.snapshot()


@router.get("/audit")
async def audit(
    actor: ActorContext = Depends(get_actor_context),
    action: str | None = None,
    fact_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
):
    """Query audit log. Requires admin/system role."""
    if actor.role not in ("admin", "system"):
        raise HTTPException(403, "audit log access requires admin or system role")
    from memory.audit_log import AuditLogger
    return AuditLogger(db=_memory_api._db).query(
        actor.tenant_id, action=action, fact_id=fact_id, limit=limit,
    )
