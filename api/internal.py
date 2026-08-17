"""Internal callback endpoint for MCP-driven delegation.

The MCP delegate server (run as a child of a Claude Code subprocess) posts here
when an agent calls the `delegate` tool. We look up the caller's delegation
context by token, validate it, persist a background work order, and immediately
return the child task number. The parent is resumed after the child finishes.

Loopback-only. Not part of the public API surface.
"""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from config import settings
from core.delegate_registry import delegate_registry
from core.registry import materialize, registry
from models.schemas import DelegationRecord

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal")


class DelegateRequest(BaseModel):
    token: str
    target: str
    task: str
    access: str = "read"


class DelegateResponse(BaseModel):
    result: str = ""
    error: str | None = None
    child_run_id: str | None = None
    agent_id: str | None = None
    agent_name: str | None = None
    status: str | None = None
    artifacts: list[dict] = Field(default_factory=list)


class ProgressRequest(BaseModel):
    token: str
    message: str
    kind: str = "checkpoint"


class MemorySearchRequest(BaseModel):
    token: str
    query: str
    domain: str = "agent"
    scope_id: str = ""
    top_k: int = 5


class MemorySearchResponse(BaseModel):
    results: list[dict] = Field(default_factory=list)
    error: str | None = None


# --- Wiki endpoints (agent-facing, token-authenticated) ---

class WikiRequest(BaseModel):
    token: str
    title: str | None = None
    content: str | None = None
    node_token: str | None = None


def _is_loopback(request: Request) -> bool:
    client = request.client.host if request.client else ""
    return client in ("127.0.0.1", "::1", "localhost", "testclient")


async def _require_capability(token: str, capability: str):
    ctx = await delegate_registry.get(token)
    if ctx is None:
        return None, "invalid or expired token"
    if capability not in ctx.mcp_capabilities:
        return None, f"capability not granted: {capability}"
    return ctx, None


async def _emit(ctx, kind: str, data: dict) -> None:
    if not ctx.on_progress:
        return
    try:
        await ctx.on_progress(kind, data)
    except Exception:
        logger.warning("progress callback failed for %s", kind, exc_info=True)


@router.post("/progress", response_model=DelegateResponse)
async def report_progress(req: ProgressRequest, request: Request):
    if not _is_loopback(request):
        return DelegateResponse(error="forbidden: loopback only")
    ctx, error = await _require_capability(req.token, "report_progress")
    if error:
        return DelegateResponse(error=error)
    if not settings.checkpoint_delivery:
        return DelegateResponse(result="ok", status="suppressed")
    from core.output_safety import sanitize_agent_text

    safe = sanitize_agent_text(req.message)
    if not safe.text.strip():
        return DelegateResponse(error="progress message required")
    event_kind = req.kind if req.kind in {"checkpoint", "stalled"} else "checkpoint"
    await _emit(
        ctx,
        event_kind,
        {
            "title": safe.text[:500],
            "phase": ctx.phase,
            "agent_id": ctx.caller_agent_id or "",
            "agent_name": ctx.caller_agent_name,
            "child_run_id": ctx.invocation_id,
            "safety_findings": list(safe.findings),
        },
    )
    return DelegateResponse(result="ok", status="recorded")


@router.post("/wiki/write", response_model=DelegateResponse)
async def wiki_write(req: WikiRequest, request: Request):
    if not _is_loopback(request):
        return DelegateResponse(error="forbidden: loopback only")
    ctx, error = await _require_capability(req.token, "wiki_write")
    if error:
        return DelegateResponse(error=error)
    project_id = ctx.project_id
    if not project_id:
        return DelegateResponse(error="project context required")
    if not req.title or req.content is None:
        return DelegateResponse(error="title and content required")
    try:
        from integrations.feishu.wiki_client import feishu_wiki_client
        from core.project_store import project_store
        from core.output_safety import sanitize_agent_text
        project = await project_store.get(project_id)
        name = project.name if project else "CatTogether"
        space_id = await feishu_wiki_client.ensure_space(project_id, name)
        safe_title = sanitize_agent_text(req.title).text
        safe_content = sanitize_agent_text(req.content)
        node_token, url = await feishu_wiki_client.create_doc_node(space_id, safe_title, safe_content.text)
        artifact = {
            "artifact_id": node_token,
            "type": "feishu_wiki",
            "title": safe_title,
            "url": url,
            "status": "created",
        }
        ctx.artifacts.append(artifact)
        await _emit(ctx, "artifact_created", {
            "title": f"Wiki 文档已生成：{safe_title}",
            "detail": (
                "文档内容中的敏感凭据样式已隐藏；如为真实凭据请立即轮换。"
                if safe_content.findings else ""
            ),
            "url": url,
            "artifact_id": node_token,
            "artifact_type": "feishu_wiki",
            "status": "created",
            "agent_id": ctx.caller_agent_id or "",
            "agent_name": ctx.caller_agent_name,
            "child_run_id": ctx.invocation_id,
        })
        return DelegateResponse(result=url, status="created", artifacts=[artifact])
    except Exception as exc:
        from core.output_safety import sanitize_agent_text
        message = sanitize_agent_text(str(exc)).text
        await _emit(ctx, "artifact_failed", {
            "title": "Wiki 文档创建失败",
            "detail": message,
            "agent_id": ctx.caller_agent_id or "",
            "agent_name": ctx.caller_agent_name,
            "child_run_id": ctx.invocation_id,
            "artifact_type": "feishu_wiki",
            "status": "failed",
        })
        return DelegateResponse(error=message, status="failed")


@router.post("/wiki/append", response_model=DelegateResponse)
async def wiki_append(req: WikiRequest, request: Request):
    if not _is_loopback(request):
        return DelegateResponse(error="forbidden: loopback only")
    ctx, error = await _require_capability(req.token, "wiki_append")
    if error:
        return DelegateResponse(error=error)
    if not req.node_token or req.content is None:
        return DelegateResponse(error="node_token and content required")
    try:
        from integrations.feishu.wiki_client import feishu_wiki_client
        from core.output_safety import sanitize_agent_text
        safe_content = sanitize_agent_text(req.content)
        await feishu_wiki_client.append_content(req.node_token, safe_content.text)
        url = await feishu_wiki_client.node_url(req.node_token)
        artifact = {
            "artifact_id": req.node_token,
            "type": "feishu_wiki",
            "title": req.title or "已更新的 Wiki 文档",
            "url": url,
            "status": "updated",
        }
        ctx.artifacts.append(artifact)
        await _emit(ctx, "artifact_created", {
            "title": "Wiki 文档已更新",
            "detail": (
                "新增内容中的敏感凭据样式已隐藏；如为真实凭据请立即轮换。"
                if safe_content.findings else ""
            ),
            "artifact_id": req.node_token,
            "artifact_type": "feishu_wiki",
            "status": "updated",
            "url": url,
            "agent_id": ctx.caller_agent_id or "",
            "agent_name": ctx.caller_agent_name,
            "child_run_id": ctx.invocation_id,
        })
        return DelegateResponse(result=url, status="updated", artifacts=[artifact])
    except Exception as exc:
        from core.output_safety import sanitize_agent_text
        message = sanitize_agent_text(str(exc)).text
        await _emit(ctx, "artifact_failed", {
            "title": "Wiki 文档更新失败",
            "detail": message,
            "artifact_id": req.node_token,
            "agent_id": ctx.caller_agent_id or "",
            "agent_name": ctx.caller_agent_name,
            "child_run_id": ctx.invocation_id,
            "artifact_type": "feishu_wiki",
            "status": "failed",
        })
        return DelegateResponse(error=message, status="failed")


@router.post("/wiki/read", response_model=DelegateResponse)
async def wiki_read(req: WikiRequest, request: Request):
    if not _is_loopback(request):
        return DelegateResponse(error="forbidden: loopback only")
    ctx, error = await _require_capability(req.token, "wiki_read")
    if error:
        return DelegateResponse(error=error)
    if not req.node_token:
        return DelegateResponse(error="node_token required")
    try:
        from integrations.feishu.wiki_client import feishu_wiki_client
        content = await feishu_wiki_client.read_node(req.node_token)
        return DelegateResponse(result=content)
    except Exception as exc:
        from core.output_safety import sanitize_agent_text
        return DelegateResponse(error=sanitize_agent_text(str(exc)).text)


@router.post("/wiki/list", response_model=DelegateResponse)
async def wiki_list(request: Request, req: WikiRequest):
    if not _is_loopback(request):
        return DelegateResponse(error="forbidden: loopback only")
    ctx, error = await _require_capability(req.token, "wiki_list")
    if error:
        return DelegateResponse(error=error)
    project_id = ctx.project_id
    if not project_id:
        return DelegateResponse(error="project context required")
    try:
        from integrations.feishu.wiki_client import feishu_wiki_client
        from core.project_store import project_store
        project = await project_store.get(project_id)
        if not project or not project.wiki_space_id:
            return DelegateResponse(result="（项目尚无 Wiki 文档）")
        nodes = await feishu_wiki_client.list_nodes(project.wiki_space_id)
        lines = [f"- {n['title']}：{n['url']}" for n in nodes]
        return DelegateResponse(result="\n".join(lines) or "（空间内暂无文档）")
    except Exception as exc:
        from core.output_safety import sanitize_agent_text
        return DelegateResponse(error=sanitize_agent_text(str(exc)).text)


@router.post("/memory/search", response_model=MemorySearchResponse)
async def memory_search(req: MemorySearchRequest, request: Request):
    """Read-only memory search for agents (Agent boundary reversal, 2.1 R4).

    Agents call this via the MCP ``memory_search`` tool.  The token identifies
    the caller's delegation context; we use its project_id / user_id / agent_id
    to scope the search.  Only reads are exposed — writes go through the
    admission pipeline.

    Memory 2.2: when ``CT_MEMORY_V22_ENABLED`` is on, the search is served
    from Qdrant (``MemoryStore.search``, dense + bm25 sparse RRF) and the
    domain is a filter tag, not a hard scope. When off, the v2.1 SQLite
    HybridRetriever path is used unchanged.
    """
    if not _is_loopback(request):
        return MemorySearchResponse(error="forbidden: loopback only")
    ctx, error = await _require_capability(req.token, "memory_search")
    if error:
        return MemorySearchResponse(error=error)

    from core.output_safety import sanitize_agent_text

    safe_query = sanitize_agent_text(req.query).text
    if not safe_query.strip():
        return MemorySearchResponse(error="query required")

    top_k = max(1, min(req.top_k, 20))
    domain = (req.domain or "agent").strip().lower()
    if domain == "workspace":
        domain = "project"

    # task domain has no entity today — return empty in both v2.1 and v2.2.
    if domain == "task":
        return MemorySearchResponse(results=[])

    # ----- Memory 2.2: Qdrant-backed search -----
    from config import settings
    if settings.memory_v22_enabled:
        try:
            from memory.v22 import get_memory_store
            store = get_memory_store()
            if store is None:
                return MemorySearchResponse(error="memory v22 store unavailable")
            from config import settings as _s
            hits = store.search(
                safe_query,
                domains=[domain],
                top_k=top_k,
                weight_vector=_s.memory22_rrf_weight_vector,
                weight_bm25=_s.memory22_rrf_weight_bm25,
            )
            # Tool form — for the calling (working) agent. Payload form
            # (uuid + linked ids) is only used internally by the extractor.
            from memory.v22.render import MemoryRenderer
            return MemorySearchResponse(results=MemoryRenderer.for_tool(hits))
        except Exception as exc:
            logger.warning("v22 memory_search failed: %s", exc, exc_info=True)
            return MemorySearchResponse(error=sanitize_agent_text(str(exc)).text)

    # ----- Memory 2.1: SQLite-backed search (fallback when v22 off) -----
    from memory.memory_api import MemoryAPI
    from memory.models import RetrievalQuery
    from memory.permissions import ActorContext
    from memory.scope import canonical_scope_id, normalize_agent_role, tenant_id_from_user_id

    # Resolve scope_id from the request or fall back to the caller's context.
    scope_id = req.scope_id or ""
    if domain == "project":
        scope_id = scope_id or ctx.project_id or ""
    elif domain == "user":
        scope_id = scope_id or ctx.user_id or "default"
    else:  # agent
        scope_id = scope_id or normalize_agent_role(ctx.caller_agent_id or "custom")

    if not scope_id:
        return MemorySearchResponse(error=f"scope_id required for domain {domain}")

    tenant_id = tenant_id_from_user_id(ctx.user_id)
    scope_id = canonical_scope_id(domain, scope_id)
    actor = ActorContext(
        tenant_id=tenant_id,
        role="agent",
        actor_id=ctx.caller_agent_id or "agent",
        display_name=ctx.caller_agent_id or "agent",
    )

    try:
        query = RetrievalQuery(
            text=safe_query,
            tenant_id=tenant_id,
            domain=domain,  # type: ignore[arg-type]
            scope_id=scope_id,
            intent="recall",
            top_k=top_k,
        )
        results_list = await MemoryAPI().search(query, actor=actor)
    except Exception as exc:
        logger.warning("memory_search failed: %s", exc, exc_info=True)
        return MemorySearchResponse(error=sanitize_agent_text(str(exc)).text)

    results_v21: list[dict] = []
    for r in results_list:
        fact = r.fact
        results_v21.append({
            "id": fact.id,
            "text": fact.text,
            "kind": fact.kind,
            "domain": fact.domain,
            "tags": list(fact.tags),
            "importance": fact.importance,
            "confidence": fact.confidence,
            "created_at": fact.created_at.isoformat() if fact.created_at else None,
        })
    return MemorySearchResponse(results=results_v21)


@router.post("/delegate", response_model=DelegateResponse)
async def delegate(req: DelegateRequest, request: Request):
    # Loopback-only: never expose delegation over the network.
    if not _is_loopback(request):
        return DelegateResponse(error="forbidden: loopback only")

    ctx, capability_error = await _require_capability(req.token, "delegate")
    if capability_error:
        return DelegateResponse(error=capability_error)

    if ctx.depth >= delegate_registry.max_depth:
        return DelegateResponse(
            error=f"委派深度超过上限（{delegate_registry.max_depth}），拒绝进一步委派。"
        )

    # Block self-delegation: an agent must not pull itself in (would loop).
    if ctx.caller_agent_id and req.target == ctx.caller_agent_id:
        return DelegateResponse(
            error=f"不能委派给自己（{req.target}）。请选别的同事，或自己直接完成。"
        )

    # Internal agents (e.g. the memory extractor) are not delegate targets.
    from agents.memory_agent import is_internal
    if is_internal(req.target):
        return DelegateResponse(error=f"{req.target} 是内部 agent，不可被委派")

    # Resolve the target agent: project spec first, then global registry.
    target_agent = None
    spec = None
    if ctx.project_id:
        from core.agent_store import agent_store
        spec = await agent_store.get(ctx.project_id, req.target)
        if spec and not spec.delegatable:
            return DelegateResponse(error=f"agent {req.target} 不可被委派（delegatable=false）")
        if spec:
            target_agent = materialize(spec)
    if target_agent is None:
        target_agent = registry.get(req.target)
    if target_agent is None:
        return DelegateResponse(error=f"未找到 agent：{req.target}")

    target_name = spec.name if spec else target_agent.name
    access = req.access if req.access in ("read", "write") else "read"
    if access == "write" and getattr(spec, "sandbox", "read") != "workspace-write":
        return DelegateResponse(error=f"agent {req.target} 没有工作空间写权限")

    child_run_id = uuid.uuid4().hex
    await _emit(ctx, "delegate_requested", {
        "title": f"正在邀请 {target_name} 协作",
        "detail": req.task,
        "target": req.target,
        "child_run_id": child_run_id,
        "agent_id": ctx.caller_agent_id or "",
        "status": "requested",
    })

    if not ctx.run_id:
        return DelegateResponse(error="delegate requires a persisted root Run")
    from core.run_supervisor import run_supervisor
    if not await run_supervisor.accepting(ctx.run_id):
        return DelegateResponse(error="父任务正在结束，拒绝创建新的子任务")

    from core.run_store import run_store
    saved = await run_store.add_delegation(
        ctx.run_id,
        DelegationRecord(
            id=child_run_id,
            parent_invocation_id=ctx.invocation_id,
            target_agent_id=req.target,
            task=req.task,
            access=access,
            depth=ctx.depth + 1,
        ),
    )
    if saved is None:
        return DelegateResponse(error="父任务已经结束，子任务未接单")

    from core.task_queue import feishu_task_queue
    queued = await feishu_task_queue.enqueue_delegation(ctx.run_id, child_run_id)
    if not queued:
        await run_store.update_delegation(
            ctx.run_id,
            child_run_id,
            status="failed",
            error="后台工作队列已满",
        )
        return DelegateResponse(error="后台工作队列已满，子任务未启动")

    return DelegateResponse(
        result=f"子任务已接单：{child_run_id}。系统会在完成后重新唤醒你验收。",
        child_run_id=child_run_id,
        agent_id=target_agent.agent_id, agent_name=target_name,
        status="queued", artifacts=list(ctx.artifacts or []),
    )
