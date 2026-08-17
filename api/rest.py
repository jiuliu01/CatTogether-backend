"""REST API: management endpoints (non-streaming).

Streaming chat happens over WebSocket (api/ws.py). These endpoints cover agents,
channels, messages history, memory, and workspaces.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agents.llm.llm_agent import LLMAgent
from agents.custom.custom_agent import CustomAgent
from core.registry import registry
from agents.roles import resolve_role_agents
from core.session_manager import session_manager
from core.workspace import workspace_manager
from core.run_store import run_store
from core.run_event_store import run_event_store
from core.project_store import project_store
from core.task_queue import feishu_task_queue
from integrations.feishu.client import feishu_client
from integrations.feishu.mapping_store import feishu_mapping_store
from models.schemas import (
    AgentInfo, AgentRun, Channel, FeishuBinding, FeishuBindingRequest,
    Message, RegisterAgentRequest, SendRequest,
    RunProgressEvent,
)

router = APIRouter(prefix="/api")


# --- agents ---
@router.get("/agents", response_model=list[AgentInfo])
async def list_agents():
    return registry.infos()


@router.get("/agents/roles")
async def agent_roles():
    return {
        role: agent.agent_id
        for role, agent in resolve_role_agents().items()
    }


@router.post("/agents", response_model=AgentInfo)
async def register_agent(req: RegisterAgentRequest):
    import uuid
    aid = f"{req.kind}-{uuid.uuid4().hex[:8]}"
    if req.kind == "llm":
        if not req.provider or not req.model:
            raise HTTPException(400, "provider and model required for llm agent")
        agent = LLMAgent(
            agent_id=aid, name=req.name, provider=req.provider, model=req.model,
            system_prompt=req.system_prompt or "", description=req.description,
        )
    else:  # custom
        if not req.module_path:
            raise HTTPException(400, "module_path required for custom agent")
        agent = CustomAgent(aid, req.name, req.module_path, req.description)
    registry.register(agent)
    return agent.info()


@router.get("/agents/{agent_id}/health")
async def agent_health(agent_id: str):
    a = registry.get(agent_id)
    if not a:
        raise HTTPException(404, "agent not found")
    return {"healthy": await a.health()}


# --- channels ---
@router.get("/channels", response_model=list[Channel])
async def list_channels():
    return await session_manager.list_channels()


@router.post("/channels", response_model=Channel)
async def create_channel(name: str = "new channel", agent_ids: str = ""):
    ids = [x for x in agent_ids.split(",") if x] if agent_ids else []
    return await session_manager.create_channel(name, ids)


@router.get("/channels/{channel_id}", response_model=Channel | None)
async def get_channel(channel_id: str):
    return await session_manager.get_channel(channel_id)


@router.delete("/channels/{channel_id}")
async def delete_channel(channel_id: str):
    ok = await session_manager.delete_channel(channel_id)
    if not ok:
        raise HTTPException(404, "channel not found")
    return {"ok": True}


@router.post("/channels/{channel_id}/agents", response_model=Channel | None)
async def add_agent(channel_id: str, agent_id: str):
    ch = await session_manager.add_agent(channel_id, agent_id)
    if ch is None:
        raise HTTPException(404, "channel not found")
    return ch


@router.delete("/channels/{channel_id}/agents/{agent_id}", response_model=Channel | None)
async def remove_agent(channel_id: str, agent_id: str):
    ch = await session_manager.remove_agent(channel_id, agent_id)
    if ch is None:
        raise HTTPException(404, "channel not found")
    return ch


@router.get("/channels/{channel_id}/messages", response_model=list[Message])
async def get_history(channel_id: str, limit: int = 100):
    return await session_manager.get_history(channel_id, limit)


@router.post("/channels/{channel_id}/messages")
async def send_message(channel_id: str, req: SendRequest):
    """Non-streaming send: triggers orchestrator dispatch. Subscribe over WS to
    receive events. Returns immediately once dispatch completes."""
    from core.orchestrator import orchestrator
    ch = await session_manager.get_channel(channel_id)
    if ch is None:
        raise HTTPException(404, "channel not found")
    await orchestrator.dispatch(channel_id, req.content, req.mentioned_agents, req.strategy)
    return {"ok": True}


@router.post("/channels/{channel_id}/workspace")
async def pin_workspace(channel_id: str, path: str):
    try:
        workspace_manager.pin(channel_id, path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "path": workspace_manager.get_dir(channel_id)}


# --- Feishu integration ---
@router.get("/integrations/feishu/status")
async def feishu_status():
    from config import settings
    return {
        "configured": settings.feishu_enabled,
        "connection_mode": settings.feishu_connection_mode,
        "connected": feishu_client.connected,
        "client_running": feishu_client.running,
        "connection_error": feishu_client.start_error,
        "workers_running": feishu_task_queue.running,
        "pending_tasks": feishu_task_queue.pending_count,
    }


@router.get("/integrations/feishu/bindings", response_model=list[FeishuBinding])
async def list_feishu_bindings():
    return await feishu_mapping_store.list_bindings()


@router.post("/integrations/feishu/bindings", response_model=FeishuBinding)
async def create_feishu_binding(req: FeishuBindingRequest):
    try:
        return await feishu_mapping_store.bind(
            req.tenant_key,
            req.chat_id,
            req.workspace_dir,
            req.channel_name,
            req.wiki_space_id,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/integrations/feishu/bindings/{binding_id}")
async def delete_feishu_binding(binding_id: str):
    if not await feishu_mapping_store.delete_binding(binding_id):
        raise HTTPException(404, "binding not found")
    return {"ok": True}


@router.get("/integrations/feishu/runs", response_model=list[AgentRun])
async def list_feishu_runs(limit: int = 100):
    return await run_store.list(max(1, min(limit, 500)))


@router.get("/integrations/feishu/runs/{run_id}", response_model=AgentRun)
async def get_feishu_run(run_id: str):
    run = await run_store.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return run


@router.get(
    "/integrations/feishu/runs/{run_id}/events",
    response_model=list[RunProgressEvent],
)
async def get_feishu_run_events(run_id: str, limit: int = 200):
    if await run_store.get(run_id) is None:
        raise HTTPException(404, "run not found")
    return await run_event_store.list(run_id, max(1, min(limit, 1000)))
