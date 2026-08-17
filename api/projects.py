"""Project & agent lifecycle REST API: 生成 / 制定 / 消除 / 移植 + project CRUD."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from core.agent_store import agent_store
from core.project_store import project_store
from models.agent_spec import AgentSpec, AgentSpecUpdateRequest
from models.project import Project, ProjectCreateRequest

router = APIRouter(prefix="/api")


# --- projects ---
@router.get("/projects", response_model=list[Project])
async def list_projects():
    return await project_store.list()


@router.post("/projects", response_model=Project)
async def create_project(req: ProjectCreateRequest):
    return await project_store.create(
        req.name,
        req.workspace_dir,
        feishu_chat_id=req.feishu_chat_id,
        tenant_key=req.tenant_key,
        default_agent_id=req.default_agent_id,
    )


@router.get("/projects/{project_id}", response_model=Project | None)
async def get_project(project_id: str):
    return await project_store.get(project_id)


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    ok = await project_store.delete(project_id)
    if not ok:
        raise HTTPException(404, "project not found")
    return {"ok": True}


# --- agents in a project ---
@router.get("/projects/{project_id}/agents", response_model=list[AgentSpec])
async def list_agents(project_id: str):
    if await project_store.get(project_id) is None:
        raise HTTPException(404, "project not found")
    return await agent_store.list(project_id)


@router.post("/projects/{project_id}/agents", response_model=AgentSpec)
async def generate_agent(
    project_id: str,
    agent_id: str,
    name: str | None = None,
    role: str = "custom",
    from_template: str | None = None,
):
    """生成: create a new agent in the project from a template."""
    if await project_store.get(project_id) is None:
        raise HTTPException(404, "project not found")
    try:
        return await agent_store.create(
            project_id, agent_id, name=name, role=role, from_template=from_template,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.patch("/projects/{project_id}/agents/{agent_id}", response_model=AgentSpec | None)
async def specify_agent(project_id: str, agent_id: str, req: AgentSpecUpdateRequest):
    """制定: update an agent's spec."""
    try:
        updated = await agent_store.update(project_id, agent_id, req)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if updated is None:
        raise HTTPException(404, "agent not found")
    return updated


@router.delete("/projects/{project_id}/agents/{agent_id}")
async def eliminate_agent(project_id: str, agent_id: str, keep_memory: bool = True):
    """消除: remove an agent from the project, optionally backing up memory."""
    ok = await agent_store.delete(project_id, agent_id, keep_memory=keep_memory)
    if not ok:
        raise HTTPException(404, "agent not found")
    return {"ok": True}


# --- transplant ---
@router.post("/projects/{dst_project_id}/agents/transplant", response_model=AgentSpec)
async def transplant_agent(
    dst_project_id: str,
    src_project_id: str,
    agent_id: str,
    mode: str = "copy",
    new_agent_id: str | None = None,
):
    """移植: copy/move an agent (with memory) from one project to another."""
    try:
        return await agent_store.transplant(
            src_project_id, agent_id, dst_project_id, mode=mode, new_agent_id=new_agent_id,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
