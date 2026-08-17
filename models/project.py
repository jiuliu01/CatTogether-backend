"""Project model: one Feishu group <-> one project.

A project is the host and boundary for a set of agents, their memory, and a
workspace directory. Agents always belong to a project.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field


def now() -> datetime:
    return datetime.now(timezone.utc)


class Project(BaseModel):
    id: str
    name: str
    feishu_chat_id: str | None = None
    tenant_key: str | None = None
    workspace_dir: str
    default_agent_id: str | None = None
    agent_ids: list[str] = Field(default_factory=list)
    wiki_space_id: str | None = None
    routing_strategy: Literal["explicit", "keyword", "default"] = "explicit"
    created_at: datetime = Field(default_factory=now)


class ProjectCreateRequest(BaseModel):
    name: str
    workspace_dir: str
    feishu_chat_id: str | None = None
    tenant_key: str | None = None
    default_agent_id: str | None = None
