"""Project store: JSON-backed CRUD for projects under data/projects/<pid>/.

A project is the host for agents, their memory, and a workspace. One Feishu
group maps to one project. No database.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from config import settings
from core.identifiers import validate_storage_id
from core.persistence import atomic_write_text
from models.project import Project


def _uuid() -> str:
    return uuid.uuid4().hex


class ProjectStore:
    def __init__(self) -> None:
        self._root = settings.data_dir / "projects"
        self._projects: dict[str, Project] = {}
        self._lock = asyncio.Lock()
        self._load()

    def _root_dir(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root

    def _project_dir(self, project_id: str) -> Path:
        safe = validate_storage_id(project_id, "project_id")
        return self._root_dir() / safe

    def _load(self) -> None:
        root = self._root_dir()
        for path in root.glob("*/project.json"):
            try:
                project = Project.model_validate_json(path.read_text(encoding="utf-8"))
                self._projects[project.id] = project
            except (OSError, ValueError):
                continue

    def _write(self, project: Project) -> None:
        d = self._project_dir(project.id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "workspace").mkdir(parents=True, exist_ok=True)
        (d / "agents" / "memory").mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            d / "project.json",
            project.model_dump_json(indent=2),
        )

    async def create(
        self,
        name: str,
        workspace_dir: str,
        *,
        feishu_chat_id: str | None = None,
        tenant_key: str | None = None,
        default_agent_id: str | None = None,
        project_id: str | None = None,
    ) -> Project:
        pid = project_id or _uuid()
        project = Project(
            id=pid,
            name=name,
            feishu_chat_id=feishu_chat_id,
            tenant_key=tenant_key,
            workspace_dir=workspace_dir,
            default_agent_id=default_agent_id,
        )
        async with self._lock:
            self._projects[pid] = project
            self._write(project)
        return project

    async def get(self, project_id: str) -> Project | None:
        return self._projects.get(project_id)

    async def list(self) -> list[Project]:
        return list(self._projects.values())

    async def get_for_chat(self, tenant_key: str, chat_id: str) -> Project | None:
        for project in self._projects.values():
            if project.tenant_key == tenant_key and project.feishu_chat_id == chat_id:
                return project
        return None

    async def get_or_create_for_chat(
        self,
        tenant_key: str,
        chat_id: str,
        workspace_dir: str,
        name: str | None = None,
    ) -> Project:
        existing = await self.get_for_chat(tenant_key, chat_id)
        if existing:
            if existing.workspace_dir != workspace_dir:
                await self.update(existing.id, workspace_dir=workspace_dir)
            return existing
        return await self.create(
            name=name or f"Feishu {chat_id[-8:]}",
            workspace_dir=workspace_dir,
            feishu_chat_id=chat_id,
            tenant_key=tenant_key,
        )

    async def update(self, project_id: str, **fields) -> Project | None:
        async with self._lock:
            project = self._projects.get(project_id)
            if not project:
                return None
            for key, value in fields.items():
                if value is not None and hasattr(project, key):
                    setattr(project, key, value)
            self._write(project)
            return project

    async def set_wiki_space(self, project_id: str, space_id: str) -> None:
        async with self._lock:
            project = self._projects.get(project_id)
            if project:
                project.wiki_space_id = space_id
                self._write(project)

    async def add_agent(self, project_id: str, agent_id: str) -> Project | None:
        async with self._lock:
            project = self._projects.get(project_id)
            if not project:
                return None
            if agent_id not in project.agent_ids:
                project.agent_ids.append(agent_id)
            self._write(project)
            return project

    async def remove_agent(self, project_id: str, agent_id: str) -> Project | None:
        async with self._lock:
            project = self._projects.get(project_id)
            if not project:
                return None
            if agent_id in project.agent_ids:
                project.agent_ids.remove(agent_id)
            if project.default_agent_id == agent_id:
                project.default_agent_id = None
            self._write(project)
            return project

    async def delete(self, project_id: str) -> bool:
        async with self._lock:
            project = self._projects.pop(project_id, None)
            if not project:
                return False
            import shutil
            d = self._project_dir(project_id)
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            return True

    def workspace_dir(self, project_id: str) -> str:
        project = self._projects.get(project_id)
        if project:
            return project.workspace_dir
        d = self._project_dir(project_id) / "workspace"
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    def memory_dir(self, project_id: str) -> Path:
        """Legacy per-project Agent memory directory, retained for migration."""
        d = self._project_dir(project_id) / "agents" / "memory"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def workspace_memory_path(self, project_id: str) -> Path:
        """V2 workspace-domain memory file."""
        return self._project_dir(project_id) / "memory.json"

    def agents_dir(self, project_id: str) -> Path:
        d = self._project_dir(project_id) / "agents"
        d.mkdir(parents=True, exist_ok=True)
        return d


project_store = ProjectStore()
