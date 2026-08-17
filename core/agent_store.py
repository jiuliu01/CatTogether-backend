"""Agent spec store: per-project JSON files for agent specifications.

Specs live at   data/projects/<pid>/agents/<agent_id>.json
Legacy per-project Agent memory remains read-only until the v2 migration ends.

This store backs the four lifecycle operations:
  - generate  (create)
  - specify   (update)
  - eliminate (delete, optional memory backup)
  - transplant (copy/move spec; role memory is global and is not copied)
"""
from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path

from config import settings
from core.agent_templates import default_agent_set, template
from core.identifiers import validate_storage_id
from core.persistence import atomic_write_text
from core.project_store import project_store
from models.agent_spec import AgentSpec, AgentSpecUpdateRequest


def _uuid() -> str:
    return uuid.uuid4().hex


class AgentStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    def _spec_path(self, project_id: str, agent_id: str) -> Path:
        safe_agent_id = validate_storage_id(agent_id, "agent_id")
        return project_store.agents_dir(project_id) / f"{safe_agent_id}.json"

    def _memory_path(self, project_id: str, agent_id: str) -> Path:
        safe_agent_id = validate_storage_id(agent_id, "agent_id")
        return project_store.memory_dir(project_id) / f"{safe_agent_id}.json"

    def _load_spec(self, project_id: str, agent_id: str) -> AgentSpec | None:
        path = self._spec_path(project_id, agent_id)
        if not path.exists():
            return None
        try:
            spec = AgentSpec.model_validate_json(path.read_text(encoding="utf-8"))
            if spec.memory.enabled and "memory_search" not in spec.mcp_capabilities:
                spec.mcp_capabilities.append("memory_search")
                self._enforce_safety(spec)
                self._save_spec(spec)
            return spec
        except (OSError, ValueError):
            return None

    def _save_spec(self, spec: AgentSpec) -> None:
        path = self._spec_path(spec.project_id, spec.agent_id)
        atomic_write_text(path, spec.model_dump_json(indent=2))

    # --- generate ---
    async def create(
        self,
        project_id: str,
        agent_id: str,
        *,
        name: str | None = None,
        role: str = "custom",
        from_template: str | None = None,
        **overrides,
    ) -> AgentSpec:
        async with self._lock:
            if self._load_spec(project_id, agent_id):
                raise ValueError(f"agent {agent_id} already exists in project {project_id}")
            tpl_role = from_template or role
            spec = template(tpl_role, agent_id, project_id, **overrides)
            if name:
                spec.name = name
            if role and not from_template:
                spec.role = role
            spec.origin = "generated"
            self._enforce_safety(spec)
            self._save_spec(spec)
            await project_store.add_agent(project_id, agent_id)
            return spec

    async def seed_defaults(self, project_id: str) -> list[AgentSpec]:
        """Seed a project with the default agent set (idempotent).

        Creates any of the four cats that are missing, without touching
        existing ones. Returns the specs created this call.
        """
        specs: list[AgentSpec] = []
        for spec in default_agent_set(project_id):
            if not self._load_spec(project_id, spec.agent_id):
                async with self._lock:
                    self._save_spec(spec)
                    await project_store.add_agent(project_id, spec.agent_id)
                specs.append(spec)
        # Ensure a default entry agent is set.
        project = await project_store.get(project_id)
        if project and not project.default_agent_id:
            default_id = settings.default_entry_agent or "coordinator"
            if self._load_spec(project_id, default_id):
                await project_store.update(project_id, default_agent_id=default_id)
        return specs

    # --- read ---
    async def get(self, project_id: str, agent_id: str) -> AgentSpec | None:
        return self._load_spec(project_id, agent_id)

    async def list(self, project_id: str) -> list[AgentSpec]:
        d = project_store.agents_dir(project_id)
        specs: list[AgentSpec] = []
        for path in d.glob("*.json"):
            spec = self._load_spec(project_id, path.stem)
            if spec is not None and spec.project_id == project_id:
                specs.append(spec)
        return specs

    # --- specify ---
    async def update(self, project_id: str, agent_id: str, req: AgentSpecUpdateRequest) -> AgentSpec | None:
        async with self._lock:
            spec = self._load_spec(project_id, agent_id)
            if not spec:
                return None
            data = req.model_dump(exclude_unset=True)
            for key, value in data.items():
                if value is None and key != "builtin_tools":
                    continue
                setattr(spec, key, value)
            self._enforce_safety(spec)
            self._save_spec(spec)
            return spec

    # --- eliminate ---
    async def delete(self, project_id: str, agent_id: str, *, keep_memory: bool = True) -> bool:
        async with self._lock:
            spec_path = self._spec_path(project_id, agent_id)
            if not spec_path.exists():
                return False
            spec_path.unlink()
            # V2 role memory is global and must survive elimination. Legacy
            # project memory is deliberately left untouched for migration.
            await project_store.remove_agent(project_id, agent_id)
            return True

    # --- transplant ---
    async def transplant(
        self,
        src_project_id: str,
        agent_id: str,
        dst_project_id: str,
        *,
        mode: str = "copy",
        new_agent_id: str | None = None,
    ) -> AgentSpec:
        async with self._lock:
            src_spec = self._load_spec(src_project_id, agent_id)
            if not src_spec:
                raise ValueError(f"agent {agent_id} not found in project {src_project_id}")
            if not await project_store.get(dst_project_id):
                raise ValueError(f"destination project {dst_project_id} does not exist")

            dst_agent_id = new_agent_id or agent_id
            if self._load_spec(dst_project_id, dst_agent_id):
                if dst_agent_id == agent_id and src_project_id != dst_project_id:
                    dst_agent_id = f"{agent_id}-from-{src_project_id[:8]}"
                if self._load_spec(dst_project_id, dst_agent_id):
                    dst_agent_id = f"{dst_agent_id}-{_uuid()[:6]}"

            # Build the transplanted spec.
            dst_spec = src_spec.model_copy(update={
                "agent_id": dst_agent_id,
                "project_id": dst_project_id,
                "origin": "transplanted",
            })
            self._enforce_safety(dst_spec)
            self._save_spec(dst_spec)

            await project_store.add_agent(dst_project_id, dst_agent_id)

            if mode == "move":
                src_spec_path = self._spec_path(src_project_id, agent_id)
                src_spec_path.unlink(missing_ok=True)
                await project_store.remove_agent(src_project_id, agent_id)

            return dst_spec

    @staticmethod
    def _redact_paths(entries: list[dict], src_project_id: str) -> None:
        """Replace old-project absolute paths in carried memory with a note."""
        import re
        pattern = re.compile(rf"data/projects/{re.escape(src_project_id)}/[^\s\"']+")
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("text"), str):
                entry["text"] = pattern.sub("[旧项目路径]", entry["text"])

    @staticmethod
    def _enforce_safety(spec: AgentSpec) -> None:
        """Reject specs that would let a Feishu-triggered agent escape sandbox."""
        spec.disallowed_tools = list(dict.fromkeys(spec.disallowed_tools + ["Agent", "Bash"]))
        if spec.builtin_tools is not None:
            known = {"Read", "Glob", "Grep", "Edit", "Write", "WebSearch", "WebFetch"}
            spec.builtin_tools = [
                tool for tool in dict.fromkeys(spec.builtin_tools)
                if tool in known and tool.lower() not in {"agent", "bash"}
            ]
        known_caps = {
            "delegate", "report_progress", "wiki_read", "wiki_list",
            "wiki_write", "wiki_append", "memory_search",
        }
        spec.mcp_capabilities = [
            capability for capability in dict.fromkeys(spec.mcp_capabilities)
            if capability in known_caps
        ]
        if spec.sandbox not in ("read", "workspace-write"):
            raise ValueError("sandbox must be read or workspace-write")
        if spec.sandbox == "read" and spec.builtin_tools is not None:
            spec.builtin_tools = [
                tool for tool in spec.builtin_tools if tool.lower() not in {"edit", "write"}
            ]


agent_store = AgentStore()
