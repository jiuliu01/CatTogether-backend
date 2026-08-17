"""JSON-backed Feishu group bindings and inbound-message idempotency."""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from core.project_store import project_store
from core.session_manager import session_manager
from core.workspace import workspace_manager
from models.schemas import FeishuBinding, FeishuInboundMessage


def _atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


class FeishuMappingStore:
    def __init__(self) -> None:
        self._dir = settings.data_dir / "feishu"
        self._bindings_path = self._dir / "bindings.json"
        self._processed_path = self._dir / "processed.json"
        self._bindings: dict[str, FeishuBinding] = {}
        self._processed: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._group_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._load()

    def _load(self) -> None:
        try:
            items = json.loads(self._bindings_path.read_text(encoding="utf-8"))
            self._bindings = {
                item["id"]: FeishuBinding.model_validate(item)
                for item in items
                if isinstance(item, dict) and item.get("id")
            }
            for binding in self._bindings.values():
                try:
                    workspace_manager.pin(binding.channel_id, binding.workspace_dir)
                except ValueError:
                    # Keep the record visible to administrators, but never
                    # activate a path outside the configured allow-list.
                    continue
        except (OSError, ValueError, TypeError, KeyError):
            self._bindings = {}
        try:
            value = json.loads(self._processed_path.read_text(encoding="utf-8"))
            self._processed = value if isinstance(value, dict) else {}
        except (OSError, ValueError, TypeError):
            self._processed = {}

    def _save_bindings(self) -> None:
        _atomic_write(
            self._bindings_path,
            [binding.model_dump(mode="json") for binding in self._bindings.values()],
        )

    def _save_processed(self) -> None:
        # Bound the no-database idempotency file. Dict insertion order keeps the
        # newest entries at the end on supported Python versions.
        if len(self._processed) > 5000:
            self._processed = dict(list(self._processed.items())[-5000:])
        _atomic_write(self._processed_path, self._processed)

    async def list_bindings(self) -> list[FeishuBinding]:
        return list(self._bindings.values())

    async def get_binding(self, tenant_key: str, chat_id: str) -> FeishuBinding | None:
        return next(
            (
                binding
                for binding in self._bindings.values()
                if binding.tenant_key == tenant_key
                and binding.chat_id == chat_id
                and binding.enabled
            ),
            None,
        )

    async def bind(
        self,
        tenant_key: str,
        chat_id: str,
        workspace_dir: str,
        channel_name: str | None = None,
        wiki_space_id: str | None = None,
    ) -> FeishuBinding:
        group_lock = self._group_locks.setdefault((tenant_key, chat_id), asyncio.Lock())
        async with group_lock:
            workspace = workspace_manager.validate_path(workspace_dir)
            existing = await self.get_binding(tenant_key, chat_id)
            if existing:
                existing.workspace_dir = str(workspace)
                workspace_manager.pin(existing.channel_id, str(workspace))
                if existing.project_id:
                    await project_store.update(
                        existing.project_id, workspace_dir=str(workspace)
                    )
                if wiki_space_id and existing.project_id:
                    await project_store.set_wiki_space(
                        existing.project_id, wiki_space_id
                    )
                async with self._lock:
                    self._bindings[existing.id] = existing
                    self._save_bindings()
                return existing

            channel = await session_manager.create_channel(
                channel_name or f"Feishu {chat_id[-8:]}"
            )
            project = await project_store.get_or_create_for_chat(
                tenant_key, chat_id, str(workspace),
                name=channel_name or f"Feishu {chat_id[-8:]}",
            )
            if wiki_space_id:
                await project_store.set_wiki_space(project.id, wiki_space_id)
            binding = FeishuBinding(
                id=uuid.uuid4().hex,
                tenant_key=tenant_key,
                chat_id=chat_id,
                channel_id=channel.id,
                workspace_dir=str(workspace),
                project_id=project.id,
            )
            workspace_manager.pin(channel.id, str(workspace))
            async with self._lock:
                self._bindings[binding.id] = binding
                self._save_bindings()
            return binding

    async def ensure_binding(self, message: FeishuInboundMessage) -> FeishuBinding:
        group_key = (message.tenant_key, message.chat_id)
        group_lock = self._group_locks.setdefault(group_key, asyncio.Lock())
        async with group_lock:
            binding = await self.get_binding(message.tenant_key, message.chat_id)
            if binding:
                channel = await session_manager.get_channel(binding.channel_id)
                if channel is not None:
                    workspace_manager.pin(binding.channel_id, binding.workspace_dir)
                    # Backfill project_id on legacy bindings created before the
                    # project layer existed.
                    if not binding.project_id:
                        project = await project_store.get_or_create_for_chat(
                            message.tenant_key, message.chat_id, binding.workspace_dir,
                            name=f"Feishu {message.chat_id[-8:]}",
                        )
                        binding.project_id = project.id
                        async with self._lock:
                            self._save_bindings()
                    elif binding.project_id:
                        project = await project_store.get(binding.project_id)
                        if project and project.workspace_dir != binding.workspace_dir:
                            await project_store.update(
                                binding.project_id,
                                workspace_dir=binding.workspace_dir,
                            )
                    return binding

            channel = await session_manager.create_channel(f"Feishu {message.chat_id[-8:]}")
            workspace = workspace_manager.get_dir(channel.id)
            project = await project_store.get_or_create_for_chat(
                message.tenant_key, message.chat_id, workspace,
                name=f"Feishu {message.chat_id[-8:]}",
            )
            if binding:
                binding.channel_id = channel.id
                binding.workspace_dir = workspace
                binding.project_id = project.id
            else:
                binding = FeishuBinding(
                    id=uuid.uuid4().hex,
                    tenant_key=message.tenant_key,
                    chat_id=message.chat_id,
                    channel_id=channel.id,
                    workspace_dir=workspace,
                    project_id=project.id,
                )
            async with self._lock:
                self._bindings[binding.id] = binding
                self._save_bindings()
            return binding

    async def delete_binding(self, binding_id: str) -> bool:
        async with self._lock:
            removed = self._bindings.pop(binding_id, None)
            if removed:
                self._save_bindings()
            return removed is not None

    async def mark_processed(self, message_id: str) -> bool:
        """Return True only for the first observation of a message."""
        async with self._lock:
            if message_id in self._processed:
                return False
            self._processed[message_id] = datetime.now(timezone.utc).isoformat()
            self._save_processed()
            return True


feishu_mapping_store = FeishuMappingStore()
