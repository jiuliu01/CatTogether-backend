"""Pydantic schemas shared across the backend (and mirrored by frontend types).

These are the single source of truth for the wire format. The WebSocket event
envelope (AgentEvent) is defined here and referenced by api/ws.py, the event
bus, the orchestrator, and the frontend useWsEvents hook.
"""
from __future__ import annotations

from typing import Literal, Any
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field


def now() -> datetime:
    return datetime.now(timezone.utc)


AgentKind = Literal["cli", "llm", "custom"]
AgentStatus = Literal["idle", "thinking", "running", "error"]
EventType = Literal["text_delta", "tool_call", "tool_result", "status", "done", "error"]
ProgressKind = Literal[
    "stage", "action", "checkpoint", "delegate", "delegate_requested",
    "delegate_started", "delegate_completed", "delegate_failed", "artifact",
    "artifact_created", "artifact_failed", "heartbeat", "stalled", "error"
]
Strategy = Literal["direct", "round_robin", "fan_out"]
MessageRole = Literal["user", "agent", "system"]
RunStatus = Literal[
    "queued", "classifying", "planning", "researching", "coding", "reviewing",
    "revising", "waiting_children", "recovering", "completing",
    "completed", "failed", "cancelled",
]
MemoryDomain = Literal["user", "workspace", "agent", "project", "task"]
"""v1/v2 domains (user/workspace/agent) + v3 additions (project/task).

``project`` is the v3 rename of ``workspace``; ``task`` is a new task-level
scope.  ``workspace`` is kept for backward-compatible parsing of v1/v2 JSON.
"""
MemoryKind = Literal[
    "preference", "correction", "project_fact", "decision", "progress",
    "convention", "practice", "working_note",
]
"""``practice`` kept for v1/v2 JSON compat; v3 uses ``working_note`` (moved to
Skill layer per 2.1 R6, but FactKind still accepts it as an alias)."""
MemoryStatus = Literal["active", "superseded", "archived", "disputed"]
"""v1 had active/superseded; v3 adds archived (soft-delete) and disputed
(conflict flagged, awaiting resolution)."""
MemorySourceType = Literal[
    "user_statement", "agent_result", "workspace_change", "manual", "migration",
    "file_changed", "agent_tool_call",
]


class AgentInfo(BaseModel):
    id: str
    name: str
    kind: AgentKind
    status: AgentStatus = "idle"
    description: str = ""


class Channel(BaseModel):
    id: str
    name: str
    agent_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now)


class Message(BaseModel):
    id: str
    channel_id: str
    role: MessageRole
    agent_id: str | None = None
    content: str = ""
    created_at: datetime = Field(default_factory=now)
    seq: int = 0
    thread_id: str | None = None
    external_message_id: str | None = None


class AgentEvent(BaseModel):
    """Wire envelope for one streamed event over the WebSocket."""
    seq: int
    channel_id: str
    agent_id: str
    agent_name: str = ""
    type: EventType
    data: dict[str, Any] = Field(default_factory=dict)


class RunProgressEvent(BaseModel):
    """Persisted, user-visible work log for one Feishu run."""
    run_id: str
    seq: int = 0
    kind: ProgressKind
    agent_id: str = ""
    agent_name: str = ""
    phase: str = ""
    title: str
    detail: str = ""
    child_run_id: str | None = None
    target: str | None = None
    status: str | None = None
    artifact_id: str | None = None
    type: str | None = None
    url: str | None = None
    created_at: datetime = Field(default_factory=now)


class ArtifactRef(BaseModel):
    artifact_id: str
    type: str
    title: str
    url: str = ""
    status: Literal["created", "failed", "updated"] = "created"


class RunResult(BaseModel):
    final_message: str
    reply_agent_id: str
    reply_agent_name: str
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    process_event_count: int = 0


class MemoryEntry(BaseModel):
    id: str
    domain: MemoryDomain = "agent"
    scope_id: str = "legacy"
    kind: MemoryKind = "practice"
    text: str
    tags: list[str] = Field(default_factory=list)
    importance: float = 0.5
    confidence: float = 0.5
    status: MemoryStatus = "active"
    source_type: MemorySourceType = "agent_result"
    source_ids: list[str] = Field(default_factory=list)
    supersedes_id: str | None = None
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)
    expires_at: datetime | None = None
    version: int = 1
    # ----- v3 optional fields (2.1 four-layer / four-domain) -----
    # All default to None/empty so existing v1/v2 JSON loads without changes.
    # Populated by the v3 nine-stage pipeline; ignored by the v1 JSON backend.
    tenant_id: str | None = None
    """Multi-tenant isolation key (R5). None means v1/v2 data (treat as default)."""
    entities: list[dict[str, Any]] = Field(default_factory=list)
    """Embedded EntityRef list (R2 canonical entity, JSONB in SQLite)."""
    temporal: dict[str, Any] | None = None
    """TemporalInfo: valid_from/valid_to/event_time/time_expressions/is_snapshot."""
    provenance: dict[str, Any] | None = None
    """ProvenanceInfo: author/extractor/extraction_confidence/evidence_refs."""
    schema_version: int = 2
    """2 = v1/v2 JSON backend; 3 = v3 SQLite four-layer."""


class MemoryCandidate(BaseModel):
    domain: MemoryDomain
    kind: MemoryKind
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    source_type: MemorySourceType
    source_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class MemoryWriteEvent(BaseModel):
    event_id: str
    event_type: Literal[
        "conversation_completed", "agent_invocation",
        "agent_tool_call", "file_changed", "manual_write",
    ]
    user_id: str = "default"
    workspace_id: str | None = None
    channel_id: str
    thread_id: str | None = None
    run_id: str | None = None
    agent_id: str
    agent_role: str = "custom"
    user_text: str = ""
    final_text: str = ""
    mutation_paths: list[str] = Field(default_factory=list)
    allowed_domains: list[MemoryDomain] = Field(default_factory=list)
    source_message_ids: list[str] = Field(default_factory=list)
    succeeded: bool = True
    created_at: datetime = Field(default_factory=now)
    # ----- v3 optional fields -----
    tenant_id: str | None = None
    """Tenant isolation key (R5). None means v1/v2 (treat as default)."""
    task_id: str | None = None
    """Active task scope_id for task-domain routing. None when no active task."""


class ConversationSummary(BaseModel):
    text: str = ""
    through_seq: int = 0
    source_message_count: int = 0
    updated_at: datetime = Field(default_factory=now)
    version: int = 1


class MemoryUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    tags: list[str] | None = None
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    status: MemoryStatus | None = None


class SendRequest(BaseModel):
    content: str
    mentioned_agents: list[str] | None = None
    strategy: Strategy | None = None


class RegisterAgentRequest(BaseModel):
    """Register a custom/llm agent at runtime."""
    name: str
    kind: Literal["llm", "custom"] = "llm"
    description: str = ""
    # for llm
    provider: Literal["openai", "anthropic"] | None = None
    model: str | None = None
    system_prompt: str | None = None
    # for custom
    module_path: str | None = None  # e.g. "mypkg.myagent:run"


class Thread(BaseModel):
    id: str
    channel_id: str
    external_root_id: str
    source: Literal["web", "feishu"] = "web"
    created_at: datetime = Field(default_factory=now)


class FeishuBinding(BaseModel):
    id: str
    tenant_key: str
    chat_id: str
    channel_id: str
    workspace_dir: str
    project_id: str | None = None
    enabled: bool = True
    created_at: datetime = Field(default_factory=now)


class FeishuBindingRequest(BaseModel):
    tenant_key: str
    chat_id: str
    workspace_dir: str
    channel_name: str | None = None
    wiki_space_id: str | None = None


class FeishuInboundMessage(BaseModel):
    message_id: str
    tenant_key: str
    chat_id: str
    chat_type: str = "group"
    sender_open_id: str
    sender_type: str = "user"
    root_id: str | None = None
    parent_id: str | None = None
    content: str
    mentioned_bot: bool = False

    @property
    def external_root_id(self) -> str:
        return self.root_id or self.parent_id or self.message_id

    @property
    def external_user_id(self) -> str:
        return f"feishu:{self.tenant_key}:{self.sender_open_id}"


class DelegationRecord(BaseModel):
    """Persisted background work order created by an agent delegation."""

    id: str
    parent_invocation_id: str
    target_agent_id: str
    task: str
    access: Literal["read", "write"] = "read"
    depth: int = 1
    status: Literal[
        "queued", "running", "waiting_children", "recovering",
        "completed", "failed", "cancelled",
    ] = "queued"
    result: str = ""
    error: str = ""
    consumed: bool = False
    recovery_count: int = 0
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


class AgentRun(BaseModel):
    id: str
    channel_id: str
    thread_id: str
    trigger_message_id: str
    trigger_user_id: str | None = None
    status: RunStatus = "queued"
    plan: dict[str, Any] | None = None
    outputs: dict[str, Any] = Field(default_factory=dict)
    intent: dict[str, Any] | None = None
    error: str | None = None
    original_task: str = ""
    project_id: str | None = None
    workspace_dir: str = ""
    entry_agent_id: str = ""
    root_invocation_id: str = ""
    tenant_key: str = ""
    chat_id: str = ""
    sender_open_id: str = ""
    delegations: list[DelegationRecord] = Field(default_factory=list)
    pending_child_ids: list[str] = Field(default_factory=list)
    resume_queued_for: list[str] = Field(default_factory=list)
    last_progress_at: datetime | None = None
    last_activity_at: datetime | None = None
    last_tool: str = ""
    last_tool_id: str = ""
    last_changed_files: list[str] = Field(default_factory=list)
    recovery_count: int = 0
    stage_summary: str = ""
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)
