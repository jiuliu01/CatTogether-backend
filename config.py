"""CatTogether backend configuration.

No database. All persistence is JSON files under data_dir. Runtime state is
in-memory. Secrets (API keys) come from environment variables or a local
.env-style file the user may create; we never read interactive login tokens.
"""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field

# Load a local .env file (next to this module) so settings persist across
# terminal restarts without re-typing $env:. Existing process env wins.
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]


@dataclass
class Settings:
    host: str = field(default_factory=lambda: os.environ.get("CT_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("CT_PORT", "8000")))

    # Project root is two levels up from this file (backend/config.py -> CatTogether)
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data")

    # LLM provider keys (API-key auth, never interactive login)
    openai_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    openai_base_url: str = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    anthropic_api_key: str | None = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY"))
    anthropic_base_url: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"))

    # CLI agent settings: explicit binary paths (optional). If None, rely on PATH.
    codex_bin: str | None = field(default_factory=lambda: os.environ.get("CODEX_BIN") or None)
    claude_bin: str | None = field(default_factory=lambda: os.environ.get("CLAUDE_BIN") or None)

    # Memory 2.1 — single converged system, no version switches.
    short_term_window: int = field(default_factory=lambda: int(os.environ.get("CT_SHORT_TERM_WINDOW", "20")))
    long_term_top_k: int = field(default_factory=lambda: int(os.environ.get("CT_LONG_TERM_TOP_K", "5")))
    memory_max_entries_per_scope: int = field(
        default_factory=lambda: int(os.environ.get("CT_MEMORY_MAX_ENTRIES_PER_SCOPE", "1000"))
    )
    memory_user_top_k: int = field(default_factory=lambda: int(os.environ.get("CT_MEMORY_USER_TOP_K", "4")))
    memory_agent_top_k: int = field(default_factory=lambda: int(os.environ.get("CT_MEMORY_AGENT_TOP_K", "4")))
    memory_user_token_budget: int = field(
        default_factory=lambda: int(os.environ.get("CT_MEMORY_USER_TOKEN_BUDGET", "500"))
    )
    memory_agent_token_budget: int = field(
        default_factory=lambda: int(os.environ.get("CT_MEMORY_AGENT_TOKEN_BUDGET", "500"))
    )
    memory_summary_token_budget: int = field(
        default_factory=lambda: int(os.environ.get("CT_MEMORY_SUMMARY_TOKEN_BUDGET", "800"))
    )
    memory_recent_token_budget: int = field(
        default_factory=lambda: int(os.environ.get("CT_MEMORY_RECENT_TOKEN_BUDGET", "2000"))
    )
    memory_total_token_budget: int = field(
        default_factory=lambda: int(os.environ.get("CT_MEMORY_TOTAL_TOKEN_BUDGET", "5000"))
    )
    memory_recent_min_messages: int = field(
        default_factory=lambda: int(os.environ.get("CT_MEMORY_RECENT_MIN_MESSAGES", "6"))
    )
    memory_user_preference_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CT_MEMORY_USER_PREFERENCE_THRESHOLD", "0.8"))
    )
    memory_user_correction_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CT_MEMORY_USER_CORRECTION_THRESHOLD", "0.75"))
    )
    memory_agent_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CT_MEMORY_AGENT_THRESHOLD", "0.85"))
    )
    memory_agent_auto_write: bool = field(
        default_factory=lambda: _env_bool("CT_MEMORY_AGENT_AUTO_WRITE", False)
    )
    memory_write_queue_size: int = field(
        default_factory=lambda: int(os.environ.get("CT_MEMORY_WRITE_QUEUE_SIZE", "200"))
    )
    memory_analyzer_provider: str = field(
        default_factory=lambda: os.environ.get("CT_MEMORY_ANALYZER_PROVIDER", "rules").lower()
    )
    memory_analyzer_model: str = field(
        default_factory=lambda: os.environ.get("CT_MEMORY_ANALYZER_MODEL", "")
    )
    memory_analyzer_timeout: float = field(
        default_factory=lambda: float(os.environ.get("CT_MEMORY_ANALYZER_TIMEOUT", "20"))
    )
    memory_summary_trigger_messages: int = field(
        default_factory=lambda: int(os.environ.get("CT_MEMORY_SUMMARY_TRIGGER_MESSAGES", "200"))
    )
    memory_summary_increment: int = field(
        default_factory=lambda: int(os.environ.get("CT_MEMORY_SUMMARY_INCREMENT", "50"))
    )
    memory_summary_recent_keep: int = field(
        default_factory=lambda: int(os.environ.get("CT_MEMORY_SUMMARY_RECENT_KEEP", "50"))
    )

    # ===== Memory 2.1 four-layer / four-domain / tenant =====
    memory_hot_enabled: bool = field(default_factory=lambda: _env_bool("CT_MEMORY_HOT_ENABLED", True))
    memory_history_enabled: bool = field(
        default_factory=lambda: _env_bool("CT_MEMORY_HISTORY_ENABLED", True)
    )
    memory_skill_enabled: bool = field(default_factory=lambda: _env_bool("CT_MEMORY_SKILL_ENABLED", True))

    # SQLite
    memory_sqlite_path: Path = field(
        default_factory=lambda: Path(os.environ.get(
            "CT_MEMORY_SQLITE_PATH", str(Path(__file__).resolve().parent / "data" / "memory.db"),
        ))
    )
    memory_sqlite_wal: bool = field(default_factory=lambda: _env_bool("CT_MEMORY_SQLITE_WAL", True))

    # Vector / embedding
    memory_vector_enabled: bool = field(
        default_factory=lambda: _env_bool("CT_MEMORY_VECTOR_ENABLED", False)
    )
    memory_embedding_model: str = field(
        default_factory=lambda: os.environ.get("CT_MEMORY_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    )
    memory_embedding_dim: int = field(
        default_factory=lambda: int(os.environ.get("CT_MEMORY_EMBEDDING_DIM", "512"))
    )
    memory_embedding_provider: str = field(
        default_factory=lambda: os.environ.get("CT_MEMORY_EMBEDDING_PROVIDER", "local").lower()
    )

    # Graph
    memory_graph_enabled: bool = field(default_factory=lambda: _env_bool("CT_MEMORY_GRAPH_ENABLED", False))
    memory_graph_max_hops: int = field(default_factory=lambda: int(os.environ.get("CT_MEMORY_GRAPH_MAX_HOPS", "2")))
    memory_outbox_poll_interval: float = field(
        default_factory=lambda: float(os.environ.get("CT_MEMORY_OUTBOX_POLL_INTERVAL", "5"))
    )

    # Four-domain budgets (project/task are new; user/agent reuse v1 fields)
    memory_project_top_k: int = field(default_factory=lambda: int(os.environ.get("CT_MEMORY_PROJECT_TOP_K", "8")))
    memory_task_top_k: int = field(default_factory=lambda: int(os.environ.get("CT_MEMORY_TASK_TOP_K", "4")))
    memory_project_token_budget: int = field(
        default_factory=lambda: int(os.environ.get("CT_MEMORY_PROJECT_TOKEN_BUDGET", "1200"))
    )
    memory_task_token_budget: int = field(
        default_factory=lambda: int(os.environ.get("CT_MEMORY_TASK_TOKEN_BUDGET", "400"))
    )
    memory_hot_token_budget: int = field(
        default_factory=lambda: int(os.environ.get("CT_MEMORY_HOT_TOKEN_BUDGET", "800"))
    )

    # Hot Memory
    memory_hot_max_per_scope: int = field(
        default_factory=lambda: int(os.environ.get("CT_MEMORY_HOT_MAX_PER_SCOPE", "50"))
    )
    memory_hot_auto_approve_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CT_MEMORY_HOT_AUTO_APPROVE_THRESHOLD", "0.9"))
    )

    # Three-signal retrieval — Weighted RRF (not weighted sum)
    memory_rrf_k: int = field(default_factory=lambda: int(os.environ.get("CT_MEMORY_RRF_K", "60")))
    memory_rrf_weight_semantic: float = field(
        default_factory=lambda: float(os.environ.get("CT_MEMORY_RRF_WEIGHT_SEMANTIC", "1.0"))
    )
    memory_rrf_weight_bm25: float = field(
        default_factory=lambda: float(os.environ.get("CT_MEMORY_RRF_WEIGHT_BM25", "1.0"))
    )
    memory_rrf_weight_entity: float = field(
        default_factory=lambda: float(os.environ.get("CT_MEMORY_RRF_WEIGHT_ENTITY", "0.8"))
    )

    # ADD-only / dedup
    memory_dedup_cosine_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CT_MEMORY_DEDUP_COSINE_THRESHOLD", "0.92"))
    )
    memory_dedup_jaccard_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CT_MEMORY_DEDUP_JACCARD_THRESHOLD", "0.85"))
    )

    # Task domain
    memory_task_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CT_MEMORY_TASK_THRESHOLD", "0.6"))
    )
    memory_task_archive_on_complete: bool = field(
        default_factory=lambda: _env_bool("CT_MEMORY_TASK_ARCHIVE_ON_COMPLETE", True)
    )

    # Tenant
    memory_default_tenant: str = field(
        default_factory=lambda: os.environ.get("CT_MEMORY_DEFAULT_TENANT", "default")
    )

    # Subprocess
    cli_timeout: float = field(default_factory=lambda: float(os.environ.get("CT_CLI_TIMEOUT", "600")))
    # asyncio defaults to 64 KiB per line. Codex JSONL may put a complete tool
    # result on one line, so use a larger configurable reader buffer.
    cli_stream_limit: int = field(
        default_factory=lambda: int(os.environ.get("CT_CLI_STREAM_LIMIT", str(16 * 1024 * 1024)))
    )
    coordinator_max_revisions: int = field(
        default_factory=lambda: int(os.environ.get("CT_COORDINATOR_MAX_REVISIONS", "2"))
    )
    task_worker_count: int = field(default_factory=lambda: int(os.environ.get("CT_TASK_WORKERS", "2")))
    task_queue_size: int = field(default_factory=lambda: int(os.environ.get("CT_TASK_QUEUE_SIZE", "100")))

    # Task classification: a short-lived Claude Code subprocess categorizes each
    # Feishu task (read-only vs needs-coding, preferred agent) before the
    # coordinator runs. Heuristics always run first; the subprocess refines.
    intent_classifier_enabled: bool = field(default_factory=lambda: _env_bool("CT_INTENT_CLASSIFIER_ENABLED", True))
    intent_classifier_timeout: float = field(
        default_factory=lambda: float(os.environ.get("CT_INTENT_CLASSIFIER_TIMEOUT", "90"))
    )
    delegate_max_depth: int = field(default_factory=lambda: int(os.environ.get("CT_DELEGATE_MAX_DEPTH", "3")))
    delegate_max_concurrency: int = field(default_factory=lambda: int(os.environ.get("CT_DELEGATE_MAX_CONCURRENCY", "4")))
    default_entry_agent: str = field(default_factory=lambda: os.environ.get("CT_DEFAULT_ENTRY_AGENT", "coordinator"))
    delegate_endpoint: str = field(default_factory=lambda: os.environ.get("CT_DELEGATE_ENDPOINT", f"http://{os.environ.get('CT_HOST','127.0.0.1')}:{os.environ.get('CT_PORT','8000')}"))
    delegate_timeout: float = field(default_factory=lambda: float(os.environ.get("CT_DELEGATE_TIMEOUT", "300")))
    runtime_probe_interval: float = field(
        default_factory=lambda: float(os.environ.get("CT_RUNTIME_PROBE_INTERVAL", "30"))
    )
    stall_check_after: float = field(
        default_factory=lambda: float(os.environ.get("CT_STALL_CHECK_AFTER", "300"))
    )
    stall_confirmations: int = field(
        default_factory=lambda: int(os.environ.get("CT_STALL_CONFIRMATIONS", "3"))
    )
    recovery_max_attempts: int = field(
        default_factory=lambda: int(os.environ.get("CT_RECOVERY_MAX_ATTEMPTS", "3"))
    )
    mcp_config_dir: str = field(default_factory=lambda: str(Path(__file__).resolve().parent / "data" / "mcp_configs"))
    chat_chunk_limit: int = field(default_factory=lambda: int(os.environ.get("CT_CHAT_CHUNK_LIMIT", "3500")))
    artifact_final_message_limit: int = field(
        default_factory=lambda: int(os.environ.get("CT_ARTIFACT_FINAL_MESSAGE_LIMIT", "1000"))
    )
    checkpoint_min_interval: float = field(
        default_factory=lambda: float(os.environ.get("CT_CHECKPOINT_MIN_INTERVAL", "30"))
    )
    strict_role_tools: bool = field(default_factory=lambda: _env_bool("CT_STRICT_ROLE_TOOLS", True))
    structured_run_result: bool = field(default_factory=lambda: _env_bool("CT_STRUCTURED_RUN_RESULT", True))
    output_intent_routing: bool = field(default_factory=lambda: _env_bool("CT_OUTPUT_INTENT_ROUTING", True))
    checkpoint_delivery: bool = field(default_factory=lambda: _env_bool("CT_CHECKPOINT_DELIVERY", True))
    feishu_card_v2: bool = field(default_factory=lambda: _env_bool("CT_FEISHU_CARD_V2", True))
    legacy_final_marker: bool = field(default_factory=lambda: _env_bool("CT_LEGACY_FINAL_MARKER", True))
    agent_tool_trace: bool = field(default_factory=lambda: _env_bool("CT_AGENT_TOOL_TRACE", False))

    # ===== Memory 2.2 (LLM extraction + Qdrant) =====
    memory_v22_enabled: bool = field(default_factory=lambda: _env_bool("CT_MEMORY_V22_ENABLED", False))
    """Master switch. Off → everything runs the v2.1 rule+SQLite pipeline.
    On → agent invocations trigger the LLM memory agent and write to Qdrant."""
    qdrant_url: str = field(default_factory=lambda: os.environ.get("CT_QDRANT_URL", "http://127.0.0.1:6333"))
    qdrant_collection: str = field(default_factory=lambda: os.environ.get("CT_QDRANT_COLLECTION", "memories"))
    qdrant_api_key: str | None = field(default_factory=lambda: os.environ.get("CT_QDRANT_API_KEY") or None)
    memory_agent_max_turns: int = field(default_factory=lambda: int(os.environ.get("CT_MEMORY_AGENT_MAX_TURNS", "10")))
    memory_agent_timeout: float = field(default_factory=lambda: float(os.environ.get("CT_MEMORY_AGENT_TIMEOUT", "120")))
    memory_extract_top_k: int = field(default_factory=lambda: int(os.environ.get("CT_MEMORY_EXTRACT_TOP_K", "5")))
    memory_extract_recent_turns: int = field(default_factory=lambda: int(os.environ.get("CT_MEMORY_EXTRACT_RECENT_TURNS", "10")))
    memory22_top_k: int = field(default_factory=lambda: int(os.environ.get("CT_MEMORY22_TOP_K", "8")))
    memory22_rrf_weight_vector: float = field(default_factory=lambda: float(os.environ.get("CT_MEMORY22_RRF_WEIGHT_VECTOR", "0.5")))
    memory22_rrf_weight_bm25: float = field(default_factory=lambda: float(os.environ.get("CT_MEMORY22_RRF_WEIGHT_BM25", "0.5")))
    feishu_progress_update_interval: float = field(
        default_factory=lambda: float(os.environ.get("CT_FEISHU_PROGRESS_UPDATE_INTERVAL", "3"))
    )
    feishu_progress_heartbeat_interval: float = field(
        default_factory=lambda: float(os.environ.get("CT_FEISHU_PROGRESS_HEARTBEAT_INTERVAL", "30"))
    )
    feishu_progress_timeline_limit: int = field(
        default_factory=lambda: int(os.environ.get("CT_FEISHU_PROGRESS_TIMELINE_LIMIT", "30"))
    )

    # Role bindings. Empty values are resolved from the registered healthy agents.
    coordinator_agent_id: str | None = field(
        default_factory=lambda: os.environ.get("CT_COORDINATOR_AGENT") or None
    )
    research_agent_id: str | None = field(
        default_factory=lambda: os.environ.get("CT_RESEARCH_AGENT") or None
    )
    coding_agent_id: str | None = field(
        default_factory=lambda: os.environ.get("CT_CODING_AGENT") or None
    )
    reviewer_agent_id: str | None = field(
        default_factory=lambda: os.environ.get("CT_REVIEWER_AGENT") or None
    )

    # Feishu enterprise self-built app. The integration remains disabled until
    # both app_id and app_secret are configured.
    feishu_app_id: str | None = field(default_factory=lambda: os.environ.get("FEISHU_APP_ID") or None)
    feishu_app_secret: str | None = field(
        default_factory=lambda: os.environ.get("FEISHU_APP_SECRET") or None
    )
    feishu_bot_open_id: str | None = field(
        default_factory=lambda: os.environ.get("FEISHU_BOT_OPEN_ID") or None
    )
    feishu_connection_mode: str = field(
        default_factory=lambda: os.environ.get("FEISHU_CONNECTION_MODE", "long_connection")
    )
    feishu_api_base_url: str = field(
        default_factory=lambda: os.environ.get("FEISHU_API_BASE_URL", "https://open.feishu.cn")
    )
    # Wiki spaces cannot be created with the tenant_access_token used by the
    # bot. Point the bot at a pre-created space (where the app is an
    # administrator/member), or let it auto-select a single accessible space.
    feishu_wiki_space_id: str | None = field(
        default_factory=lambda: os.environ.get("FEISHU_WIKI_SPACE_ID") or None
    )
    # Browser-facing tenant URL, for example
    # https://example.feishu.cn/wiki (not the OpenAPI host).
    feishu_wiki_base_url: str | None = field(
        default_factory=lambda: os.environ.get("FEISHU_WIKI_BASE_URL") or None
    )
    feishu_allowed_tenant_keys: list[str] = field(
        default_factory=lambda: _env_list("FEISHU_ALLOWED_TENANT_KEYS")
    )
    feishu_allowed_chat_ids: list[str] = field(
        default_factory=lambda: _env_list("FEISHU_ALLOWED_CHAT_IDS")
    )
    feishu_allowed_open_ids: list[str] = field(
        default_factory=lambda: _env_list("FEISHU_ALLOWED_OPEN_IDS")
    )
    allowed_workspace_roots: list[str] = field(
        default_factory=lambda: _env_list("CT_ALLOWED_WORKSPACE_ROOTS")
    )
    cors_origins: list[str] = field(default_factory=lambda: [
        o.strip() for o in os.environ.get("CT_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
        if o.strip()
    ])

    def ensure_dirs(self) -> None:
        for sub in [
            "", "sessions", "memory", "workspaces", "users", "agents/memory",
            "memory_migrations", "feishu", "feishu/runs", "feishu/run_events",
            "feishu/run_tools",
        ]:
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)

    @property
    def feishu_enabled(self) -> bool:
        return bool(self.feishu_app_id and self.feishu_app_secret)


settings = Settings()
settings.ensure_dirs()
