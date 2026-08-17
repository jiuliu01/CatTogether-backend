-- Memory 2.1 / Schema v4 — all DDL for the final four-layer SQLite database.
-- Source: word/4.2.Memory2.1最终版实施规划与架构.md §6
--
-- Four layers share one memory.db (WAL mode), separated by table prefix:
--   Hot Memory:   hot_memories
--   Fact Memory:  facts + facts_fts + entities + entity_aliases
--                  + fact_entity_mentions + fact_relations + events
--                  + transaction_outbox + memory_audit_log
--   History:      session_messages + session_messages_fts
--   Skill:        skill_registry + skill_versions
--   Cross-layer:  scope_registry + schema_migrations
--
-- Key changes from v3:
--   * facts: removed denormalized entities/source_type/source_ids/supersedes_id;
--             added search_text, agent_id, task_id; schema_version=4;
--             status adds 'pending_review'.
--   * All business tables use UNIQUE(tenant_id, id) instead of id PRIMARY KEY.
--   * FTS5 indexes search_text (not text) for facts, search_text for history.
--   * entities: canonical_name + last_seen_at + merged_into_id.
--   * fact_entity_mentions: mention_text + linking_confidence.
--   * fact_relations: confidence + created_by + relation adds 'related'.
--   * hot_memories: priority + source_fact_ids + proposed_by.
--   * session_messages: search_text + redaction_status.
--   * skill_registry + skill_versions split.

-- ===== facts main table =====
CREATE TABLE IF NOT EXISTS facts (
  id            TEXT NOT NULL,
  tenant_id     TEXT NOT NULL,
  domain        TEXT NOT NULL CHECK (domain IN ('user','project','task','agent')),
  scope_id      TEXT NOT NULL,
  agent_id      TEXT,
  task_id       TEXT,
  kind          TEXT NOT NULL CHECK (kind IN (
    'preference','correction','project_fact','decision',
    'progress','convention','goal','constraint','outcome','relation'
  )),
  text          TEXT NOT NULL,
  search_text   TEXT NOT NULL DEFAULT '',
  tags          TEXT NOT NULL DEFAULT '[]',        -- JSON array
  importance    REAL NOT NULL DEFAULT 0.5,
  confidence    REAL NOT NULL DEFAULT 0.5,
  status        TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active','pending_review','superseded','archived','disputed')),
  temporal      TEXT,                              -- JSON TemporalInfo
  provenance    TEXT NOT NULL,                     -- JSON ProvenanceInfo
  embedding_model TEXT,
  created_at    TEXT NOT NULL,
  updated_at    TEXT NOT NULL,
  expires_at    TEXT,
  version       INTEGER NOT NULL DEFAULT 1,
  schema_version INTEGER NOT NULL DEFAULT 4,
  PRIMARY KEY (tenant_id, id)
);

CREATE INDEX IF NOT EXISTS idx_facts_scope
  ON facts (tenant_id, domain, scope_id, status);
CREATE INDEX IF NOT EXISTS idx_facts_kind
  ON facts (tenant_id, domain, kind);
CREATE INDEX IF NOT EXISTS idx_facts_temporal
  ON facts (tenant_id, domain, scope_id);
CREATE INDEX IF NOT EXISTS idx_facts_updated
  ON facts (updated_at);
CREATE INDEX IF NOT EXISTS idx_facts_agent
  ON facts (tenant_id, agent_id);
CREATE INDEX IF NOT EXISTS idx_facts_task
  ON facts (tenant_id, task_id);

-- ===== facts_fts (FTS5 external-content, BM25 search on search_text) =====
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
  search_text, tags, kind, domain, scope_id, tenant_id,
  content='facts', content_rowid='rowid',
  tokenize='trigram'
);
-- Sync triggers (INSERT/DELETE/UPDATE)
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
  INSERT INTO facts_fts(rowid, search_text, tags, kind, domain, scope_id, tenant_id)
  VALUES (new.rowid, new.search_text, new.tags, new.kind, new.domain, new.scope_id, new.tenant_id);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, search_text, tags, kind, domain, scope_id, tenant_id)
  VALUES ('delete', old.rowid, old.search_text, old.tags, old.kind, old.domain, old.scope_id, old.tenant_id);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, search_text, tags, kind, domain, scope_id, tenant_id)
  VALUES ('delete', old.rowid, old.search_text, old.tags, old.kind, old.domain, old.scope_id, old.tenant_id);
  INSERT INTO facts_fts(rowid, search_text, tags, kind, domain, scope_id, tenant_id)
  VALUES (new.rowid, new.search_text, new.tags, new.kind, new.domain, new.scope_id, new.tenant_id);
END;

-- ===== events =====
CREATE TABLE IF NOT EXISTS events (
  event_id         TEXT NOT NULL,
  tenant_id        TEXT NOT NULL,
  event_type       TEXT NOT NULL,
  actor            TEXT NOT NULL,       -- JSON Actor
  scope_hint       TEXT,                -- JSON MemoryScope
  payload          TEXT NOT NULL,       -- JSON
  allowed_domains  TEXT NOT NULL DEFAULT '[]',
  created_at       TEXT NOT NULL,
  processed_at     TEXT,
  processing_stage TEXT,
  attempt_count    INTEGER NOT NULL DEFAULT 0,
  status           TEXT NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending','processed','failed','skipped')),
  error            TEXT,
  fact_ids         TEXT NOT NULL DEFAULT '[]',  -- JSON array
  PRIMARY KEY (tenant_id, event_id)
);
CREATE INDEX IF NOT EXISTS idx_events_status ON events (tenant_id, status, created_at);

-- ===== entities (canonical entity main table, §5.4) =====
CREATE TABLE IF NOT EXISTS entities (
  entity_id      TEXT NOT NULL,
  tenant_id      TEXT NOT NULL,
  canonical_name TEXT NOT NULL,
  type           TEXT NOT NULL,
  aliases        TEXT NOT NULL DEFAULT '[]',
  first_seen_at  TEXT NOT NULL,
  last_seen_at   TEXT NOT NULL,
  merged_into_id TEXT,
  PRIMARY KEY (tenant_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_entities_tenant_type ON entities (tenant_id, type);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities (tenant_id, canonical_name);

-- ===== entity_aliases =====
CREATE TABLE IF NOT EXISTS entity_aliases (
  entity_id  TEXT NOT NULL,
  alias      TEXT NOT NULL,
  tenant_id  TEXT NOT NULL,
  PRIMARY KEY (tenant_id, entity_id, alias)
);
CREATE INDEX IF NOT EXISTS idx_alias_text ON entity_aliases (tenant_id, alias);

-- ===== fact_entity_mentions (§5.4) =====
CREATE TABLE IF NOT EXISTS fact_entity_mentions (
  tenant_id           TEXT NOT NULL,
  fact_id             TEXT NOT NULL,
  entity_id           TEXT,
  mention_text        TEXT NOT NULL DEFAULT '',
  role                TEXT NOT NULL DEFAULT 'mentions'
                      CHECK (role IN ('mentions','subject','object')),
  linking_confidence  REAL NOT NULL DEFAULT 0.5,
  PRIMARY KEY (tenant_id, fact_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_fem_entity ON fact_entity_mentions (tenant_id, entity_id);
CREATE INDEX IF NOT EXISTS idx_fem_fact ON fact_entity_mentions (tenant_id, fact_id);

-- ===== fact_relations (§5.7, ADD-only) =====
CREATE TABLE IF NOT EXISTS fact_relations (
  tenant_id    TEXT NOT NULL,
  src_fact_id  TEXT NOT NULL,
  dst_fact_id  TEXT NOT NULL,
  relation     TEXT NOT NULL
               CHECK (relation IN ('supersedes','supports','contradicts','derived_from','related')),
  confidence   REAL NOT NULL DEFAULT 1.0,
  created_by   TEXT NOT NULL DEFAULT 'system',
  created_at   TEXT NOT NULL,
  PRIMARY KEY (tenant_id, src_fact_id, dst_fact_id, relation)
);
CREATE INDEX IF NOT EXISTS idx_fr_src ON fact_relations (tenant_id, src_fact_id, relation);
CREATE INDEX IF NOT EXISTS idx_fr_dst ON fact_relations (tenant_id, dst_fact_id, relation);

-- ===== hot_memories (§5.8) =====
CREATE TABLE IF NOT EXISTS hot_memories (
  id              TEXT NOT NULL,
  tenant_id       TEXT NOT NULL,
  domain          TEXT NOT NULL,
  scope_id        TEXT NOT NULL,
  text            TEXT NOT NULL,
  priority        INTEGER NOT NULL DEFAULT 0,
  source_fact_ids TEXT NOT NULL DEFAULT '[]',  -- JSON array
  status          TEXT NOT NULL DEFAULT 'active'
                  CHECK (status IN ('pending_approval','active','archived')),
  proposed_by     TEXT NOT NULL DEFAULT 'system',
  approved_by     TEXT,
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  PRIMARY KEY (tenant_id, id)
);
CREATE INDEX IF NOT EXISTS idx_hot_scope ON hot_memories (tenant_id, domain, scope_id, status);

-- ===== session_messages (History layer, §5.9) =====
CREATE TABLE IF NOT EXISTS session_messages (
  message_id      TEXT NOT NULL,
  tenant_id       TEXT NOT NULL,
  channel_id      TEXT NOT NULL,
  thread_id       TEXT,
  seq             INTEGER NOT NULL,
  role            TEXT NOT NULL,
  agent_id        TEXT,
  content         TEXT NOT NULL,
  search_text     TEXT NOT NULL DEFAULT '',
  tool_call       TEXT,    -- JSON
  file_diff       TEXT,    -- JSON
  redaction_status TEXT NOT NULL DEFAULT 'clean'
                   CHECK (redaction_status IN ('clean','redacted','encrypted','blocked')),
  created_at      TEXT NOT NULL,
  PRIMARY KEY (tenant_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_session_channel ON session_messages (tenant_id, channel_id, thread_id, seq);

CREATE VIRTUAL TABLE IF NOT EXISTS session_messages_fts USING fts5(
  search_text, channel_id, tenant_id,
  content='session_messages', content_rowid='rowid',
  tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS sm_ai AFTER INSERT ON session_messages BEGIN
  INSERT INTO session_messages_fts(rowid, search_text, channel_id, tenant_id)
  VALUES (new.rowid, new.search_text, new.channel_id, new.tenant_id);
END;
CREATE TRIGGER IF NOT EXISTS sm_ad AFTER DELETE ON session_messages BEGIN
  INSERT INTO session_messages_fts(session_messages_fts, rowid, search_text, channel_id, tenant_id)
  VALUES ('delete', old.rowid, old.search_text, old.channel_id, old.tenant_id);
END;
CREATE TRIGGER IF NOT EXISTS sm_au AFTER UPDATE ON session_messages BEGIN
  INSERT INTO session_messages_fts(session_messages_fts, rowid, search_text, channel_id, tenant_id)
  VALUES ('delete', old.rowid, old.search_text, old.channel_id, old.tenant_id);
  INSERT INTO session_messages_fts(rowid, search_text, channel_id, tenant_id)
  VALUES (new.rowid, new.search_text, new.channel_id, new.tenant_id);
END;

-- ===== skill_registry (§5.10) =====
-- v4: single-table design with tenant-scoped PK and v4 status enum.
-- The split skill_versions design will be layered on top in Stage 5.
CREATE TABLE IF NOT EXISTS skill_registry (
  id                  TEXT NOT NULL,
  tenant_id           TEXT NOT NULL,
  domain              TEXT NOT NULL,
  scope_id            TEXT NOT NULL,
  name                TEXT NOT NULL,
  description         TEXT NOT NULL DEFAULT '',
  body                TEXT NOT NULL DEFAULT '',
  version             INTEGER NOT NULL DEFAULT 1,
  previous_version_id TEXT,
  tags                TEXT NOT NULL DEFAULT '[]',
  status              TEXT NOT NULL DEFAULT 'active'
                      CHECK (status IN ('draft','pending_approval','active','archived')),
  created_at          TEXT NOT NULL,
  updated_at          TEXT NOT NULL,
  PRIMARY KEY (tenant_id, id)
);
CREATE INDEX IF NOT EXISTS idx_skill_scope ON skill_registry (tenant_id, domain, scope_id, status);
CREATE INDEX IF NOT EXISTS idx_skill_name ON skill_registry (tenant_id, name, version);

-- ===== skill_versions (§5.10) =====
CREATE TABLE IF NOT EXISTS skill_versions (
  tenant_id      TEXT NOT NULL,
  skill_id       TEXT NOT NULL,
  version        INTEGER NOT NULL,
  snapshot_path  TEXT NOT NULL,
  checksum       TEXT NOT NULL,
  created_by     TEXT NOT NULL DEFAULT 'system',
  created_at     TEXT NOT NULL,
  change_summary TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (tenant_id, skill_id, version)
);

-- ===== scope_registry =====
CREATE TABLE IF NOT EXISTS scope_registry (
  tenant_id  TEXT NOT NULL,
  domain     TEXT NOT NULL,
  scope_id   TEXT NOT NULL,
  agent_id   TEXT,
  first_seen TEXT NOT NULL,
  last_seen  TEXT NOT NULL,
  PRIMARY KEY (tenant_id, domain, scope_id, agent_id)
);

-- ===== transaction_outbox (async projection / compensation) =====
CREATE TABLE IF NOT EXISTS transaction_outbox (
  id           TEXT NOT NULL,
  tenant_id    TEXT NOT NULL,
  fact_id      TEXT,
  op           TEXT NOT NULL,    -- 'insert'/'update'/'delete'/'reindex'
  payload      TEXT NOT NULL,    -- JSON
  created_at   TEXT NOT NULL,
  processed_at TEXT,
  PRIMARY KEY (tenant_id, id)
);
CREATE INDEX IF NOT EXISTS idx_outbox_pending ON transaction_outbox (processed_at);

-- ===== memory_audit_log (append-only, no body text) =====
CREATE TABLE IF NOT EXISTS memory_audit_log (
  id          TEXT NOT NULL,
  tenant_id   TEXT NOT NULL,
  action      TEXT NOT NULL,    -- 'write'/'update'/'delete'/'forget'/'approve'/'supersede'
  domain      TEXT,
  fact_id     TEXT,
  actor       TEXT NOT NULL,
  reason      TEXT,
  created_at  TEXT NOT NULL,
  PRIMARY KEY (tenant_id, id)
);

-- ===== schema_migrations =====
CREATE TABLE IF NOT EXISTS schema_migrations (
  version     INTEGER PRIMARY KEY,
  applied_at  TEXT NOT NULL,
  description TEXT
);
