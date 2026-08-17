-- Memory 2.1 / Schema v4 vector schema — sqlite-vec virtual tables for ANN.
-- Source: word/4.2.Memory2.1最终版实施规划与架构.md §6.2
--
-- This file is applied ONLY when the sqlite-vec extension loads successfully.
-- The placeholder __EMBEDDING_DIM__ is replaced at runtime with
-- settings.memory_embedding_dim (default 512 for bge-small-zh-v1.5).
--
-- fact_embeddings stores one vector per fact.  tenant_id is the partition key.
-- The vec0 module provides approximate nearest-neighbor search:
--   SELECT fact_id, distance
--   FROM fact_embeddings
--   WHERE embedding MATCH ? AND k = 10
--     AND tenant_id = ?
--   ORDER BY distance

CREATE VIRTUAL TABLE IF NOT EXISTS fact_embeddings USING vec0(
  embedding FLOAT[__EMBEDDING_DIM__],
  fact_id    TEXT,
  tenant_id  TEXT,
  domain     TEXT,
  scope_id   TEXT
);
