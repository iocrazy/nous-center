-- backend/migrations/wave1_memory.sql
-- Wave 1 · memory tables (2026-04-20)

CREATE TABLE IF NOT EXISTS memory_entries (
    id            BIGSERIAL PRIMARY KEY,
    instance_id   BIGINT NOT NULL REFERENCES service_instances(id) ON DELETE CASCADE,
    api_key_id    BIGINT,
    category      VARCHAR(32) NOT NULL,
    content       TEXT NOT NULL,
    context_key   VARCHAR(128),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mem_inst_created ON memory_entries (instance_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mem_inst_key_cat ON memory_entries (instance_id, context_key, category);
CREATE INDEX IF NOT EXISTS idx_mem_content_fts  ON memory_entries USING GIN (to_tsvector('simple', content));

CREATE TABLE IF NOT EXISTS memory_embeddings (
    entry_id      BIGINT PRIMARY KEY REFERENCES memory_entries(id) ON DELETE CASCADE,
    model         VARCHAR(64) NOT NULL,
    dim           INT NOT NULL,
    vector        BYTEA
);
