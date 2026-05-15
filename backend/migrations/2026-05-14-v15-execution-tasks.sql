-- backend/migrations/2026-05-14-v15-execution-tasks.sql
-- V1.5 Lane B · execution_tasks schema 扩展
-- spec: docs/superpowers/specs/2026-05-13-workflow-queue-and-gpu-scheduler-design.md §3.1
--
-- V1.5 把 image/TTS/LLM 推理从主进程串行执行改为 per-GPU-group runner 子进程
-- 调度。execution_tasks 需要 8 个新列记录调度元信息 + 可观测时间线：
--   * priority      — 2 级优先级（0=interactive / 10=batch），同级 FIFO
--   * gpu_group     — 落到哪个 hardware.yaml group（"llm-tp" / "image" / "tts"）
--   * runner_id     — 实际执行的 runner 实例 id
--   * queued_at     — 入队（DB commit）时刻；入队 sort key = (priority, queued_at)
--   * started_at    — dispatcher 弹出、标 running 的时刻
--   * finished_at   — completed/failed/cancelled 终态时刻
--   * node_timings  — 每节点耗时 JSON，每节点保留 cached:bool（V1.5 永远 false，V1.6 缓存用）
--   * cancel_reason — 取消原因（"user requested" / "node timeout" / "runner_crashed" ...）
--
-- 兼容性：8 列全部 nullable（priority 在 ORM 层有 default=10，DB 层 DEFAULT 10），
-- 旧行保持 NULL / 10，旧写入路径不传新列照常工作。
-- 单事务，IF NOT EXISTS 幂等（可重复跑）。
--
-- 部署：psql $DATABASE_URL -f backend/migrations/2026-05-14-v15-execution-tasks.sql

BEGIN;

ALTER TABLE execution_tasks
  ADD COLUMN IF NOT EXISTS priority      INTEGER NOT NULL DEFAULT 10,
  ADD COLUMN IF NOT EXISTS gpu_group     VARCHAR(32),
  ADD COLUMN IF NOT EXISTS runner_id     VARCHAR(32),
  ADD COLUMN IF NOT EXISTS queued_at     TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS started_at    TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS finished_at   TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS node_timings  JSONB,
  ADD COLUMN IF NOT EXISTS cancel_reason VARCHAR(200);

-- 调度器 dispatcher 按 (priority, queued_at) 排序弹队，且启动恢复时扫 status；
-- 加一个复合索引覆盖「按 group 找排队中 task」的热路径。
CREATE INDEX IF NOT EXISTS idx_execution_tasks_sched
  ON execution_tasks (status, gpu_group, priority, queued_at);

COMMIT;
