-- backend/migrations/2026-04-23-keys-m10.sql
-- m10 API Key 管理改造 · plan docs/designs/2026-04-22-ia-rebuild-v3.md
--
-- v3 IA 把 API Key 升级为一等公民。这次迁移加两列：
--   * secret_plaintext  — 明文常驻显示（参考阿里百炼模式：管理员重看 + reset）
--   * note              — 自由备注（"接 mediahub 主播放器"等）
--
-- 兼容性：两列都 nullable，旧 key 行 secret_plaintext 留 NULL；UI 在
-- 没有明文时降级为只显示 prefix，并提示用户 reset 一次即可获得明文。
-- 单事务、IF NOT EXISTS 幂等。

BEGIN;

ALTER TABLE instance_api_keys
  ADD COLUMN IF NOT EXISTS secret_plaintext VARCHAR(200),
  ADD COLUMN IF NOT EXISTS note             TEXT;

COMMIT;
