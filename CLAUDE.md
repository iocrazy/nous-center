# nous-center — Claude / AI agent notes

Single-admin inference infra. Production deploy = `backend serve frontend/dist` on
`:8000`, fronted by cloudflared tunnel `api.iocrazy.com`. vite dev (`:9999`) is
**local-only** for frontend hot reload.

## API endpoint vs UI route — DON'T MIX

| Need to hit | Use |
|---|---|
| Backend API | `/api/v1/keys`, `/api/v1/engines`, `/api/v1/services`, `/api/v1/workflows`... |
| UI route (browser address bar) | `/api-keys`, `/services`, `/workflows`, `/models`... |

The UI route `/api-keys` is the React Router path users see; the backend endpoint is
`/api/v1/keys` with no `api-` prefix. Calling `/api/v1/api-keys` returns 404.

## Operational

- Backend + cloudflared: systemd services. `sudo ./infra/systemd/install.sh`,
  then `journalctl -u nous-backend -f` for logs. Don't `nohup ... & disown`.
- Admin secrets: `./infra/security/gen-admin-secrets.sh > /tmp/secrets && cat /tmp/secrets`
  then paste into `backend/.env`. Three values: `ADMIN_PASSWORD` (browser cookie login),
  `ADMIN_SESSION_SECRET` (HMAC key), `ADMIN_TOKEN` (CLI bearer).
- Production frontend changes need `cd frontend && npm run build` after merge —
  backend serves `frontend/dist/`, not the source.
- Dev backend (manual, not systemd): `backend/scripts/dev-serve.sh` — sources `.env`
  (uv won't), runs uvicorn, tees stdout to `backend/logs/backend-dev.log` (50MB rotate).
  Structured request/audit/app/frontend logs go to the **main PostgreSQL DB**
  (4 tables via `src/models/log_entry.py`, written through `log_store.py`'s async
  queue + single batch consumer; view via `/api/v1/logs/*` or the frontend
  LogsOverlay) regardless of stdout. There is no longer a separate SQLite
  `log_db` — one DB (spec `docs/superpowers/specs/2026-06-10-log-db-merge-into-postgres-design.md`).
  Production stays on journald for raw stdout.

## Testing

- Backend tests run with `ADMIN_PASSWORD=""` forced in `tests/conftest.py` so the
  admin gate is off during the suite. Don't unset that.
- SPA catch-all is disabled in tests via `NOUS_DISABLE_FRONTEND_MOUNT=1`
  (also set in conftest). If you add a new test that registers routes after
  `create_app()`, this matters — otherwise the catch-all swallows them.

## Performance

- `/api/v1/engines`, `/api/v1/services`, `/api/v1/workflows` are wrapped with
  `@cached("prefix", ttl=30)` from `src/api/response_cache.py`. Any new write
  path that mutates these lists must call `invalidate("prefix")` (cross-resource
  writes pass multiple prefixes — see `workflow_publish.py`).
- ETag is computed on the serialized body bytes, not the dict — keeps it stable
  across non-deterministic dict/set iteration order.

## 图像引擎 (image engine)

- 引擎只剩一套 = `ModularImageBackend`(`image_modular.py`,Modular Diffusers)。
  迁移已完成,**legacy 自写 `ImageSampler`/`image_diffusers.py`/`image_sampler.py` 已删**
  (#128-132);`NOUS_IMAGE_ENGINE` 环境变量已无 legacy 选项。Anima 自定义 DiT 走
  `image_anima.py`。spec
  `docs/superpowers/specs/2026-05-22-image-engine-modular-diffusers-design.md`。
- **Modular Diffusers 是 experimental**;`diffusers` 在 `pyproject.toml` **钉死 commit**。
  改 `image_modular.py` **或升 diffusers 前,必须跑**
  `tests/manual/smoke_image_ab.py`(真模型/GPU,非 CI)并确认 SSIM ≥ 0.97 + 出图正确,
  再 bump commit。CI 跑不了真模型(conftest mock torch + 无 GPU),引擎正确性只靠这个
  standalone smoke。该 smoke 现在是 **golden 回归比对**(legacy 没了,不再是 legacy/modular
  A/B):重生成 modular 出图 → SSIM 比保存的 golden 图。
- **standalone smoke 必须在 import torch 前设 `CUDA_DEVICE_ORDER=PCI_BUS_ID`**(脚本顶部
  `os.environ.setdefault` 或命令前缀)。否则 torch 默认 FASTEST_FIRST 把 Pro 6000 排到
  `cuda:0`、`cuda:1` 变成 24G 的 3090 → `SMOKE_DEVICE=cuda:1` 装 9B 模型直接 OOM。生产
  经 `src/api/main.py` 已 setdefault,但 standalone 脚本不经它、且 `uv` 不 load `.env`。
- `diffusers.modular*` 的 import **只允许在 `image_modular.py`**(`_import_modular()`
  一处)——experimental API 变更时 blast radius 限一文件。

## Memory

User's persistent memory lives in `~/.claude/projects/.../memory/MEMORY.md`. Index
of feedback/preferences/project context. Auto-loaded into context. Read it before
making framing decisions.
