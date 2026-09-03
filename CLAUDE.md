# nous-engine — Claude / AI agent notes

Single-admin inference infra (推理算力层). Repo/product renamed **nous-center →
nous-engine** (裸 `nous-` 前缀让给上层平台;systemd 单元全套 `nous-engine-*`,CLI
`enginectl`). Production deploy = `backend serve frontend/dist` on `:8000`, fronted by
cloudflared tunnel `api.iocrazy.com` (隧道名仍是 `nous-center`,Cloudflare 侧标识,未随
仓库改名). vite dev (`:9999`) is **local-only** for frontend hot reload.

## API endpoint vs UI route — DON'T MIX

| Need to hit | Use |
|---|---|
| Backend API | `/api/v1/keys`, `/api/v1/engines`, `/api/v1/services`, `/api/v1/workflows`... |
| UI route (browser address bar) | `/api-keys`, `/services`, `/workflows`, `/models`... |

The UI route `/api-keys` is the React Router path users see; the backend endpoint is
`/api/v1/keys` with no `api-` prefix. Calling `/api/v1/api-keys` returns 404.

## Operational

- Backend + cloudflared: systemd services. `sudo ./infra/systemd/install.sh`,
  then `journalctl -u nous-engine-backend -f` for logs. Don't `nohup ... & disown`.
  一键管控 `enginectl status|up|down|restart|logs`(装到 `/usr/local/bin/enginectl`)。
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
- **测试只跑 PostgreSQL,没有 sqlite**(2026-09-02 起,全局只有一种数据库)。conftest
  按 `DATABASE_URL`(`.env` 或环境变量,角色需 `CREATEDB`)建一个
  `nous_test_<worker>_<hex>` 临时库,建表一次、每个用例后 `TRUNCATE … RESTART IDENTITY
  CASCADE`、进程退出时 DROP。**新加 `src/models/xxx.py` 必须同步加进 conftest 顶部的
  `import src.models.xxx` 列表**,否则建表时不存在、用到它的用例报 UndefinedTable。
  被 SIGKILL 的跑批不走 atexit,残留库手动
  `DROP DATABASE "nous_test_…"`。别用 `NOUS_TEST_USE_REAL_DB=1`(直连真库的逃生口,会污染生产数据)。
- **并行**:`uv run pytest tests -n 8`(pytest-xdist,每 worker 各自临时库)本地约 65s,
  串行约 6 分钟;`-n 16` 不会更快。CI 用 `-n auto`。PG `max_connections=100` 是 worker
  数上限的天花板,别在 48 核机器上 `-n auto`。
- **测试绝不能真起推理服务碰 GPU**(2026-09-02 事故:本地生产 venv 装着 vllm,
  `test_vllm_adapter` 真起 `python -m vllm.entrypoints…`、`CUDA_VISIBLE_DEVICES` 被 adapter
  覆盖成 "0",多 worker 并发初始化 CUDA 把 RTX 3090 驱动跑挂,nvidia-smi 全 D 态,后端 API
  与桌面一起冻死,只能重启)。conftest 有 Popen 护栏:argv 含 `vllm.entrypoints` /
  `sglang.launch_server` / `sgl-omni` 直接 AssertionError,只有 `NOUS_RUN_GPU_TESTS=1`
  放行。写涉及 adapter.load() 的测试一律 mock `subprocess.Popen`。派 subagent 跑测试时,
  两个人**别同时**跑全量(runner 子进程用例会互相拖挂),且先 `nvidia-smi` 确认驱动活着。

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

## ComfyUI 桥 (comfy bridge)

- **模板即服务**:`POST /api/v1/comfy-templates {name, workflow}` 同时建
  `comfy_templates` 表行 + 一个 `ServiceInstance`(`source_type="comfy_template"`,
  服务名 = 传入的 `name`)。`PUT /api/v1/comfy-templates/{id}/mapping
  {exposed_params:[{key,label,type,comfy_node_id,comfy_input,...}]}` 定义哪些
  ComfyUI 节点输入可被参数化。`comfy_bridge.py::ComfyUIWorkflowNode.invoke` 取值是
  `data.get(key, m.get("default"))`——prediction `input` 里显式给的值优先,没给才落
  mapping 的 `default`;**只有 mapping 也没设 `default` 时才用 workflow 原值**(注册时
  冻结的快照)。想让某个 key 在不传 input 时也走某个值(如把时长压到最短),直接在
  exposed_params 里给它写 `default`,不必去改 workflow 导出文件本身。
- **长任务走 respond-async**:`POST /v1/services/{name}/predictions`(注意前缀是
  `/v1/`,不是 `/api/v1/`)带 `Prefer: respond-async` → 202 `{id,status:"starting"}`,
  轮询 `GET /v1/predictions/{id}` 到终态(`succeeded|failed|canceled`)。鉴权跟
  `/api/v1/*` 管理端点是两套:`/v1/*` 带了 `Authorization` header 优先走 M:N
  `InstanceApiKey` 校验,校验失败(含 `ADMIN_TOKEN` 这种压根不是 InstanceApiKey 的
  bearer)会退回 admin-session 校验(cookie 或 `Authorization: Bearer $ADMIN_TOKEN`
  都认)——脚本化调用(无浏览器 cookie)可以直接用 `ADMIN_TOKEN` 当 bearer,也可以
  先用它经 `POST /api/v1/keys {label, service_ids:[<service 数字 id>]}` 铸一把授权给该
  service 的 M:N key,再拿它的 `secret` 调 predictions(两条路都通)。
- **选项可依赖另一个参数**:`exposed_params` 项上写
  `options_depends_on: "<另一个 key>"` + `options_source: "comfy_styles"`,该字段的
  选项清单就在**运行期**按依赖参数的当前值拉(krea2:`styles` 随 `style_pack` 切换),
  不再只认注册时冻结的那份静态 enum。链路:mapping → `constraints.options_*` →
  schema 的 `x-options-depends-on`/`x-options-source` → 前端 `DependentOptionField`
  按 `GET /api/v1/comfy/styles?pack=` 拉 + 后端 `comfy/style_options.py`
  (10 分钟 TTL 缓存)按包校验。**静态 enum 照旧写**,是 sidecar 不可达时的兜底
  (拿不到清单只 warning + 退回静态,绝不让预测 500)。`options_depends_on` 指向不存在
  的 key 在 PUT mapping 时就拒(400 `validation_error`)。
  krea2 那批风格包的缩略图是 sidecar 侧**相对路径**,经
  `GET /api/v1/comfy/style-image?path=<文件路径>` 代理:**只收文件路径、不收 URL**,
  路由写死 + httpx `params=` 传参。别改回"收 src 再转发 + 前缀白名单"——httpx 合并
  相对 URL 会归一化点段,`/easyuse/../history` 到 sidecar 就是 `/history`,白名单形同
  虚设(2026-09-03 审查实测)。缓存头是 `private`(端点要 admin 鉴权,`public` 会让
  cloudflared/中间缓存把字节回给未鉴权者)。
- 动态清单的取数(`style_options.py`)有三条闸门:① 依赖参数的值必须先落在它自己的
  静态 enum 里(没 enum 则只认 default)才会去打 sidecar —— 这段跑在
  `validate_service_input` **之前**,不设闸等于让任何持 key 的人拿随机包名驱动出站请求
  + 撑大进程内缓存;② 缓存 256 条上限、按插入序驱逐,失败与空结果只缓存 30s(负缓存),
  成功缓存 10 分钟;③ 预校验取数用 5s 短超时(列清单那条仍是 15s)。
  **「取不到」(None,退回静态 enum)与「取到了但为空」([],空白名单全拒)是两回事**,
  后者退回默认包的静态 enum 正是本机制要防的错配。
- **`multiple` 与静态 options 正交**:`multiple` 说的是**值的形状**(逗号分隔串),
  跟「有没有在注册时冻结一份静态 enum」无关 —— 只声明依赖、不带 options 的 mapping
  同样要落 `constraints.multiple` / schema 的 `x-multiple`,否则运行期会拿动态清单去
  整串比对 `'a,b'`,多选必 422。
- **文件类参数(image/file/audio/video/binary/media)不能声明 `options_depends_on`
  /`options_source`**:PUT mapping 直接 400 `validation_error`。它的值是上传的文件,
  挂上动态清单等于给上传字段发一份风格名白名单,上传必 422。schema 侧对文件类也不
  输出 `x-options-*`(双保险,兜老数据)。
- **已知限制**:`_thumbnail_url` 把 sidecar 的相对缩略图改写成 admin-only 的代理地址
  (`/api/v1/comfy/style-image`)。如果编辑器把某个动态包的 options **冻结进 mapping**,
  `/v1/services/{name}/schema` 里的 `x-option-meta[].image` 对只持 `InstanceApiKey` 的
  第三方就是 401 —— 缩略图渲不出来(值本身照常可用)。前端 Playground 走 admin 会话不
  受影响。要给第三方也能看的缩略图,得先给这个代理端点开一条 InstanceApiKey 能过的路。
- env 三件套:`NOUS_COMFY_URL`(sidecar 地址,默认 `http://127.0.0.1:8188`)、
  `NOUS_COMFY_TIMEOUT`(渲染等待上限,默认 14400s)、
  `NOUS_COMFY_DOWNLOAD_TIMEOUT`(产物下载超时,默认 120s)。
- **渲染串行化 = 单进程部署前提**:桥节点(`comfy_bridge.py`)靠一个模块级
  `asyncio.Semaphore(1)`(`_SEM`)保证「sidecar 一次只服务一个渲染」,这假设 uvicorn
  跑无 `--workers`(生产就是这样)。上了多 worker/多进程部署,每个进程各有自己的
  `_SEM`,互不认识,这条不变式就破了——需要先换成跨进程锁(Postgres advisory lock
  之类)才能加 `--workers`。
- sidecar 是独立 systemd 单元 `nous-engine-comfyui`,默认 **disabled**(装机时模板
  路径 `/opt/comfyui` 多半还没铺,`enginectl restart` 不会强启一个禁用单元——2026-08-10
  设计修正,见 `infra/systemd/enginectl`)。启用前置:① GSP 缓解脚本已上机
  (`infra/gpu/setup-gpu-mitigations.sh`,Pro 6000 满载崩卡 bug);② 核对单元里的
  `CUDA_VISIBLE_DEVICES` 跟目标卡对得上(`infra/systemd/nous-engine-comfyui.service`)。
  `enginectl status|up|down|restart|logs comfyui` 纳管。
- 真机 smoke(非 CI,需 sidecar 已跑 + 权重已入 ComfyUI models):
  `cd backend && uv run python tests/manual/smoke_comfy_h3.py --base http://127.0.0.1:8000
  --admin-token $ADMIN_TOKEN --workflow /path/to/xxx-api.json --mapping /path/to/mapping.json`
  ——建模板 → 设 mapping → 铸临时 key → 异步 prediction → 轮询 → 下载 mp4 →
  `ffprobe` 断言存在 video + audio 流 → 跑完自动清理(临时 key + 模板/服务),
  `--keep` 保留。
- 前端联调注意:`npm run build` 实际是 `prebuild`(跑 `wasm:build`,需要
  `wasm-pack`)+ `tsc -b && vite build`——worktree 里没装 `wasm-pack` 会卡在 prebuild;
  `tsc -b`(project references 构建)不能拿裸 `npx tsc` 替代,后者不认 composite
  引用会报一堆假错误。

## Memory

User's persistent memory lives in `~/.claude/projects/.../memory/MEMORY.md`. Index
of feedback/preferences/project context. Auto-loaded into context. Read it before
making framing decisions.
