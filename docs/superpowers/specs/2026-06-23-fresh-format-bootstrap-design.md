# 裸机格式化后的一键 bootstrap(从空系统到全栈在跑)

- Date: 2026-06-23
- Status: 设计
- Trigger: 用户问「下次格式化进来能否一键安装服务?」。现状是 `infra/systemd/install.sh`
  只装 systemd 单元(服务层一键),但格式化后的**前置依赖**(原生 PG、Python/Node 依赖、
  admin secret、cloudflared 凭证、aligner venv、专用 prod 检出)散落在 `infra/PROD_CHECKOUT.md`
  的手抄命令序列里,没有任何脚本统揽。单管理员自托管机,目标是「重装系统后一条命令到
  全栈在跑」。长远稳 > 省事。

## 1. 背景 / 现状

### 1.1 已经一键的部分

`sudo ./infra/systemd/install.sh` 幂等装好全部单元:
`nous-backend` / `nous-cloudflared` / `nous-status` / `nous-aligner` 服务
+ `nous-healthprobe` / `nous-dbbackup` timer + `nous.target` + `nousctl` + 免密 sudoers。
开机自启 + 失败自重启。**这一层不动。**

### 1.2 格式化后缺口盘点(install.sh 之前必须先就位)

| 前置 | 现状坐标 | 一键现状 |
|---|---|---|
| 原生 PostgreSQL 17 + role/库 | `docs/.../2026-06-18-native-pg-systemd-stack-design.md`(Phase1 已落地,Phase2 待迁盘) | 手动(spec 有命令,无单脚本) |
| DB 数据恢复(从备份) | `infra/db/backup.sh`(pg_dump→大盘),无 restore 脚本 | ❌ |
| 后端 venv `uv sync --extra inference` | `PROD_CHECKOUT.md` 第 3 步;漏 `--extra` → 常驻 0/N | 手动 |
| 前端 `npm ci && npm run build` | `PROD_CHECKOUT.md` 第 4 步 | 手动 |
| `backend/.env` 三个 admin secret | `infra/security/gen-admin-secrets.sh` | 半自动 |
| cloudflared 二进制 + tunnel 凭证 | 无脚本;凭证是机器特定的 secret | ❌ |
| aligner 独立 venv | `infra/aligner/setup.sh` | 单独跑 |
| 专用 prod 检出 + `.nous-production` | `PROD_CHECKOUT.md` 一次性序列 | 手抄 |
| GPU 驱动 / CUDA_DEVICE_ORDER | `backend/.env` 强制;驱动靠系统装 | OS 层,不纳入 |

### 1.3 难点:哪些能脚本化,哪些天生要人/机器特定 secret

- **可完全自动**:apt 装包、建 cluster/role/库、uv sync、npm build、aligner venv、
  install.sh、生成 admin secret。
- **机器特定 secret(无法 commit,必须外部带入)**:cloudflared tunnel 凭证
  (`~/.cloudflared/*.json` + cert)、DB 密码、HF token。bootstrap 不能凭空造,只能
  **检测缺失 → 明确报缺 + 指到恢复来源**(备份盘 / 1Password / `gen-admin-secrets.sh`)。
- **数据**:DB 内容靠最近一次 `nous-dbbackup` 的 dump 恢复;模型权重在大盘不入 git,
  格式化若保留大盘则无需重下。

## 2. 目标设计

一个分阶段、幂等、可断点续跑的 `infra/bootstrap.sh`,把 1.2 全部串起来。
**不取代** install.sh,而是在它之前补齐前置,最后调它。

### 2.1 形态

```
sudo ./infra/bootstrap.sh                # 全量:检测 → 装缺的 → 起全栈 → 自检
sudo ./infra/bootstrap.sh --check        # 只体检(dry-run),打印每项 OK/缺,不改系统
sudo ./infra/bootstrap.sh --stage db     # 只跑某阶段(db|deps|secrets|build|services)
```

阶段顺序(每阶段先检测「已就位?」幂等跳过):

```
preflight  → OS/盘/驱动/网络 体检(只读,缺关键项即停并报清单)
db         → apt 装 pg17 + 建 cluster/role/库 + (可选)从备份 restore
secrets    → .env 缺则 gen-admin-secrets;cloudflared 凭证缺则报缺指源(不自动造)
deps       → 后端 uv sync --extra inference + aligner venv(setup.sh)
build      → 前端 npm ci + npm run build(dist 时间戳校验)
checkout   → (可选)建/校验专用 prod 检出 + .nous-production 标记
services   → 调 install.sh + nousctl up + 自检 banner(/healthz 本机+公网)
```

### 2.2 关键设计点

- **幂等 + 断点续跑**:每阶段开头判定目标态(库已存在?dist 比源新?venv 有 vllm?),
  已达成则跳过。任一阶段失败可单独 `--stage X` 重跑。
- **secret 不入 git、不自动伪造**:缺 cloudflared 凭证时打印
  「从 <备份盘路径> 复制 `~/.cloudflared/` 或重跑 `cloudflared tunnel login`」并把该阶段
  标记为「需人工」,继续跑能跑的,最后汇总待办。
- **DB restore 可选**:`--restore <dump>` 显式传 dump 才恢复;默认建空库由 backend
  lifespan `create_all` 自建 schema(单管理员、无 alembic,与现状一致)。
- **复用现有脚本**,不重写:db 段调迁移/建库逻辑,aligner 调 `infra/aligner/setup.sh`,
  secret 调 `gen-admin-secrets.sh`,services 调 `install.sh`。bootstrap 是**编排器**。
- **与 prod/dev 检出分离兼容**:bootstrap 默认在当前检出跑;`--prod` 时建/用 nous-prod。
- **PG 落盘路径**承接 native-pg spec 的 Phase2(datahub/大盘)结论,不在此 spec 重定。

## 3. 实施(分独立 PR,各走 CI/CD)

- **PR-1**:`bootstrap.sh` 骨架 + `--check` 体检模式(纯只读,打印每项 OK/缺 + 恢复指引)。
  先把「缺什么」一屏看清,零风险,可立即在真机跑验证清单完整。
- **PR-2**:`db` + `deps` + `build` 阶段(可自动化、无 secret 的部分),幂等。
- **PR-3**:`secrets` + `checkout` + `services` 阶段(含人工 secret 的检测/指引 + 调
  install.sh + 自检 banner),收口成「一条命令」。
- **PR-4**:文档 —— `PROD_CHECKOUT.md` 手抄序列改为指向 bootstrap;新增「格式化重建 SOP」。

> 每个 PR 单独分支、preflight 跑 ruff+tsc+vite build、CI Backend 绿后再合。
> bootstrap 本身是 bash,CI 至少过 shellcheck(若仓库已有)/`bash -n` 语法检查。

## 4. 验证

- `--check` 在当前(已就位)机器上跑 → 应全 OK,零改动。
- 真正的端到端验证只能在**真格式化 / 干净 VM** 上做;在此之前以 `--check` + 各阶段
  幂等「已就位即跳过」为主要保证。真机重装时按 SOP 实跑一次并回灌修正。

## 5. 非目标

- 不装 NVIDIA 驱动 / CUDA(OS 层,交给系统重装流程;bootstrap 只 `--check` 报缺)。
- 不管理模型权重下载(在大盘、不入 git;格式化保留大盘即可)。
- 不造任何机器特定 secret(只检测 + 指源)。
