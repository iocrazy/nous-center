# 原生 PostgreSQL + systemd 整栈托管(脱离 Docker Desktop)+ 启动自检 banner + 自动备份

- Date: 2026-06-18
- Status: 设计
- Trigger: 用户数据库现跑在 **Docker Desktop**(context=desktop-linux,VM 内 + 用户态 socket + 不接 systemd 开机链)。诉求三条:(1) DB 改命令行 / systemd 托管,随主栈一起起;(2) 一条短命令拉起全栈(现在要分别点 Desktop + `systemctl start nous-backend` + `nous-cloudflared`);(3) 启动要有**自检面板**(参照 PAPERCLIP 的 banner:DB/迁移/备份/心跳一屏看清),现在 systemd 起完一片黑。单管理员自托管推理机,长远稳 > 省事。

## 1. 背景 / 现状

### 1.1 根因:Docker Desktop 不适合做常驻数据层

- `docker context ls` → 当前是 `desktop-linux *`(Docker Desktop),**非原生 dockerd**。
- Docker Desktop on Linux 跑在 **VM** 里(多吃 ~2G 内存),socket 是用户态 `~/.docker/desktop/docker.sock`,需要图形登录 + Desktop GUI 起着。**不接 systemd 开机链** → 没法"随 systemctl 一起起",冷重启后不保证自启。
- `nous-backend.service` 里**已写** `After=network-online.target postgresql.service`,但系统中**根本没有** `postgresql.service`(pg 在 Desktop 里)→ 这行是**空操作**,数据库现在压根不在启动依赖链上。这是"要手动分头起"的直接原因。

### 1.2 现状盘点

| 关注点 | 现状坐标 |
|---|---|
| DB 引擎 | Docker Desktop 容器 `nous-center-postgres-1`,`postgres:17-alpine`(实际 **17.9**),`restart: always` + healthcheck |
| 端口 | `127.0.0.1:5432`(只绑 localhost,公网不可达) |
| 数据持久化 | bind-mount `/media/heygo/Program/Docker/Containers/datahub/nous-center/postgre-data` → `/var/lib/postgresql/data` |
| 连接串 | `backend/.env` `DATABASE_URL=postgresql+asyncpg://nous_heygo:***@localhost:5432/nous_center` |
| 库/角色 | 库 `nous_center`,owner/role `nous_heygo`;同 server 另有 n8n 的 `nous_database`(init-nous.sh 注释印证"一个 server 多库") |
| compose | `/media/heygo/Program/Docker/Containers/datahub/nous-center/{docker-compose.yml,.env,init-nous.sh}` |
| schema 管理 | **无 alembic**;`backend/src/api/main.py:51` lifespan 内 `Base.metadata.create_all` + 幂等 `ALTER ... ADD COLUMN IF NOT EXISTS` 微迁移 |
| systemd 现状 | `infra/systemd/`:`nous-backend.service` / `nous-cloudflared.service`(PartOf backend) / `nous-status.service` / `nous-healthprobe.{service,timer}`;装机脚本 `install.sh` |
| 启动可见性 | systemd → journald,无聚合自检输出;`journalctl -u nous-backend` 只见零散 INFO |

### 1.3 磁盘约束(决定落盘策略)

- 系统盘 `/`(nvme0n1p1):937G,**空 584G**。
- 大盘 `/media/heygo/Program`(nvme2n1p1):1.9T,**95% 满,仅剩 110G**。
- 结论:**pg 数据落系统盘默认位置**(`/var/lib/postgresql/17/main`,584G 够;DB 只含元数据+结构化日志,模型/图都不在库里,体量小)。**备份落大盘**(nvme2)→ 数据与备份天然分处**两块物理盘**,单盘损坏不全失。

## 2. 目标设计

整栈四件套,分四个独立 PR。每个 PR 独立分支走 CI/CD。

### 2.1 总体形态

```
nous.target                         ← 总闸:一条命令拉起/停整栈
├── postgresql.service              ← 原生 pg17(系统自带 unit,开机自启)
├── nous-backend.service            ← Requires/After postgresql;启动打自检 banner
│   └── nous-cloudflared.service    ← PartOf backend(已有)
├── nous-status.service             ← 公开状态监控(已有)
└── nous-dbbackup.timer             ← 每日 pg_dump → 大盘,保留 N 天(新)

nousctl  up|down|restart|status|logs   ← 便捷 CLI:短命令 + 起完打印 banner
```

### 2.2 PR-1:原生 pg17 迁移 + 接入依赖链

**装 + 建库**
- `apt install postgresql-17`(若 apt 源无 17,加 PGDG 源 `apt.postgresql.org`)。Ubuntu 经 `postgresql-common` 自动建 cluster `17-main`;若 5432 被 Docker 占用,`pg_createcluster` 会**自动顺延到 5433**(迁移期双 pg 并存正需如此)。
- 建 role + 库,**保持与现状完全一致**(连接串不用改):
  ```sql
  CREATE ROLE nous_heygo LOGIN PASSWORD '...';
  CREATE DATABASE nous_center OWNER nous_heygo;
  ```

**迁移数据(零丢失:先 dump 验证再切)**
1. 装客户端工具(`postgresql-client-17` 带 `pg_dump`/`pg_restore`,主机当前无)。
2. 双 pg 并存期(docker@5432 + native@5433)跨端口迁:
   ```bash
   pg_dump -h 127.0.0.1 -p 5432 -U nous_heygo -Fc nous_center > /tmp/nous_center.dump
   pg_restore -h 127.0.0.1 -p 5433 -U nous_heygo -d nous_center --no-owner /tmp/nous_center.dump
   ```
3. **校验**:逐表 `count(*)` 比对 docker vs native 一致(脚本输出对照表),不一致即中止、不切换。
4. 校验过 → `docker compose down`(停 docker pg,释放 5432);保留 `postgre-data` 目录**冷备数日**再删。
5. native cluster 改回 **5432**(`pg_ctlcluster` / cluster `port` 配置 → `pg_renamecluster` 或直接改 `postgresql.conf` 的 `port=5432` + `pg_ctlcluster 17 main restart`)。
6. `backend/.env` 的 `DATABASE_URL` **不变**(仍 `localhost:5432/nous_center`)。

**接依赖链**
- `nous-backend.service` 已有 `After=postgresql.service` —— 现在它**真的指向存在的 unit**了。补 `Requires=postgresql.service`(没库不该起后端)。
- `postgresql.service` `systemctl enable`(开机自启)。
- 安全:native pg `listen_addresses='localhost'`(对齐原 compose 只绑 localhost,公网不可达);`pg_hba.conf` 本机 `scram-sha-256`。

**交付物**:迁移脚本 `infra/db/migrate-docker-to-native.sh`(幂等 + 校验 + 不通过即中止)、`infra/db/README.md` 操作记录。compose 目录原样保留作冷备(不删,标注"已弃用")。

### 2.3 PR-2:`nous.target` 总闸 + `nousctl` 便捷 CLI

**`nous.target`**(systemd 分组,grouping 的惯用做法):
```ini
[Unit]
Description=nous-center 全栈(DB + 后端 + 隧道 + 状态)
Wants=postgresql.service nous-backend.service nous-cloudflared.service nous-status.service
After=postgresql.service
[Install]
WantedBy=multi-user.target
```
- `systemctl start nous.target` → 按依赖序拉起全栈;`stop` 同理停全栈。
- 注:`systemctl start nous`(无后缀)默认找 `.service`,故规范命令是 `systemctl start nous.target`。短命令交给 `nousctl`。

**`nousctl`**(`infra/systemd/nousctl`,装到 `/usr/local/bin`):
```
nousctl up        # sudo systemctl start nous.target,然后打印自检 banner(读 journal)
nousctl down      # 停全栈
nousctl restart   # 重启全栈
nousctl status    # 各 unit active? + 端口 + 隧道 一屏
nousctl logs [u]  # journalctl -f(默认 backend)
```
- `nousctl up` 同时满足"命令短"+"看得到自检"两诉求(起完直接把 PR-3 的 banner 从 journal 拉出来显示)。

**装机**:`install.sh` 的 `UNIT_FILES`/`SERVICES` 增 `nous.target`;装 `nousctl` 到 `/usr/local/bin`;`enable nous.target`。

### 2.4 PR-3:启动自检 banner(对齐 PAPERCLIP)

后端 `lifespan`(`main.py`)wiring 完成后,**聚合一屏自检**打到 stdout(→ journald,`nousctl up`/`journalctl` 可见)。内容:

```
  ███  NOUS-CENTER  ███   v0.1.0

  Mode        production · serving frontend/dist        (dev 时:vite-dev :9999)
  Bind        0.0.0.0:8000
  Auth        admin gate ENABLED (cookie + bearer)      (ADMIN_PASSWORD 空时:DISABLED)
  Tunnel      api.iocrazy.com  ·  cloudflared active
  Database    connected · native pg17 @ localhost:5432/nous_center
  Schema      ensured (create_all + N micro-migrations)
  GPUs        3 · cuda:0 Pro6000 96G · cuda:1 3090 24G · cuda:2 3090 24G
  Resident    <常驻模型列表 / none>
  DB Backup   enabled · last 03:00 · keep 14d → /media/.../db-backups
  Logs        journalctl -u nous-backend -f
```

- 每行**真实探测**,非硬编码:DB 连通/版本来自启动期连接;GPU 来自现有 monitor/检测;隧道 active 来自探针或 `pg_isready` 同款轻探;常驻来自 supervisor;备份状态读最新 dump 文件 mtime。
- 探测**全 best-effort**:任一行探测失败显 `unknown`/`unavailable`,**绝不阻断启动**(banner 是观测,非门禁)。
- 失败/降级项**高亮**(无 emoji,用文字标 `[WARN]`)。
- 落点:抽 `backend/src/api/startup_banner.py`,`lifespan` 末尾调用一次。

### 2.5 PR-4:自动 DB 备份

- `nous-dbbackup.service`(oneshot)+ `nous-dbbackup.timer`(每日,如 `OnCalendar=*-*-* 03:00`,`Persistent=true` 补跑错过的)。
- 脚本 `infra/db/backup.sh`:`pg_dump -Fc nous_center` → `/media/heygo/Program/.../db-backups/nous_center-YYYYMMDD-HHMM.dump`(**大盘 nvme2,与数据盘 nvme0 分离**);保留期 `NOUS_DB_BACKUP_KEEP_DAYS`(默认 14)滚动删旧。
- 备份后**校验 dump 非空 + 可 `pg_restore --list` 读**,否则告警(写 journal,被 healthprobe/status 可选感知)。
- banner 的 `DB Backup` 行读此目录最新文件展示。
- 装机:`install.sh` 增 timer。

## 3. 迁移 / 回滚

- **迁移**:PR-1 脚本先 dump→restore→校验,不通过即中止、Docker pg 原封不动。切换仅在校验通过后。
- **回滚**:Docker compose 目录 + `postgre-data` 数日内不删 → 出问题 `docker compose up -d` + `.env` 指回即恢复(数据是 bind-mount 目录,未被触碰)。
- n8n:其 pg 在另一容器、库 `nous_database`,本 spec **不动**;Docker Desktop 仍可保留给 n8n(或后续单独迁原生 dockerd,不在本 spec 范围)。

## 4. 非目标

- 不引入 alembic(维持 create_all + 微迁移现状)。
- 不上 k8s / postgres-operator / 高可用副本(单机自托管,过度设计)。
- 不迁 n8n、不卸 Docker Desktop(只把 **nous 的数据层**拔到原生 pg)。
- 不改 `DATABASE_URL` 形态、不改业务表结构。

## 5. 验收

- [ ] 冷重启机器,**不做任何手动操作**,`nousctl status` 显示 pg/backend/隧道全 active,公网 `api.iocrazy.com` 通。
- [ ] `nousctl up` 一条命令拉起全栈并打印完整自检 banner。
- [ ] 迁移后逐表 count 与 Docker pg 一致;真机 e2e(建 key → 跑工作流真出图 → 用量入库)通。
- [ ] 触发一次 `nous-dbbackup.service`,大盘生成可 `pg_restore --list` 的 dump;banner 反映最新备份时间。
- [ ] `docker ps` 不再有 `nous-center-postgres-1`(或已 down);5432 由原生 pg 监听(`ss -ltnp`)。

## 6. PR 拆分

| PR | 范围 | 验证 |
|---|---|---|
| PR-1 | 原生 pg17 装+迁+校验+接依赖链 | 逐表 count 一致 + 后端连原生库 e2e |
| PR-2 | `nous.target` + `nousctl` + install.sh | `nousctl up/down/status`、冷重启自启 |
| PR-3 | 启动自检 banner | journal 见完整面板、降级项不阻断 |
| PR-4 | 自动 DB 备份 timer + 脚本 | dump 生成可读、滚动保留、banner 反映 |
