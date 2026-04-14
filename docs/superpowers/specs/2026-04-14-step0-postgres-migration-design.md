# Step 0 · SQLite → PostgreSQL 迁移

## Context

nous-center 既有代码默认连 PG（`config.py` 的 `DATABASE_URL` 默认值指向
`mindcenter:mindcenter@localhost:5432/mindcenter`），但 `backend/.env` 被
临时覆盖成 SQLite（`sqlite+aiosqlite:///data/nous.db`）。结果两边都有数据但
不同步：PG 里是 3 月的测试数据，SQLite 里是 4 月的近期开发数据。

同时一个遗留的容器 `postgres` 跑着 PG 17、占着 5432 端口、挂着一个 docker
volume `postgres_data`（里面是 mindcenter 库）。

后续 5 个升级步骤（用量查询、Context Cache、Responses API 等）全部依赖
统一的持久化层，继续用 SQLite 会撑不住 Responses API 的 JSON 存储规模，
也享受不到 PG 的 jsonb、窗口函数、分区表。因此先切 PG。

目标状态：
- 单一 PG 17 实例，容器化，只绑 `127.0.0.1:5432`
- 数据卷 `bind mount` 到项目目录下，便于备份
- 所有历史数据（mindcenter + SQLite）合并进一个 `nous_center` 数据库
- 旧 mindcenter 容器保留 stopped，作为一周回滚窗口

## 决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| DB 引擎 | PostgreSQL 17-alpine | 用户指定 17；alpine 镜像小、补丁及时 |
| 部署方式 | docker-compose（独立目录） | 对齐现有 datahub 惯例；backend 留宿主机 |
| 数据卷 | `bind mount` 到 `./postgre-data` | 便于可视化、备份、迁移；非 named volume |
| 端口暴露 | `127.0.0.1:5432:5432` | backend 在同机；公网不可达 |
| nous-center 数据库 | 单库 `nous_center` | `POSTGRES_DB` 首次启动自动建 |
| 数据合并策略 | mindcenter dump + SQLite 迁移脚本 | 两边时代不重叠，ID 不冲突 |
| SQLite schema drift | 脚本剥离 `preset_id` 老列 | `service_instances` 已改用 `source_type/source_id` |
| FK 顺序 | `SET session_replication_role = replica` | pg_dump 按字母序插表，FK 会乱；迁移时关掉校验 |
| Redis | 推迟到 v2 | 单 worker 场景内存计数器够用 |

## 架构

```
┌──────────────────────── 宿主机 ────────────────────────┐
│                                                        │
│  nous-center backend (uvicorn + vLLM)    nous-center   │
│   .venv/bin/python  port 8000             frontend     │
│         │                                  (vite)       │
│         │ DATABASE_URL                                  │
│         ▼                                               │
│  ┌─────────────────────┐        (n8n/mindcenter         │
│  │ docker-compose      │         volumes 保留但 stopped) │
│  │   postgres:17-alpine│                                │
│  │   127.0.0.1:5432    │                                │
│  │   ./postgre-data    │                                │
│  └─────────────────────┘                                │
└────────────────────────────────────────────────────────┘
```

## 关键文件

| 路径 | 内容 |
|------|------|
| `/media/heygo/Program/Docker/Containers/datahub/nous-center/docker-compose.yml` | PG 17 service，`127.0.0.1:5432`，`./postgre-data` 挂载 |
| `/media/heygo/Program/Docker/Containers/datahub/nous-center/.env` | `DATABASE_URL`、PG 凭证 |
| `backend/.env` | `DATABASE_URL=postgresql+asyncpg://nous_heygo:Heygo01!@localhost:5432/nous_center` |
| `backend/scripts/migrate_sqlite_to_pg.py` | 幂等迁移脚本（bool/datetime/JSON 类型转换 + FK disable + ON CONFLICT DO NOTHING） |
| `backend/data/archive/nous.db` | 原 SQLite 归档，一周稳定后删除 |

## 执行流程（已完成）

1. `pg_dump` mindcenter 数据（`--data-only --inserts --column-inserts`）
2. 剥离 PG 17 特有的 `\restrict` / `\unrestrict` 指令
3. 停掉 `postgres` / `redis` 旧容器（`docker stop`）
4. `docker compose up -d` 启动新 PG 17
5. 后端通过 `Base.metadata.create_all` 建 schema（11 张表，含新 `llm_usage`）
6. Python 脚本剥离 dump 里的 `preset_id` 老列
7. `SET session_replication_role = replica` + 恢复 dump
8. 运行 `migrate_sqlite_to_pg.py` 把 SQLite 的 workflows (5) 和 service_instances (1) 合并进 PG
9. `backend/.env` 切到 PG
10. 重启 backend，`curl /api/v1/instances` 返回合并后的数据

## 验证

```bash
# PG 可达
docker compose -f /media/heygo/Program/Docker/Containers/datahub/nous-center/docker-compose.yml exec postgres psql -U nous_heygo -d nous_center -c "SELECT COUNT(*) FROM service_instances"
# 预期: 5

# backend 连接正常
curl -s --noproxy '*' http://localhost:8000/api/v1/instances | jq length
# 预期: 5

# workflows 9 条 (4 老 + 5 新)
curl -s --noproxy '*' http://localhost:8000/api/v1/workflows | jq length
# 预期: 9
```

## 回滚方案（一周内）

如果发现 PG 有问题：
1. `backend/.env` 恢复 `DATABASE_URL=sqlite+aiosqlite:///data/archive/nous.db`
2. `mv backend/data/archive/*.db backend/data/`
3. `cd /media/heygo/Program/Docker/Containers/datahub/nous-center && docker compose down`
4. `docker start postgres redis`（旧 mindcenter 容器还在，卷还在）

一周后确认稳定：
- `docker rm -v postgres redis`（释放 mindcenter 容器 + volume）
- `rm -rf backend/data/archive/`（丢弃 SQLite 归档）
