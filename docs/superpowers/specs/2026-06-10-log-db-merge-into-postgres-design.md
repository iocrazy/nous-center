# 日志库并入 PostgreSQL —— 只保留一个数据库

- 日期: 2026-06-10
- 状态: 设计
- 触发: 用户「只保留一个数据库吧」(当前 = 主库 PostgreSQL + 独立 SQLite `log_db`)
- 方向(用户拍): **留 PG,日志并入**(非 SQLite-only —— 多 runner 子进程并发写主库,
  SQLite 单写者会 `database is locked`,拿掉 PG 是退步)

## 1. 现状:为什么日志当初是独立 SQLite

`src/services/log_db.py` 是一个独立 SQLite(`backend/data/logs.db`,WAL + `busy_timeout=5s`),
存 4 类日志,与主库(`postgresql+asyncpg://.../nous_center`)完全解耦。解耦带来 4 个性质:

1. **写不阻塞请求**:request/audit 经 `loop.run_in_executor` 丢线程池;app 日志经
   `DbLogHandler` 后台线程缓冲(1s / 50 条 flush);frontend 经路由同步写。
2. **写失败不拖垮请求**:middleware / handler 全 `except: pass` 静默吞。
3. **PG 挂了还能看日志**(独立连接,不依赖主库健康)。
4. **高频日志写不抢主库连接池**。

并入 PG 后**性质 1/2/4 必须用『异步队列 + 单消费者批量写』保住**;性质 3 在**单机单管理员**
部署下是空头支票(PG 挂 = 整个 app 挂,这时看 journald / `dev-serve.sh` 的
`backend/logs/`,不是这个库),**丢得起**,这是本次唯一净损失。

## 2. 现状盘点(代码坐标)

| 关注点 | 现状 |
|---|---|
| 4 表 schema | `log_db.py:9-53`(request/app/frontend/audit) |
| 时间戳 | CST 字符串 `"%Y-%m-%d %H:%M:%S"`(`_now()`,`log_db.py:112`) |
| 写:request | `middleware.py:53-62` `run_in_executor(insert_request_log)` |
| 写:audit | `middleware.py:98-108` `run_in_executor(insert_audit_log)` |
| 写:app | `log_collector.py` `DbLogHandler`(后台线程缓冲 → `insert_app_log`) |
| 写:frontend | `routes/logs.py:54` 路由内同步 `insert_frontend_log` |
| 读 | `routes/logs.py` 4 个 GET → `query_logs`(since/search/level/type/method/status + 分页) |
| 清理 | `main.py:480-492` 每 1h `to_thread(cleanup_logs)`(7天 / 10万行/表) |
| 初始化 | `main.py:127-129` `init_log_db()` |
| handler 安装 | `main.py:132-151` `DbLogHandler` + stdout |
| 测试 | `test_log_db.py` / `test_log_collector.py` / `test_logs_api.py` / `test_middleware_logs.py` / `conftest_logs.py`(静默 DbLogHandler 防写真 logs.db) |

## 3. 目标设计

### 3.1 存储:4 个 SQLAlchemy model 进主库

新建 `src/models/log_entry.py`:`RequestLog` / `AppLog` / `FrontendLog` / `AuditLog`,
继承 `database.Base`,列与现 SQLite schema **逐字对应**。

- **时间戳保留 CST 字符串列**(`String(32)`,沿用 `_now()` 格式):零前端契约风险
  (前端解析 `"YYYY-MM-DD HH:MM:SS"`)、`since` 归一化逻辑逐字移植(定宽零填充 → 字典序比较
  正确)、排序本就靠 `id DESC` 非时间戳。日志是 ephemeral 观测数据,字符串列足够,不引入
  时区转换的新风险面。
- `id` BigInteger 自增主键;`timestamp` 上加索引(对齐现 SQLite 索引)。
- model 须在 `main.py` 建表块(现 60-72 行 import 区)被 import,`Base.metadata.create_all`
  才会建表。

### 3.2 写路径:异步队列 + 单消费者批量写(保住性质 1/2/4)

新建 `src/services/log_store.py`,核心是一个模块级 `LogWriter`:

- 启动时(lifespan)`start(loop)`:记录 event loop 引用,建 `asyncio.Queue(maxsize=10000)`,
  起单个消费者 task。
- `enqueue(kind, fields)` **同步、线程安全、永不阻塞/抛**:
  - 在 event loop 线程(middleware / 路由)→ 直接 `queue.put_nowait`。
  - 跨线程(`DbLogHandler` flush 线程)→ `loop.call_soon_threadsafe(queue.put_nowait, item)`。
  - 队列满 → 丢弃 + `dropped` 计数器自增(永不阻塞调用方)。
  - writer 未启动(极早期/测试无 lifespan)→ 丢弃。
- 消费者 coroutine:批量取(攒到 N 条或短超时),按 kind 分组,**一个 async session 一次
  flush bulk insert + commit**;异常 → 打到 stderr(**不回灌 logging**,防递归)后继续。
  → 单写者天然不抢主库连接池(同一时刻最多占一个 session);批量比逐请求 insert 压力更小。

这套**比现有 3 套机制(executor / 线程缓冲 / 同步路由写)更统一**:所有写都收敛成
`enqueue` + 单消费者。

落点改写:
- `middleware.py`:`run_in_executor(insert_*)` → `log_store.enqueue("request"/"audit", {...})`(仍裹 `try/except: pass` 双保险)。
- `log_collector.py`:`DbLogHandler.emit` → `log_store.enqueue("app", {...})`(跨线程 safe handoff);
  去掉自身缓冲/线程(writer 已批量),退化成薄适配器。
- `routes/logs.py:54` frontend 写 → `log_store.enqueue("frontend", {...})`。

### 3.3 读路径:async SQLAlchemy

`log_store.query_logs(session, table, ...)` async 版,逐项移植 `log_db.query_logs` 的过滤
(since 归一化 / search LIKE / level 最低级 / type / method / status 的 `Nxx` 前缀匹配)+
`func.count` + `ORDER BY id DESC LIMIT/OFFSET`,返回 `{total, items}`,items 键与现在一致。
`routes/logs.py` 4 个 GET 改 async + `Depends(get_async_session)`。

### 3.4 清理:async DELETE

`log_store.cleanup_logs(session, max_age_days=7, max_rows=100_000)` async:按 `timestamp < cutoff`
删 + 行数上限(取第 max_rows 个 `id` 阈值,`DELETE WHERE id < threshold`,避免 PG 大 `NOT IN`
子查询慢)。`main.py` 清理 loop 改为直接 await(不再 `to_thread`,已是 async DB)。

### 3.5 旧数据 & 文件

`backend/data/logs.db`(~35MB)**不迁移**:观测日志 7 天滚动,迁移无价值。cutover 后删
`log_db.py` + 文件由用户手动删(spec 注明)。

## 4. PR 切分

> 依赖:PR-1 引入新代码不接线(无行为变化,可独立测);PR-2 切换 + 删 SQLite。

### PR-1 PG 日志存储(新增,不接线)
- `src/models/log_entry.py`:4 个 model。
- `src/services/log_store.py`:`LogWriter`(队列 + 消费者 + enqueue)/ async `query_logs` /
  async `cleanup_logs`。
- 单测:enqueue 线程安全 / 消费者批量落库 / query 各过滤 / cleanup age+rowcap / 队列满丢弃计数。
- 不改 middleware / routes / main.py → 行为零变化。

### PR-2 切换并删除 SQLite log_db
- `middleware.py` / `log_collector.py` / `routes/logs.py` / `main.py` 全切到 `log_store`。
- lifespan:删 `init_log_db`,加 `log_store` writer `start`/`stop`(随 bg_tasks 优雅收尾);
  清理 loop 改 await async cleanup。
- 删 `src/services/log_db.py`。
- 测试:重写 4 个 log 测试到 async store(跑测试 DB = conftest 的 SQLite,`Base` 表自动建);
  删 `conftest_logs.py` 特判(不再写真 logs.db)。
- `CLAUDE.md`:更新「log_db 独立 SQLite」段 → 「日志表在主 PG 库,经 log_store 异步队列写」。
- 用户手动删 `backend/data/logs.db*`。

## 5. 验证

- CI:ruff + 4 个 log 测试(async store)+ middleware/路由集成测试绿。
- 真机:`dev-serve.sh` 起后端(PG)→ 打几个请求/触发一次 admin 变更/前端报错 →
  `/api/v1/logs/{requests,app,audit,frontend}` 各拉到对应记录 → LogsOverlay 四类正常 →
  确认无 `backend/data/logs.db` 新生成。压一批请求确认写不阻塞、丢弃计数为 0。

## 6. 非目标 / follow-up

- 不动结构化日志的**字段/UI 契约**(纯换存储后端)。
- 时间戳列若日后要做时区/范围聚合再升 `timestamptz`(本次保字符串)。
- 队列满丢弃计数可日后暴露成监控指标(现仅内部计数 + stderr)。

关联 [[project_run_history_artifacts]]
