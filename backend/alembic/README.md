# Alembic 迁移

版本化 schema 迁移。**当前为「已引入、未接管启动」阶段** —— `lifespan` 仍用
`Base.metadata.create_all` + `_MICRO_MIGRATIONS` 建表(见 `src/api/main.py`)。alembic 作为
未来 schema 变更的正规工具**并行存在**,待你在生产 stamp 后再切换启动路径。

## 为什么 baseline 是安全的

baseline 迁移(`versions/*_baseline.py`)由 autogenerate 对空库生成,`test_alembic_baseline.py`
每次 CI 验证「fresh DB `upgrade head` 后与 models 零 diff」。且引入前已核对:`lifespan` 里所有
手写微迁移(`_MICRO_MIGRATIONS`)都已回填进 models → `create_all(models)` == 生产 schema →
baseline == 生产 schema。

## 生产切换步骤(需你在 prod 机器上跑一次)

你的生产库是既有的、已有数据的,**不能**对它 `upgrade head`(会重复建表)。正确姿势是
**stamp**:告诉 alembic「这库已在 baseline 版本」,不执行 DDL。

```bash
cd backend
# 1)(强烈建议)先核对 baseline 真的等于你 prod 的实际 schema:
#    导出 prod schema,和 fresh upgrade 的结果 diff。若一致 → 放心 stamp。
pg_dump --schema-only "$DATABASE_URL" > /tmp/prod_schema.sql

# 2) 把 prod 标记为已在 baseline(不建表、不改数据):
DATABASE_URL="<prod url>" uv run alembic stamp head

# 3) 之后所有 schema 变更:改 model → 生成迁移 → 审阅 → 提交
DATABASE_URL="<prod url>" uv run alembic revision --autogenerate -m "描述"
DATABASE_URL="<prod url>" uv run alembic upgrade head
```

## 日常:加一个 schema 变更

```bash
cd backend
# 改完 src/models/*.py 后:
DATABASE_URL="sqlite+aiosqlite:///./_dev.db" uv run alembic upgrade head        # 先把本地库拉到 head
DATABASE_URL="sqlite+aiosqlite:///./_dev.db" uv run alembic revision --autogenerate -m "add xxx"
# 审阅生成的 versions/*.py(autogenerate 偶尔漏 import 如 Text,或误判 type 变更),再提交。
```

## 注意

- `sqlalchemy.url` 不写在 `alembic.ini`,由 `env.py` 从 `src.config.get_settings()` 注入。
- `env.py` 显式 import 全部 model 模块填满 `Base.metadata`;新增 model 文件记得在 env.py 补 import,
  否则 autogenerate 会漏表(甚至生成 drop_table)。
- autogenerate 对 `JSONB(astext_type=Text())` 会漏 `from sqlalchemy import Text`,生成后手动补。
