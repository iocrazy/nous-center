"""add memory_entries FTS GIN index

Revision ID: a3f1c07d5b2e
Revises: 91fbe4eca0b3
Create Date: 2026-09-02 10:12:03.415920
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3f1c07d5b2e'
down_revision: Union[str, None] = '91fbe4eca0b3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 表达式必须与 pg_provider.py::prefetch 的查询侧逐字一致(含 'simple'),
    # 否则 planner 匹配不上,索引形同虚设。
    op.create_index(
        'idx_mem_content_fts',
        'memory_entries',
        [sa.text("to_tsvector('simple', content)")],
        unique=False,
        postgresql_using='gin',
    )


def downgrade() -> None:
    op.drop_index('idx_mem_content_fts', table_name='memory_entries')
