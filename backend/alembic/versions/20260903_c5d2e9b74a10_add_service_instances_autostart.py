"""add service_instances.autostart

服务级「开机启动」开关。默认 false —— 开机只加载 ① resident 模型
② autostart=true 的服务引用到的模型,其余按需 get_or_load。

Revision ID: c5d2e9b74a10
Revises: a3f1c07d5b2e
Create Date: 2026-09-03 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c5d2e9b74a10'
down_revision: Union[str, None] = 'a3f1c07d5b2e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # server_default 是**存量行**用的(NOT NULL 加列必须给默认值);模型侧同样带
    # server_default=false,两边一致 → `alembic check` 零 diff。
    op.add_column(
        'service_instances',
        sa.Column('autostart', sa.Boolean(), nullable=False, server_default=sa.text('false')),
    )


def downgrade() -> None:
    op.drop_column('service_instances', 'autostart')
