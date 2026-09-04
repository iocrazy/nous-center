"""add model_runtime_overrides.gpus

模型级 GPU 组(张量并行):`gpus = [0, 2]` 表示这俩卡当一个 48G 单元用,
tp=len(gpus)。NULL = 未覆盖(回退单卡 `gpu` / 自动选卡)。

JSONB 而非多个 typed 列:组大小可变(2 卡 / 4 卡),用列表是唯一不用改 schema
就能扩的形状。

Revision ID: d7a4b1e6c093
Revises: c5d2e9b74a10
Create Date: 2026-09-03 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'd7a4b1e6c093'
down_revision: Union[str, None] = 'c5d2e9b74a10'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'model_runtime_overrides',
        sa.Column('gpus', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('model_runtime_overrides', 'gpus')
