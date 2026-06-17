"""状态页采样表(2026-06-17,status 页 v1)。

后台采样器每 60s 给每个组件(后端/DB/LLM/embedding/image·tts runner/GPU)写一行
当前状态,供 status 页画「过去 7 天 uptime 条」。当前实时状态由端点现算(不读本表),
本表只供历史/uptime 聚合。8 天外的行由采样器定期清理。
"""
from datetime import datetime, timezone

from sqlalchemy import BigInteger, DateTime, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.models.database import Base

# BIGINT 在 SQLite(测试)不自增,只有 INTEGER PRIMARY KEY 自增 → with_variant 兜底。
_AutoId = BigInteger().with_variant(Integer, "sqlite")


class StatusSample(Base):
    __tablename__ = "status_samples"

    id: Mapped[int] = mapped_column(_AutoId, primary_key=True, autoincrement=True)
    component: Mapped[str] = mapped_column(String(40), nullable=False)
    # operational | degraded | down
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (Index("idx_status_component_ts", "component", "ts"),)
