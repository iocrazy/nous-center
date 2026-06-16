"""每模型运行时覆盖(resident / gpu / vram_budget)的 Postgres 表。

数据加载统一(2026-06-16,用户拍「拆数据表」):运行时覆盖从 gitignore 的
runtime_overrides.json 文件迁到关系库 —— 拆成正经 typed 列(非 jsonb 文件搬家),
与服务/key/用量同库一处。静态定义仍在 models.yaml;此表只存"每机运行时调整"。

列语义:NULL = 未覆盖(回退 models.yaml);非 NULL = 显式覆盖(含 resident=False、gpu=0
这类有效值,故用 nullable 区分"没设"与"设成 False/0")。vram_budget 拆成 mode + value 两列
(mode=auto 时 value 可 NULL)。
"""
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String

from src.models.database import Base


class ModelRuntimeOverride(Base):
    __tablename__ = "model_runtime_overrides"

    model_id = Column(String(200), primary_key=True)
    resident = Column(Boolean, nullable=True)
    gpu = Column(Integer, nullable=True)
    vram_budget_mode = Column(String(20), nullable=True)   # auto | percent | absolute
    vram_budget_value = Column(Float, nullable=True)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def to_overrides(self) -> dict:
        """→ 与旧 overlay 同形状的 dict:{resident?, gpu?, vram_budget?}(只含已设字段)。
        消费方(load_model_configs / registry / resolve_vram_utilization)契约不变,只换存储。"""
        out: dict = {}
        if self.resident is not None:
            out["resident"] = self.resident
        if self.gpu is not None:
            out["gpu"] = self.gpu
        if self.vram_budget_mode is not None:
            vb: dict = {"mode": self.vram_budget_mode}
            if self.vram_budget_value is not None:
                vb["value"] = self.vram_budget_value
            out["vram_budget"] = vb
        return out
