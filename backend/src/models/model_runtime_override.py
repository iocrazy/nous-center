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
from sqlalchemy.dialects.postgresql import JSONB

from src.models.database import Base


class ModelRuntimeOverride(Base):
    __tablename__ = "model_runtime_overrides"

    model_id = Column(String(200), primary_key=True)
    resident = Column(Boolean, nullable=True)
    gpu = Column(Integer, nullable=True)
    # GPU 组(张量并行):`[0, 2]` = 这俩卡当一个单元用,tp=2。NULL = 未覆盖(走单卡 gpu /
    # 自动)。与 gpu 并存且**优先**:给了 gpus 就以它为准,gpu 只当单卡钉卡用。
    # 列而非 typed 多列:组大小可变(2/4 卡),JSONB 是唯一不用改 schema 就能扩的形状。
    gpus = Column(JSONB, nullable=True)
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
        if self.gpus:
            out["gpus"] = [int(i) for i in self.gpus]
        if self.vram_budget_mode is not None:
            vb: dict = {"mode": self.vram_budget_mode}
            if self.vram_budget_value is not None:
                vb["value"] = self.vram_budget_value
            out["vram_budget"] = vb
        return out
