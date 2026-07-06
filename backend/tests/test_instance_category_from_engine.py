"""model-backed 服务(v3 /services/register-model)按引擎 type 派生 category
(修 API key 页调用示例端点分发)。双轨收敛(#3)后 legacy instances.py 已删,
本测试改查 v3 register_model_service 源码 + 注册表口径(CI 安全,无 torch/DB)。"""
from __future__ import annotations

import pathlib


def test_register_model_derives_category_from_engine_type():
    src = (pathlib.Path(__file__).parent.parent / "src/api/routes/services.py").read_text()
    assert "def register_model_service" in src, "缺 v3 建模型服务端点"
    assert "load_model_configs()" in src, "register_model 未按引擎配置派生 category"
    assert "category=category" in src, "ServiceInstance 未写入派生的 category"


def test_engine_types_cover_embedding():
    """注册表里 embedding 引擎 type 必须是 'embedding'(前端 endpointsFor 据此给 /v1/embeddings)。"""
    from src.config import collect_model_entries

    root = pathlib.Path(__file__).parent.parent
    by_id = {m["id"]: m for m in collect_model_entries(root / "configs/models.yaml")}
    assert by_id["qwen3_embedding_4b"]["type"] == "embedding"
    assert by_id["qwen3_embedding_8b"]["type"] == "embedding"
