"""model-backed 服务创建时按引擎 type 派生 category(修 API key 页调用示例端点分发)。

此前只工作流服务自动探测 category → model 服务恒 None → 前端给 embedding/tts key
的调用示例落到 chat 端点(错)。源码 + 注册表口径检查(CI 安全)。
"""
from __future__ import annotations

import pathlib


def test_create_instance_derives_category_from_engine_type():
    src = (pathlib.Path(__file__).parent.parent / "src/api/routes/instances.py").read_text()
    assert "load_model_configs()" in src, "create_instance 未按引擎配置派生 category"
    assert "category=category" in src, "ServiceInstance 未写入派生的 category"


def test_engine_types_cover_embedding():
    """注册表里 embedding 引擎 type 必须是 'embedding'(前端 endpointsFor 据此给 /v1/embeddings)。"""
    import yaml

    cfg = yaml.safe_load(
        (pathlib.Path(__file__).parent.parent / "configs/models.yaml").read_text())
    by_id = {m["id"]: m for m in cfg["models"]}
    assert by_id["qwen3_embedding_4b"]["type"] == "embedding"
    assert by_id["qwen3_embedding_8b"]["type"] == "embedding"
