"""src.services.runner_models —— 聚合 runner 上报的已加载 adapter 快照(Bug 3)。"""
from __future__ import annotations

from types import SimpleNamespace

from src.services.runner_models import aggregate_runner_loaded, loaded_source_stems


class _Sup:
    def __init__(self, group_id, loaded_models):
        self.group_id = group_id
        self.loaded_models = loaded_models


def _img(mid, files):
    return {"model_id": mid, "model_type": "image", "gpu_index": 1,
            "gpu_indices": [1], "vram_mb": 19000, "pipeline_class": "Flux2KleinPipeline",
            "source_files": files, "last_used_ago_sec": 1.0}


def test_aggregate_tags_group_id_and_merges_supervisors():
    state = SimpleNamespace(runner_supervisors=[
        _Sup("image", [_img("image:foo:1", ["/m/a.safetensors"])]),
        _Sup("tts", [{"model_id": "qwen-tts", "model_type": "tts", "gpu_index": 0,
                      "source_files": [], "vram_mb": 5000}]),
    ])
    out = aggregate_runner_loaded(state)
    assert {e["model_id"] for e in out} == {"image:foo:1", "qwen-tts"}
    assert next(e for e in out if e["model_id"] == "image:foo:1")["group_id"] == "image"
    assert next(e for e in out if e["model_id"] == "qwen-tts")["group_id"] == "tts"


def test_aggregate_robust_to_missing_and_mock_state():
    # 无 runner_supervisors / model_manager 属性 → 空列表,不抛
    assert aggregate_runner_loaded(SimpleNamespace()) == []
    # model_manager.loaded_models_snapshot 抛异常 → 跳过,不拖垮聚合
    bad = SimpleNamespace(loaded_models_snapshot=lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert aggregate_runner_loaded(SimpleNamespace(model_manager=bad)) == []


def test_loaded_source_stems_normalizes_basenames():
    state = SimpleNamespace(runner_supervisors=[
        _Sup("image", [_img("image:foo:1",
                            ["/models/Flux2-Klein-9B.safetensors", "/m/qwen3.safetensors"])]),
    ])
    stems = loaded_source_stems(state)
    # 去目录 + 去扩展名 + 小写
    assert "flux2-klein-9b" in stems
    assert "qwen3" in stems
