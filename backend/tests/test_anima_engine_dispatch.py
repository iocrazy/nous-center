"""PR-anima-6:engine 集成路由测(CI 跑;真 forward 走 standalone smoke)。

验证:
  - pipeline_class="AnimaPipeline" 走 _get_or_load_anima_adapter(不是 modular)
  - runner_process._build_request adapter_arch="anima" → pipeline_class="AnimaPipeline"
  - 默认 adapter_arch=flux2 → "Flux2KleinPipeline"(回归不变)
"""
from __future__ import annotations

# grep-style 验源代码 — conftest mock torch + AnimaImageBackend 依赖真 torch nn.Module,
# CI 真跑路由集成需 GPU + 真 anima 权重(留 tests/manual/smoke_anima_pr*.py)。


def test_image_anima_backend_module_exists():
    """新增 src/services/inference/image_anima.py + AnimaImageBackend class。"""
    import pathlib  # noqa: PLC0415

    p = pathlib.Path(__file__).parent.parent / "src/services/inference/image_anima.py"
    assert p.exists()
    src = p.read_text()
    for sym in [
        "class AnimaImageBackend",
        "def _ensure_pipe",
        "async def infer",
        "AnimaPipeline",  # 引用了 arch_anima 的 pipeline
        "NOUS_ANIMA_QWEN_TOKENIZER",
        "image/png",
        '"engine": "anima"',
    ]:
        assert sym in src, f"image_anima.py missing {sym!r}"


def test_model_manager_has_anima_dispatch():
    """model_manager.get_or_load_image_adapter pipeline_class=AnimaPipeline → _get_or_load_anima_adapter。"""
    import pathlib  # noqa: PLC0415

    p = pathlib.Path(__file__).parent.parent / "src/services/model_manager.py"
    src = p.read_text()
    for sym in [
        "_get_or_load_anima_adapter",
        '"AnimaPipeline"',
        "AnimaImageBackend",
        "PR-anima-6",
    ]:
        assert sym in src, f"model_manager.py missing PR-anima-6 hook {sym!r}"


def test_runner_process_arch_to_pipeline_class():
    """runner_process._build_request:adapter_arch='anima' → pipeline_class='AnimaPipeline'。"""
    import pathlib  # noqa: PLC0415

    p = pathlib.Path(__file__).parent.parent / "src/runner/runner_process.py"
    src = p.read_text()
    for sym in [
        'arch = unet_spec.get("adapter_arch") or "flux2"',
        'pipeline_class = "AnimaPipeline" if arch == "anima" else "Flux2KleinPipeline"',
    ]:
        assert sym in src, f"runner_process.py missing arch routing {sym!r}"


def test_node_yaml_adapter_arch_includes_anima():
    """node.yaml flux2_load_diffusion_model 的 adapter_arch widget 暴露 anima 选项。"""
    import pathlib  # noqa: PLC0415

    p = pathlib.Path(__file__).parent.parent / "nodes/flux2-components/node.yaml"
    src = p.read_text()
    assert "value: anima" in src
    assert "Anima 2B 自定义 DiT" in src or "AnimaPipeline" in src


def test_get_or_load_image_adapter_dispatch_branch_source():
    """grep-style 验 model_manager.get_or_load_image_adapter 含 anima 分支(不真跑 — 真跑需 GPU)。"""
    import pathlib  # noqa: PLC0415

    p = pathlib.Path(__file__).parent.parent / "src/services/model_manager.py"
    src = p.read_text()
    # 必须先检 anima,否则 fall through 到 modular(顺序敏感)
    anima_idx = src.find('if pipeline_class == "AnimaPipeline":')
    modular_idx = src.find("_get_or_load_modular_adapter(\n")
    assert anima_idx > 0, "缺 AnimaPipeline 分支"
    assert modular_idx > anima_idx, "anima 分支必须在 modular fallback 之前"
