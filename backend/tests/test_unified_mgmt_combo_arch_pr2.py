"""统一模型管理收尾 PR-2:已加载 combo 卸载 + diffusion_models 组件 arch 推断。

combo 卸载:引擎库「已加载」卡卸载按钮 → /loaded-adapter/unload(by model_id)→ runner unload_model。
arch 推断:diffusion_models 单文件按文件名/路径推 arch,引擎库预热传给 /component/preload(避免默认 flux2 错配)。
CI 安全:路由/catalog 源码 + 纯函数。跨进程卸载真机另验。
"""
from __future__ import annotations

import pathlib

from src.services.engine_catalog import _infer_arch

_SRC = pathlib.Path(__file__).parent.parent / "src"


def test_loaded_adapter_unload_route_before_param_route():
    """`/loaded-adapter/unload` 必须在 `/{name}/unload` **之前**(否则参数路由抢先匹配致 404)。"""
    src = (_SRC / "api/routes/engines.py").read_text()
    assert '"/loaded-adapter/unload"' in src
    assert "sup.client.unload_model(" in src
    assert src.index('"/loaded-adapter/unload"') < src.index('"/{name}/unload"')


def test_unload_clears_overrides():
    """ModularImageBackend.unload() 必须清 *_override —— 否则 adapter 持着 ~20GB 权重,
    _release_combo_components 置池 module=None 后这条引用还在 → combo 卸载只释放包装层(真机验过的根因)。"""
    src = (_SRC / "services/inference/image_modular.py").read_text()
    u = src[src.index("def unload(self)"):src.index("def _ensure_pipe(self)")]
    assert "self._transformer_override = None" in u
    assert "self._text_encoder_override = None" in u
    assert "self._vae_override = None" in u


def test_infer_arch_heuristic():
    assert _infer_arch("Z-Image-base.safetensors", "/m/diffusion_models/Z-Image-base.safetensors") == "z-image"
    assert _infer_arch("z_image_turbo_bf16.safetensors", "/m/z_image_turbo_bf16.safetensors") == "z-image"
    assert _infer_arch("anima-base-v1.0.safetensors", "/m/diffusion_models/anima/anima-base-v1.0.safetensors") == "anima"
    assert _infer_arch("Flux2-Klein-9B-True-v2-bf16.safetensors", "/m/flux/Flux2-Klein.safetensors") == "flux2"


def test_catalog_sets_arch_on_diffusion_models(monkeypatch):
    """component_catalog_entries 给 diffusion_models 条目带推断 arch;clip/vae/lora 不带(None)。"""
    import src.services.engine_catalog as EC

    def _fake_scan(role):
        if role == "diffusion_models":
            return [{"filename": "Z-Image-base.safetensors",
                     "abs_path": "/m/diffusion_models/Z-Image-base.safetensors", "size_mb": 12000}]
        if role == "clip":
            return [{"filename": "qwen_3_4b.safetensors",
                     "abs_path": "/m/text_encoders/qwen_3_4b.safetensors", "size_mb": 8000}]
        return []

    monkeypatch.setattr("src.services.component_scanner.scan_components", _fake_scan)
    monkeypatch.setattr(EC, "_loaded_index", lambda s: ({}, []))
    monkeypatch.setattr(EC, "_component_loaded_index", lambda s: {})
    entries = EC.component_catalog_entries(None)
    by_role = {e.name.split(":")[1]: e for e in entries}
    assert by_role["diffusion_models"].arch == "z-image"
    assert by_role["clip"].arch is None  # 只 diffusion_models 推断
