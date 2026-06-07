"""_component_need_mb 分片整模型显存估算(2026-06-08 真机 bug:Qwen-Image-Edit 54GB 被 auto
派到 24GB 3090 → OOM,因 spec.file 只是第 1 片 ~6GB 被当成整个 transformer)。"""
from __future__ import annotations

from types import SimpleNamespace

from src.services.gpu_allocator import GPUAllocator
from src.services.inference.registry import ModelRegistry
from src.services.model_manager import ModelManager


class _EmptyRegistry(ModelRegistry):
    def __init__(self):
        self._config_path = ""
        self._specs = {}


def _mm() -> ModelManager:
    return ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())


def _write(path, mb: int) -> None:
    path.write_bytes(b"\0" * (mb * 1024 * 1024))


def test_single_file_size_times_1_3(tmp_path):
    f = tmp_path / "model.safetensors"
    _write(f, 100)
    mm = _mm()
    need = mm._component_need_mb(SimpleNamespace(kind="vae", file=str(f)))
    assert 125 <= need <= 135  # 100MB × 1.3


def test_sharded_sums_all_shards(tmp_path):
    """5 片各 100MB → 估算应 ~650MB(500×1.3),不是单片 130MB。"""
    d = tmp_path / "transformer"
    d.mkdir()
    for i in range(1, 6):
        _write(d / f"diffusion_pytorch_model-{i:05d}-of-00005.safetensors", 100)
    mm = _mm()
    spec = SimpleNamespace(
        kind="diffusion_models",
        file=str(d / "diffusion_pytorch_model-00001-of-00005.safetensors"),
    )
    need = mm._component_need_mb(spec)
    assert 600 <= need <= 700, f"应按 5 片总和 ~650MB 估,实得 {need}(只数了第 1 片?)"


def test_missing_file_falls_back_to_table(tmp_path):
    mm = _mm()
    need = mm._component_need_mb(SimpleNamespace(kind="vae", file=str(tmp_path / "nope.safetensors")))
    assert need > 0  # 回退 _VRAM_EST_MB 表,不崩
