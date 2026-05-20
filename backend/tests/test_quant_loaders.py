"""QuantLoaderRegistry dispatch + per-format loader correctness.

For each quant format we use a small synthetic safetensors fixture rather than the
real ~18GB Flux2 files — the dequant logic only needs a handful of tensors to
exercise the code path. Real-file end-to-end is covered later by PR-2 smoke.
"""
from __future__ import annotations

# conftest.py stubs `torch` with MagicMock for GPU-less test runs (see the
# `for mod_name in [...]: sys.modules[mod_name] = MagicMock()` block). This
# file needs the REAL torch + safetensors to round-trip actual tensors, so
# restore the genuine modules before importing them. Safe because real torch
# is installed in .venv and we hide CUDA via CUDA_VISIBLE_DEVICES="".
import sys as _sys

# Save the stub objects before evicting them, so we can put them back if real
# torch isn't installed. CI runs `uv sync --frozen` WITHOUT the `image` extra →
# no real torch; if we delete conftest's MagicMock stub and don't restore it,
# every test module collected/run afterwards sees `import torch` fail (a
# session-wide cascade). Restore + skip keeps CI's other modules intact.
_saved_torch_stub = {
    _n: _sys.modules[_n]
    for _n in list(_sys.modules)
    if _n == "torch" or _n.startswith("torch.")
}
for _n in _saved_torch_stub:
    del _sys.modules[_n]

from pathlib import Path

import pytest

try:
    import torch
    from safetensors.torch import save_file
except ModuleNotFoundError:
    _sys.modules.update(_saved_torch_stub)
    pytest.skip(
        "real torch not installed (CI without image extra)",
        allow_module_level=True,
    )

from src.services.inference.component_spec import ComponentSpec
from src.services.inference.quant_loaders import QuantLoaderRegistry, QUANT_LOADERS, UnsupportedQuantError


def _make_plain_safetensors(tmp_path: Path, name: str) -> Path:
    """Synthetic plain bf16 safetensors with 3 small tensors."""
    sd = {
        "block.0.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "block.0.bias":   torch.zeros(8, dtype=torch.bfloat16),
        "block.1.weight": torch.randn(8, 4, dtype=torch.bfloat16),
    }
    path = tmp_path / f"{name}.safetensors"
    save_file(sd, str(path))
    return path


def test_registry_register_and_dispatch():
    reg = QuantLoaderRegistry()
    seen = []

    @reg.register(match=lambda spec: "marker_a" in spec.file)
    def loader_a(spec):
        seen.append("a")
        return "result_a"

    @reg.register(match=lambda spec: "marker_b" in spec.file)
    def loader_b(spec):
        seen.append("b")
        return "result_b"

    spec_a = ComponentSpec(kind="vae", file="/p/marker_a.safetensors", device="cpu", dtype="bfloat16")
    assert reg.dispatch(spec_a) == "result_a"
    assert seen == ["a"]

    spec_b = ComponentSpec(kind="vae", file="/p/marker_b.safetensors", device="cpu", dtype="bfloat16")
    assert reg.dispatch(spec_b) == "result_b"


def test_registry_first_match_wins():
    """Registration order = match priority. Specific must register before generic."""
    reg = QuantLoaderRegistry()

    @reg.register(match=lambda spec: "fp8" in spec.file)
    def specific(spec):
        return "specific"

    @reg.register(match=lambda spec: spec.file.endswith(".safetensors"))
    def generic(spec):
        return "generic"

    spec_fp8 = ComponentSpec(kind="unet", file="/p/foo-fp8.safetensors", device="cpu", dtype="bfloat16")
    assert reg.dispatch(spec_fp8) == "specific"

    spec_plain = ComponentSpec(kind="unet", file="/p/foo.safetensors", device="cpu", dtype="bfloat16")
    assert reg.dispatch(spec_plain) == "generic"


def test_registry_no_match_raises():
    reg = QuantLoaderRegistry()

    @reg.register(match=lambda spec: False)
    def never(spec):
        return None

    spec = ComponentSpec(kind="vae", file="/p/x.gguf", device="cpu", dtype="bfloat16")
    with pytest.raises(UnsupportedQuantError, match="no quant loader matches"):
        reg.dispatch(spec)


def test_plain_safetensors_loader_loads_tensors(tmp_path):
    """The plain safetensors loader returns a state_dict-like mapping with original dtype."""
    sf = _make_plain_safetensors(tmp_path, "plain_bf16")
    spec = ComponentSpec(kind="vae", file=str(sf), device="cpu", dtype="bfloat16")

    result = QUANT_LOADERS.dispatch(spec)

    # Plain loader returns dict[str, Tensor] (caller wraps into module)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"block.0.weight", "block.0.bias", "block.1.weight"}
    assert result["block.0.weight"].dtype == torch.bfloat16
    assert result["block.0.weight"].shape == (8, 8)


def test_plain_safetensors_loader_honors_device(tmp_path):
    sf = _make_plain_safetensors(tmp_path, "plain_for_cpu")
    spec = ComponentSpec(kind="vae", file=str(sf), device="cpu", dtype="bfloat16")
    result = QUANT_LOADERS.dispatch(spec)
    assert result["block.0.weight"].device.type == "cpu"


def test_plain_safetensors_loader_gguf_not_supported(tmp_path):
    """GGUF is V2 PR-7 — V1 dispatches to UnsupportedQuantError."""
    gguf = tmp_path / "fake.gguf"
    gguf.write_bytes(b"GGUF\x00" * 16)
    spec = ComponentSpec(kind="unet", file=str(gguf), device="cpu", dtype="bfloat16")
    with pytest.raises(UnsupportedQuantError, match="GGUF .* V2"):
        QUANT_LOADERS.dispatch(spec)


def test_dtype_str_to_torch_raises_on_unknown():
    """Silent bfloat16 fallback would miscast user weights — must raise instead."""
    from src.services.inference.quant_loaders import _dtype_str_to_torch
    with pytest.raises(UnsupportedQuantError, match="unknown target dtype"):
        _dtype_str_to_torch("bf16")  # typo (correct is "bfloat16")
    with pytest.raises(UnsupportedQuantError, match="unknown target dtype"):
        _dtype_str_to_torch("fp4")   # future format not yet registered


def test_load_safetensors_plain_raises_on_unknown_dtype(tmp_path):
    """The plain loader propagates the dtype error via _dtype_str_to_torch."""
    from safetensors.torch import save_file
    sd = {"w": torch.zeros(2, 2, dtype=torch.bfloat16)}
    p = tmp_path / "bad_dtype.safetensors"
    save_file(sd, str(p))
    # ComponentSpec.dtype is just a str — validator only checks device — so "bf16"
    # (typo for "bfloat16") passes pydantic and the failure surfaces inside the loader.
    spec_bad = ComponentSpec(kind="vae", file=str(p), device="cpu", dtype="bf16")
    with pytest.raises(UnsupportedQuantError, match="unknown target dtype"):
        QUANT_LOADERS.dispatch(spec_bad)


# ---------- fp8mixed (PR-1 Task 3) ----------


def _make_fp8mixed_safetensors(tmp_path: Path, name: str) -> Path:
    """Synthetic comfy_quant-style fp8 fixture: 1 fp8 tensor + companion scale, plus 1 plain tensor."""
    weight_fp8 = torch.randn(4, 4).to(torch.float8_e4m3fn)
    weight_scale = torch.tensor([0.125], dtype=torch.float32)
    plain = torch.randn(4, 4, dtype=torch.bfloat16)
    sd = {
        "block.0.weight": weight_fp8,
        "block.0.weight_scale": weight_scale,
        "block.0.weight.comfy_quant": torch.tensor([1], dtype=torch.uint8),  # marker
        "block.1.weight": plain,
    }
    path = tmp_path / f"{name}.safetensors"
    save_file(sd, str(path))
    return path


def test_fp8mixed_loader_dequants_and_drops_metadata(tmp_path):
    sf = _make_fp8mixed_safetensors(tmp_path, "Flux2-Klein-9B-True-v2-fp8mixed")
    spec = ComponentSpec(kind="unet", file=str(sf), device="cpu", dtype="bfloat16")

    sd = QUANT_LOADERS.dispatch(spec)

    # fp8 tensor was dequant'd (multiplied by scale) into bfloat16
    assert "block.0.weight" in sd
    assert sd["block.0.weight"].dtype == torch.bfloat16
    # plain tensor passed through
    assert sd["block.1.weight"].dtype == torch.bfloat16
    # metadata keys must be dropped before caller's load_state_dict
    assert "block.0.weight_scale" not in sd
    assert "block.0.weight.comfy_quant" not in sd


def test_fp8mixed_loader_match_priority_over_plain():
    """File with 'fp8mixed' in name must dispatch to fp8 loader, not plain."""
    matchers = [m for m, _fn in QUANT_LOADERS._loaders]
    fp8_idx = next(i for i, m in enumerate(matchers)
                   if m(ComponentSpec(kind="unet", file="x-fp8mixed.safetensors",
                                      device="cpu", dtype="bfloat16")))
    plain_idx = next(i for i, m in enumerate(matchers)
                     if m(ComponentSpec(kind="unet", file="plain.safetensors",
                                        device="cpu", dtype="bfloat16")))
    assert fp8_idx < plain_idx


def test_fp8mixed_loader_dispatches_via_header_sniff_when_name_lacks_marker(tmp_path):
    """A file WITHOUT 'fp8mixed' in name but WITH comfy_quant header keys must
    still dispatch to load_fp8mixed (priority via _has_comfy_quant_metadata).
    Regression for the case where users rename quantized files.
    """
    # Build a synthetic safetensors with comfy_quant marker keys, but named generically
    weight_fp8 = torch.randn(4, 4).to(torch.float8_e4m3fn)
    weight_scale = torch.tensor([0.125], dtype=torch.float32)
    sd = {
        "block.0.weight": weight_fp8,
        "block.0.weight_scale": weight_scale,
        "block.0.weight.comfy_quant": torch.tensor([1], dtype=torch.uint8),
    }
    p = tmp_path / "plain-looking-name.safetensors"   # no 'fp8mixed' substring
    save_file(sd, str(p))
    spec = ComponentSpec(kind="unet", file=str(p), device="cpu", dtype="bfloat16")

    sd_loaded = QUANT_LOADERS.dispatch(spec)

    # If dispatch went to plain loader, metadata keys would leak through.
    # If dispatch went to fp8 loader (correct), metadata keys are dropped.
    assert "block.0.weight" in sd_loaded
    assert "block.0.weight_scale" not in sd_loaded
    assert "block.0.weight.comfy_quant" not in sd_loaded
    assert sd_loaded["block.0.weight"].dtype == torch.bfloat16  # dequant'd to target


def test_has_comfy_quant_metadata_returns_false_on_malformed_file(tmp_path):
    """Malformed safetensors file → _has_comfy_quant_metadata returns False
    (fail-soft) so dispatch falls through to plain loader, not crash.
    """
    from src.services.inference.quant_loaders import _has_comfy_quant_metadata

    bad = tmp_path / "garbage.safetensors"
    bad.write_bytes(b"\x00\x01\x02\x03not a safetensors file at all")
    assert _has_comfy_quant_metadata(str(bad)) is False


def test_has_comfy_quant_metadata_returns_false_on_missing_file(tmp_path):
    """Missing file → fail-soft False (not FileNotFoundError)."""
    from src.services.inference.quant_loaders import _has_comfy_quant_metadata
    assert _has_comfy_quant_metadata(str(tmp_path / "does-not-exist.safetensors")) is False


# ---------- mxfp8mixed (PR-1 Task 4) ----------


def _make_mxfp8mixed_safetensors(tmp_path: Path, name: str) -> Path:
    """Synthetic mxfp8 fixture: tensor in mxfp8 format with per-block scale.

    Layout used by community Flux2 mxfp8 quants:
      <name>.weight        : fp8_e4m3 weights (flat)
      <name>.weight_scale  : uint8 E8M0 per-32-element block scales (shape = numel/32)
      <name>.weight.comfy_quant : marker tensor (uint8)
    Real Flux2-Klein-9B-True-v2-mxfp8mixed.safetensors uses this layout.
    """
    # 64 elements → 2 blocks of 32 → 2 E8M0 scales
    w_fp8 = torch.randn(64).to(torch.float8_e4m3fn)
    # E8M0 scales: stored as uint8 representing power-of-2 exponent (bias 127)
    w_scale = torch.tensor([130, 128], dtype=torch.uint8)  # 2^3 and 2^1
    plain = torch.randn(4, 4, dtype=torch.bfloat16)
    sd = {
        "block.0.weight": w_fp8,
        "block.0.weight_scale": w_scale,
        "block.0.weight.comfy_quant": torch.tensor([2], dtype=torch.uint8),  # arch=2 == mxfp8
        "block.1.weight": plain,
    }
    path = tmp_path / f"{name}.safetensors"
    save_file(sd, str(path))
    return path


def test_mxfp8mixed_loader_block_dequants_to_target_dtype(tmp_path):
    sf = _make_mxfp8mixed_safetensors(tmp_path, "Flux2-X-mxfp8mixed")
    spec = ComponentSpec(kind="unet", file=str(sf), device="cpu", dtype="bfloat16")

    sd = QUANT_LOADERS.dispatch(spec)

    assert "block.0.weight" in sd
    assert sd["block.0.weight"].dtype == torch.bfloat16
    assert sd["block.0.weight"].numel() == 64
    assert "block.0.weight_scale" not in sd
    assert "block.0.weight.comfy_quant" not in sd
    # plain tensor preserved
    assert sd["block.1.weight"].shape == (4, 4)


def test_mxfp8mixed_loader_priority_over_fp8mixed():
    """File named mxfp8mixed must NOT fall through to fp8mixed loader."""
    matchers = [m for m, _fn in QUANT_LOADERS._loaders]
    mxfp8_idx = next(i for i, m in enumerate(matchers)
                     if m(ComponentSpec(kind="unet", file="x-mxfp8mixed.safetensors",
                                        device="cpu", dtype="bfloat16")))
    fp8_idx = next(i for i, m in enumerate(matchers)
                   if m(ComponentSpec(kind="unet", file="x-fp8mixed.safetensors",
                                      device="cpu", dtype="bfloat16")))
    assert mxfp8_idx < fp8_idx, "mxfp8mixed matcher must register before fp8mixed"


# ---------- nvfp4mixed (PR-1 Task 5) ----------


def _make_nvfp4mixed_safetensors(tmp_path: Path, name: str) -> Path:
    """Synthetic nvfp4 fixture: 4-bit weights packed 2-per-uint8 + per-16-block fp32 scale.

    Format spec (community Flux2 nvfp4 quants):
      <name>.weight        : uint8, shape = (numel / 2,)  — two 4-bit weights per byte
      <name>.weight_scale  : float32, shape = (numel / 16,) — one scale per 16-element block
      <name>.weight.comfy_quant : marker (uint8 [3])
      <name>.weight_shape  : int32 [H, W] — original shape for unpack
    """
    NUMEL = 64
    BLOCK = 16
    packed = torch.randint(0, 256, (NUMEL // 2,), dtype=torch.uint8)
    scale = torch.randn(NUMEL // BLOCK, dtype=torch.float32).abs() + 0.1
    plain = torch.randn(2, 2, dtype=torch.bfloat16)
    sd = {
        "block.0.weight": packed,
        "block.0.weight_scale": scale,
        "block.0.weight.comfy_quant": torch.tensor([3], dtype=torch.uint8),
        "block.0.weight_shape": torch.tensor([8, 8], dtype=torch.int32),  # original shape for unpack
        "block.1.weight": plain,
    }
    path = tmp_path / f"{name}.safetensors"
    save_file(sd, str(path))
    return path


def test_nvfp4mixed_loader_unpacks_4bit_blocks(tmp_path):
    sf = _make_nvfp4mixed_safetensors(tmp_path, "Flux2-X-nvfp4mixed")
    spec = ComponentSpec(kind="unet", file=str(sf), device="cpu", dtype="bfloat16")

    sd = QUANT_LOADERS.dispatch(spec)

    # 4-bit unpacked → 64 elements → reshaped to (8, 8)
    assert sd["block.0.weight"].shape == (8, 8)
    assert sd["block.0.weight"].dtype == torch.bfloat16
    # metadata dropped
    for suffix in ("_scale", ".comfy_quant", "_shape"):
        assert not any(k.endswith(suffix) for k in sd)
    # plain tensor preserved
    assert sd["block.1.weight"].shape == (2, 2)


def test_nvfp4mixed_loader_priority_over_mxfp8():
    matchers = [m for m, _fn in QUANT_LOADERS._loaders]
    nvfp4_idx = next(i for i, m in enumerate(matchers)
                     if m(ComponentSpec(kind="unet", file="x-nvfp4mixed.safetensors",
                                        device="cpu", dtype="bfloat16")))
    mxfp8_idx = next(i for i, m in enumerate(matchers)
                     if m(ComponentSpec(kind="unet", file="x-mxfp8mixed.safetensors",
                                        device="cpu", dtype="bfloat16")))
    assert nvfp4_idx < mxfp8_idx
