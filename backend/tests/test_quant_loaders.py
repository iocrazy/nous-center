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
for _n in list(_sys.modules.keys()):
    if _n == "torch" or _n.startswith("torch."):
        del _sys.modules[_n]

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

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
