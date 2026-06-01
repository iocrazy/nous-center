"""SeedVR2 PR-1 骨架测试(CI 安全 —— 不 import torch/diffusers 重链)。

真推理 import + 出图验证靠 standalone smoke(tests/manual/smoke_seedvr2.py,真模型/GPU,非 CI)。
本测试只验:① 兼容 patch 幂等且不依赖 transformers 真装 ② vendored 目录结构在位
③ adapter 模块源码 wiring(patch 在 import vendored 前调)。
"""
from __future__ import annotations

import pathlib

import pytest

_VENDOR = pathlib.Path(__file__).parent.parent / "src/services/inference/seedvr2_vendor"


def test_compat_patch_idempotent_and_safe():
    """apply_seedvr2_compat_patches 幂等;transformers 缺失时静默(不抛)。"""
    from src.services.inference.seedvr2_compat import apply_seedvr2_compat_patches
    # 多次调用不抛(幂等)。CI 无 transformers → 走 except 静默分支。
    apply_seedvr2_compat_patches()
    apply_seedvr2_compat_patches()


def test_compat_patch_fixes_flash_attn_key_when_transformers_present():
    """transformers **真装**时,patch 补上缺失的 PACKAGE_DISTRIBUTION_MAPPING['flash_attn'] key。
    CI mock torch(transformers import 会触发 torch.__spec__ 错)→ skip。只在 dev box 真 transformers 跑。"""
    try:
        import transformers.utils.import_utils as iu  # noqa: PLC0415
    except Exception:  # noqa: BLE001 — CI mock torch / 无 transformers
        pytest.skip("transformers 不可用(CI mock torch)")
    mapping = getattr(iu, "PACKAGE_DISTRIBUTION_MAPPING", None)
    if mapping is None:
        pytest.skip("transformers 版本无 PACKAGE_DISTRIBUTION_MAPPING")
    from src.services.inference.seedvr2_compat import apply_seedvr2_compat_patches
    apply_seedvr2_compat_patches()
    assert "flash_attn" in mapping
    # is_flash_attn_2_available 不再 KeyError/IndexError(flash_attn 没装时返 False)
    assert isinstance(iu.is_flash_attn_2_available(), bool)


def test_vendored_dirs_present():
    """vendoring 完整:保留 NumZ 原 src/ 结构(6 目录在 src/ 下)+ configs/emb 在根 + LICENSE。
    不带 interfaces(ComfyUI 桥层)。结构必须对齐 NumZ —— 否则 config 路径
    (script_directory/src/models/...、script_directory/configs_7b)解析错。"""
    for d in ["core", "optimization", "common", "utils", "models", "data"]:
        assert (_VENDOR / "src" / d).is_dir(), f"vendored 缺 src/{d}/"
    assert (_VENDOR / "configs_7b/main.yaml").exists(), "缺 configs_7b(DiT config 在 vendored 根)"
    assert (_VENDOR / "pos_emb.pt").exists(), "缺 pos_emb.pt(文本嵌入,推理必需)"
    assert (_VENDOR / "neg_emb.pt").exists(), "缺 neg_emb.pt"
    assert (_VENDOR / "LICENSE_SEEDVR2").exists(), "缺 Apache LICENSE 署名"
    assert not (_VENDOR / "src" / "interfaces").exists(), "interfaces/ 是 ComfyUI 桥层,不该 vendoring"


def test_vendored_core_has_inference_entrypoints():
    """vendored core 有 prepare_runner + 4 阶段函数(adapter PR-2 要调的)。"""
    gu = (_VENDOR / "src/core/generation_utils.py").read_text()
    gp = (_VENDOR / "src/core/generation_phases.py").read_text()
    assert "def prepare_runner(" in gu
    for fn in ["encode_all_batches", "upscale_all_batches", "decode_all_batches", "postprocess_all_batches"]:
        assert f"def {fn}(" in gp, f"vendored 缺阶段 {fn}"


def test_adapter_patches_before_vendor_import():
    """image_seedvr2.py 必须**先调 patch 再 import vendored**(否则 transformers bug 触发)。"""
    src = (_VENDOR.parent / "image_seedvr2.py").read_text()
    patch_pos = src.find("apply_seedvr2_compat_patches()")
    assert patch_pos != -1, "adapter 没调 compat patch"
    # vendored.core 的真 import 只在 vendored_import_ok() 内部(延迟),顶层不直接 import 业务模块,
    # 所以 patch(顶层 apply)一定在任何 vendored 重链 import 之前执行。
