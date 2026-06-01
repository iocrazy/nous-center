"""SeedVR2 vendored 包的兼容性 patch —— **必须在 import seedvr2_vendor 任何模块前调用**。

为什么:我们 pyproject 把 transformers 钉死 git+main(commit 657f650「Gemma 4 support」,
Gemma 4 需 transformers ≥5.5.0,models.yaml 还配着 gemma-4-26B)。但 transformers 5.6-dev 有个
**回归 bug**:`transformers/utils/import_utils.py is_flash_attn_2_available()` 里
`PACKAGE_DISTRIBUTION_MAPPING["flash_attn"]` 在 flash_attn 未装时 **key 不存在 → KeyError**
(本该优雅返 False)。被 diffusers `models/transformers/auraflow_transformer_2d.py` import 触发
(SeedVR2 vendored 代码 import diffusers single_file 链上)。

flash_attn 装不上(系统 CUDA 13.0 ≠ PyTorch 编译的 12.8,源码编译失败),transformers 不能降
(Gemma 4)。所以 patch:把缺失的 `flash_attn` key 补成正常映射 `['flash-attn']` ——
flash_attn 真没装时 `_is_package_available` 返 False,`is_available and ...` 短路,bug 不触发,
返 False(回退 SDPA,SeedVR2 本就支持)。
"""
from __future__ import annotations


def apply_seedvr2_compat_patches() -> None:
    """幂等。import seedvr2_vendor 前调用一次。"""
    try:
        import transformers.utils.import_utils as iu  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return
    mapping = getattr(iu, "PACKAGE_DISTRIBUTION_MAPPING", None)
    if mapping is not None and "flash_attn" not in mapping:
        # 正常映射(import 名→distribution 名);flash_attn 未装时不会被命中。
        mapping["flash_attn"] = ["flash-attn"]
        fn = getattr(iu, "is_flash_attn_2_available", None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()
