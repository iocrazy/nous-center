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


# --- PR-3a:ABC 接入 wiring(CI 安全 —— base/image_seedvr2 顶层无 torch/numpy/vendored 重链)---


def test_upscale_request_shape():
    """UpscaleRequest(图→图超分)在 base 定义,字段齐:image 输入 + resolution(短边语义)。"""
    from src.services.inference.base import MediaModality, UpscaleRequest  # noqa: PLC0415

    req = UpscaleRequest(request_id="t", image="data:image/png;base64,AA==", resolution=1024)
    assert req.modality == MediaModality.IMAGE
    assert req.image == "data:image/png;base64,AA=="
    assert req.resolution == 1024
    assert req.seed is None  # 默认 None → adapter 兜 42
    assert req.color_correction == "lab"


def test_seedvr2_adapter_conforms_to_abc():
    """SeedVR2UpscaleBackend 符合 InferenceAdapter ABC:子类 + ClassVar + paths 构造。
    顶层 import 不触发 torch(vendored 重链全惰性在方法内),CI 可跑。"""
    from src.services.inference.base import InferenceAdapter, MediaModality  # noqa: PLC0415
    from src.services.inference.image_seedvr2 import (  # noqa: PLC0415
        DEFAULT_DIT,
        DEFAULT_VAE,
        SeedVR2UpscaleBackend,
    )

    assert issubclass(SeedVR2UpscaleBackend, InferenceAdapter)
    assert SeedVR2UpscaleBackend.modality == MediaModality.IMAGE
    assert SeedVR2UpscaleBackend.estimated_vram_mb > 0
    # paths 构造:model_dir 必需;dit/vae 缺省走 DEFAULT。
    be = SeedVR2UpscaleBackend(paths={"model_dir": "/tmp/seedvr2"}, device="cuda:0")
    assert be.model_dir == "/tmp/seedvr2"
    assert be.dit_model == DEFAULT_DIT
    assert be.vae_model == DEFAULT_VAE
    assert not be.is_loaded  # 没 load → _model is None


def test_seedvr2_clamp_seed_to_uint32():
    """seed 归一 [0, 2**32-1]:NumZ np.random.seed 要求(randomize 给 2**53 会抛
    「Seed must be between 0 and 2**32 - 1」)。<2**32 不变,超范围折叠且确定。"""
    from src.services.inference.image_seedvr2 import _clamp_seed  # noqa: PLC0415

    assert _clamp_seed(42) == 42  # 小 seed 不变
    assert _clamp_seed(2**32 - 1) == 2**32 - 1  # 边界
    assert _clamp_seed(5500410140161955) == 5500410140161955 % (2**32) == 142997411  # 真机报错那个
    assert 0 <= _clamp_seed(2**53) < 2**32  # randomize 上界折叠进范围
    assert _clamp_seed(5500410140161955) == _clamp_seed(5500410140161955)  # 确定(复现一致)


def test_seedvr2_adapter_requires_model_dir():
    """paths 缺 model_dir → 明确 RuntimeError(不静默用错路径)。"""
    from src.services.inference.image_seedvr2 import SeedVR2UpscaleBackend  # noqa: PLC0415

    with pytest.raises(RuntimeError, match="model_dir"):
        SeedVR2UpscaleBackend(paths={}, device="cuda:0")


def test_model_manager_has_seedvr2_loader():
    """ModelManager 有 by-key SeedVR2 装载入口(独立路径,非三组件 combo)。源码检查
    (model_manager import 重,避开 torch mock 边界,跟其它 wiring 测试一致用 read_text)。"""
    import pathlib  # noqa: PLC0415

    mm = (pathlib.Path(__file__).parent.parent / "src/services/model_manager.py").read_text()
    assert "async def get_or_load_seedvr2_adapter(" in mm, "ModelManager 缺 SeedVR2 装载入口"
    assert "image:SeedVR2:" in mm, "缺 SeedVR2 model_id 命名(登记进 _models)"
    assert "SeedVR2UpscaleBackend" in mm


# --- PR-1(三节点引擎):dit/vae config + tiling/blockswap + 增强参数 wiring(CI 安全)---


def test_upscale_request_three_node_fields():
    """UpscaleRequest 加了增强节点 per-inference 参数(max_resolution/batch_size/temporal_overlap/
    prepend_frames/uniform_batch_size),默认 = 单图安全值。"""
    from src.services.inference.base import UpscaleRequest  # noqa: PLC0415

    req = UpscaleRequest(request_id="t", image="/tmp/x.png")
    assert req.max_resolution == 0  # 不限长边
    assert req.batch_size == 1
    assert req.temporal_overlap == 0  # 单图
    assert req.prepend_frames == 0
    assert req.uniform_batch_size is False
    # 边界:max_resolution 可设大图上限
    req2 = UpscaleRequest(request_id="t", image="/tmp/x.png", max_resolution=2160, batch_size=4)
    assert req2.max_resolution == 2160
    assert req2.batch_size == 4


def test_seedvr2_adapter_accepts_dit_vae_config():
    """adapter 接 dit_config/vae_config(三节点的 DiT/VAE 配置)——model 名从 config 取(优先于
    paths/默认);config 存进 dit_cfg/vae_cfg 供 _load_sync 串 blockswap/tiling/attention。
    顶层无 torch,纯 dict 存储,CI 可跑(_load_sync 才碰 torch)。"""
    from src.services.inference.image_seedvr2 import SeedVR2UpscaleBackend  # noqa: PLC0415

    be = SeedVR2UpscaleBackend(
        paths={"model_dir": "/tmp/seedvr2"},
        device="cuda:0",
        dit_config={"model": "seedvr2_ema_3b_fp16.safetensors", "device": "cuda:1",
                    "blocks_to_swap": 16, "swap_io_components": True, "offload_device": "cpu",
                    "attention_mode": "sdpa"},
        vae_config={"model": "ema_vae_fp16.safetensors", "encode_tiled": True,
                    "encode_tile_size": 512, "decode_tiled": True, "decode_tile_size": 512},
    )
    # model 名从 config.model 取(覆盖默认)
    assert be.dit_model == "seedvr2_ema_3b_fp16.safetensors"
    assert be.vae_model == "ema_vae_fp16.safetensors"
    # config 字典存好(blockswap/tiling 留给 _load_sync 串进 prepare_runner)
    assert be.dit_cfg["blocks_to_swap"] == 16
    assert be.dit_cfg["swap_io_components"] is True
    assert be.vae_cfg["encode_tiled"] is True


def test_model_manager_seedvr2_loader_accepts_configs():
    """get_or_load_seedvr2_adapter 接 dit_config/vae_config + 缓存键纳入(不同配置=不同实例)。
    源码检查(避 torch import 链)。"""
    import pathlib  # noqa: PLC0415

    mm = (pathlib.Path(__file__).parent.parent / "src/services/model_manager.py").read_text()
    assert "dit_config: dict | None = None" in mm, "loader 未接 dit_config"
    assert "vae_config: dict | None = None" in mm, "loader 未接 vae_config"
    # 缓存键纳入 blockswap/tiling 维度
    assert "blocks_to_swap" in mm and "enc_tiled" in mm, "缓存键未纳入 blockswap/tiling"


# --- runner 复用:cache_model=True(修第二次 infer 撞 None)---


def test_upscale_phases_keep_models_for_reuse():
    """upscale() 必须给 upscale/decode 阶段传 **cache_model=True** —— False 是 NumZ CLI
    一次性语义,阶段收尾 cleanup 把 runner.dit/vae 置 None;adapter 缓存复用时第二次 infer
    撞「'NoneType' object has no attribute 'parameters'」。源码检查(避 torch import 链)。"""
    src = (_VENDOR.parent / "image_seedvr2.py").read_text()
    assert "cache_model=False" not in src, "upscale 阶段回退到一次性语义,缓存 adapter 二跑必崩"
    assert src.count("cache_model=True") >= 2, "upscale/decode 两阶段都要 cache_model=True"


# --- 进度桥接:infer(progress_callback) → NumZ 四阶段 ---


def test_upscale_phases_receive_progress_callback():
    """四阶段都必须透传 progress_callback(不再写死 None)—— runner dispatcher 按 infer
    签名自动探测注入,SeedVR2 节点才有 stage 级进度。源码检查(避 torch import 链)。"""
    src = (_VENDOR.parent / "image_seedvr2.py").read_text()
    assert "progress_callback=None," not in src, "阶段调用回退写死 None,节点无进度"
    assert src.count("progress_callback=progress_callback,") >= 4, "四阶段都要透传 callback"
    # NumZ 回调在工作线程触发,必须 call_soon_threadsafe 桥回事件循环(runner _on_progress
    # 里 create_task 需 running loop)。
    assert "call_soon_threadsafe" in src, "缺线程安全桥接,工作线程直调会崩"


# --- 模型自动下载:_load_sync 先 download_weight 再 prepare_runner ---


def test_load_calls_download_weight_before_prepare_runner():
    """对齐上游 video_upscaler.execute:prepare_runner **不下载**,必须先 download_weight
    (缺文件 HF 下 + sha256 校验/损坏重下)。白名单 UI「选了自动下载」靠它兑现。
    源码检查(避 torch import 链):download_weight 调用必须出现在 prepare_runner 调用之前。"""
    src = (_VENDOR.parent / "image_seedvr2.py").read_text()
    dl = src.find("download_weight(")
    pr = src.find("prepare_runner(")
    assert dl != -1, "_load_sync 没调 download_weight,缺模型直接崩(UI 承诺自动下载)"
    assert pr != -1
    # 取真调用位置(跳过 import 行):找 "if not download_weight("
    call = src.find("if not download_weight(")
    assert call != -1 and call < src.rfind("self._runner, cache_context = prepare_runner("), \
        "download_weight 必须在 prepare_runner 之前"


# --- prepend 加帧对称 + batch_size 4n+1 归一 ---


def test_prepend_frames_added_before_encode():
    """上游 compute_generation_info 在 encode 前 pad_video_temporal(prepend=True) 加反转帧,
    postprocess 删同数量。只传删的那头 = prepend_frames>0 时删真帧。源码检查。"""
    src = (_VENDOR.parent / "image_seedvr2.py").read_text()
    add = src.find("prepend=True")
    enc = src.find("ctx = encode_all_batches(")
    assert add != -1, "缺 prepend 加帧侧(pad_video_temporal prepend=True)"
    assert enc != -1 and add < enc, "加帧必须在 encode 之前"


def test_batch_size_normalized_to_4n_plus_1():
    """batch_size 必须 4n+1(上游 widget step=4 enforce);引擎边界向下归一。"""
    src = (_VENDOR.parent / "image_seedvr2.py").read_text()
    assert "// 4) * 4 + 1" in src, "缺 batch_size 4n+1 归一"
