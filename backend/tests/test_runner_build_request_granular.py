"""PR-1 T4: _build_request 见嵌套 latent+vae → ImageRequest;clip/vae device=unet device。"""
from __future__ import annotations

import pytest

from src.runner import protocol as P
from src.runner.runner_process import _build_request


def _node(inputs):
    return P.RunNode(task_id=1, node_id="dec", node_type="image", model_key=None, inputs=inputs)


def _granular_inputs(unet_dev="cuda:1", loras=None, arch="flux2", input_image=None):
    model = {"_type": "flux2_model",
             "spec": {"kind": "diffusion_models", "file": "/m/u.safe", "device": unet_dev, "dtype": "fp8_e4m3", "adapter_arch": arch},
             "loras": loras or []}
    cond = {"_type": "flux2_conditioning",
            "clip": {"_type": "flux2_clip", "type": "flux2",
                     "encoders": [{"kind": "clip", "file": "/m/c.safe", "dtype": "default"}]},
            "text": "a cat", "negative": ""}
    latent = {"_type": "flux2_latent", "model": model, "conditioning": cond,
              "width": 768, "height": 768, "steps": 9, "cfg_scale": 4.0, "seed": 42}
    if input_image is not None:
        latent["input_image"] = input_image
    vae = {"_type": "flux2_vae", "spec": {"kind": "vae", "file": "/m/v.safe", "dtype": "default"}}
    return {"latent": latent, "vae": vae, "url_ttl_seconds": "3600"}


def test_granular_flatten_single_card():
    req = _build_request(_node(_granular_inputs(unet_dev="cuda:1")))
    assert req.components is not None
    # 逐组件选卡(2026-06-04):runner 不再强制 clip/vae 同卡。描述符无显式 device →
    # clip/vae 带 'auto'(下游 get_or_load_image_adapter 把 auto 解析成跟随 transformer 卡 = 零回归)。
    assert req.components["diffusion_models"].device == "cuda:1"
    assert req.components["clip"].device == "auto"
    assert req.components["vae"].device == "auto"
    assert req.components["clip"].file == "/m/c.safe"
    assert req.prompt == "a cat"
    assert (req.width, req.height, req.steps, req.seed) == (768, 768, 9, 42)
    assert req.pipeline_class == "Flux2KleinPipeline"


def test_granular_zimage_arch_routes_to_zimage_pipeline():
    """adapter_arch='z-image' → ImageRequest.pipeline_class='ZImagePipeline'(P1,经注册表派发)。"""
    req = _build_request(_node(_granular_inputs(arch="z-image")))
    assert req.pipeline_class == "ZImagePipeline"
    assert req.components["diffusion_models"].adapter_arch == "z-image"


def test_granular_no_input_image_is_none():
    """无 input_image 字段(纯文生图,默认)→ req.input_image is None(零回归)。"""
    req = _build_request(_node(_granular_inputs()))
    assert req.input_image is None


def test_granular_input_image_local_path_passthrough():
    """latent.input_image = 本地路径(非 /files/ 签名 URL)→ _resolve 原样过 → req.input_image。"""
    req = _build_request(_node(_granular_inputs(input_image="/tmp/foo.png")))
    assert req.input_image == "/tmp/foo.png"


def test_granular_input_image_data_uri_passthrough():
    """data URI 也原样过(交给引擎 _decode_input_image),不当作签名 URL 解析。"""
    uri = "data:image/png;base64,AAAA"
    req = _build_request(_node(_granular_inputs(input_image=uri)))
    assert req.input_image == uri


def test_granular_input_image_multi_comma_resolved_each():
    """多参考图(逗号分隔)→ 每路各自 resolve,再逗号拼回(本地路径原样)。"""
    req = _build_request(_node(_granular_inputs(input_image="/tmp/a.png, /tmp/b.png")))
    assert req.input_image == "/tmp/a.png,/tmp/b.png"


def test_granular_carries_loras():
    inp = _granular_inputs(loras=[{"name": "a", "path": "/m/loras/a.safe", "strength": 0.8}])
    req = _build_request(_node(inp))
    assert req.components["diffusion_models"].loras[0].name == "a"
    assert req.components["diffusion_models"].loras[0].path == "/m/loras/a.safe"


def test_granular_auto_device_passthrough():
    req = _build_request(_node(_granular_inputs(unet_dev="auto")))
    # auto 不在此解析(runner get_or_load_image_adapter 解析);三组件都带 auto
    assert req.components["diffusion_models"].device == "auto"
    assert req.components["vae"].device == "auto"


def test_granular_carries_sampler_scheduler():
    """PR-2:latent 的 sampler_name/scheduler 透传到 ImageRequest。"""
    inp = _granular_inputs()
    inp["latent"]["sampler_name"] = "heun"
    inp["latent"]["scheduler"] = "karras"
    req = _build_request(_node(inp))
    assert req.sampler_name == "heun" and req.scheduler == "karras"


def test_granular_sampler_scheduler_defaults():
    """无 sampler_name/scheduler key → 默认 euler/normal(= 参考库现状)。"""
    req = _build_request(_node(_granular_inputs()))
    assert req.sampler_name == "euler" and req.scheduler == "normal"


def test_granular_blank_seed_is_none():
    inp = _granular_inputs()
    inp["latent"]["seed"] = ""
    req = _build_request(_node(inp))
    assert req.seed is None


def test_granular_multi_encoder_gated():
    inp = _granular_inputs()
    inp["latent"]["conditioning"]["clip"]["encoders"].append(
        {"kind": "clip", "file": "/m/c2.safe", "dtype": "default"})
    with pytest.raises(ValueError, match="执行未就绪|多编码器架构"):
        _build_request(_node(inp))


def test_non_granular_image_falls_back_to_model_key_path():
    """非细粒度图 inputs(无 latent/vae)→ model_key 单模型 ImageRequest(无 components);
    runner 据 node.model_key 走 get_or_load。Family B flat unet/clip/vae 组件分支已删。"""
    req = _build_request(_node({"prompt": "x", "steps": 7}))
    assert req.components is None
    assert req.prompt == "x" and req.steps == 7
