"""SeedVR2 PR-3b:runner dispatch 接入(CI 安全 —— runner_process/base/image_output_storage
顶层无 torch 重链)。

验:① node_type="upscale" → _build_request 构 UpscaleRequest(image 解析 + resolution/seed/
color)② 签名 image_url → 本地磁盘路径解析(免 HTTP 回环)③ workflow_executor 把
seedvr2_upscale 映射成 group_id=image(routes image runner)+ runner role=upscale(构
UpscaleRequest 而非 ImageRequest)—— role 与 group_id 解耦的关键。
"""
from __future__ import annotations

import pytest

from src.runner import protocol as P
from src.runner.runner_process import _build_request, _resolve_input_image_path


def _node(inputs, node_type="upscale"):
    return P.RunNode(task_id=7, node_id="up", node_type=node_type, model_key=None, inputs=inputs)


def test_build_upscale_request_from_image_url():
    """node_type=upscale + inputs.image_url → UpscaleRequest(本地路径 + resolution/seed/color)。"""
    from src.services.inference.base import MediaModality, UpscaleRequest

    req = _build_request(_node({
        "image_url": "/tmp/in.png",  # 本地路径(非 /files/)→ 原样
        "resolution": 1024, "seed": 123, "color_correction": "wavelet",
    }))
    assert isinstance(req, UpscaleRequest)
    assert req.modality == MediaModality.IMAGE
    assert req.image == "/tmp/in.png"
    assert req.resolution == 1024
    assert req.seed == 123
    assert req.color_correction == "wavelet"


def test_build_upscale_request_defaults():
    """缺 resolution/seed/color → 走默认(1080 / None / lab)。"""
    req = _build_request(_node({"image_url": "/tmp/in.png"}))
    assert req.resolution == 1080
    assert req.seed is None
    assert req.color_correction == "lab"


def test_build_upscale_request_three_node_params():
    """PR-2:增强节点 per-inference 参数(max_resolution/batch_size/noise)从 inputs 读;缺→默认。"""
    req = _build_request(_node({
        "image_url": "/tmp/in.png", "max_resolution": "2160", "batch_size": "4",
        "input_noise_scale": "0.1", "latent_noise_scale": "0.05",
    }))
    assert req.max_resolution == 2160
    assert req.batch_size == 4
    assert abs(req.input_noise_scale - 0.1) < 1e-6
    assert abs(req.latent_noise_scale - 0.05) < 1e-6
    # 缺 → 默认(单图安全值)
    req2 = _build_request(_node({"image_url": "/tmp/in.png"}))
    assert req2.max_resolution == 0
    assert req2.batch_size == 1


def test_seedvr2_loaders_are_inline():
    """三节点:DiT/VAE loader 是 inline(CPU 产配置,主进程);只有增强是 dispatch(GPU runner)。"""
    from src.services.node_routing import node_exec_class

    assert node_exec_class("seedvr2_load_dit") == "inline"
    assert node_exec_class("seedvr2_load_vae") == "inline"
    assert node_exec_class("seedvr2_upscale") == "dispatch"


def test_build_upscale_request_missing_image_raises():
    """缺上游 image → ValueError(node-executor 转 failed,不静默)。"""
    with pytest.raises(ValueError, match="image"):
        _build_request(_node({"resolution": 1024}))


def test_resolve_signed_image_url_to_disk_path():
    """签名 image_url(/files/images/<date>/<uuid>.<ext>?token=...)→ 本地磁盘路径
    (resolve_path 拼 NAS_OUTPUTS_PATH/<date>/<uuid>.<ext>),runner 直接读图免 HTTP。"""
    url = "/files/images/2026-06-02/abc123.png?token=deadbeef&expires=9999999999"
    path = _resolve_input_image_path(url)
    assert path.endswith("/2026-06-02/abc123.png"), path
    assert "?token" not in path  # query 已剥离


def test_resolve_passthrough_non_files():
    """本地路径 / data URI 原样透传(交给 adapter._decode_image)。"""
    assert _resolve_input_image_path("/abs/local.png") == "/abs/local.png"
    assert _resolve_input_image_path("data:image/png;base64,AA==") == "data:image/png;base64,AA=="


def test_unknown_node_type_still_raises():
    """非 image/tts/upscale → ValueError(含三种合法 role 提示)。"""
    with pytest.raises(ValueError, match="upscale"):
        _build_request(_node({}, node_type="bogus"))


def test_workflow_executor_role_group_decoupled():
    """seedvr2_upscale:group_id=image(routes image runner)但 runner role=upscale
    (构 UpscaleRequest)。这是 SeedVR2 跑在 image GPU 组却需独立 request 的关键解耦。"""
    from src.services.workflow_executor import (
        _NODE_TYPE_TO_GROUP_ID,
        _NODE_TYPE_TO_RUNNER_ROLE,
    )

    assert _NODE_TYPE_TO_GROUP_ID["seedvr2_upscale"] == "image"
    assert _NODE_TYPE_TO_RUNNER_ROLE["seedvr2_upscale"] == "upscale"
    # 其它节点 role == group_id(无解耦),回退逻辑(role 取不到时用 group_id)对它们安全。
    assert _NODE_TYPE_TO_RUNNER_ROLE["flux2_vae_decode"] == _NODE_TYPE_TO_GROUP_ID["flux2_vae_decode"]
    assert _NODE_TYPE_TO_RUNNER_ROLE["tts_engine"] == _NODE_TYPE_TO_GROUP_ID["tts_engine"]
