"""集成:ideogram4 双 DiT 描述符经**真 runner IPC 编解码(msgpack)**后 _build_request 仍带 unconditional_file。

runner 子进程边界:主进程 encode(RunNode) → msgpack 过 pipe → 子进程 decode → _build_request。
合并节点把第二 DiT 文件挂在 latent.model.unconditional_file(嵌套 dict 里的字符串)。本测验这条
**IPC 序列化契约**:descriptor 经真 protocol.encode/decode(msgpack)往返后,_build_request 仍能读出
unconditional_file → 子进程 get_or_load 才能建第二 DiT。(全栈真出图经 GPU smoke 已覆盖;此处补 IPC 边界。)
"""
from __future__ import annotations

from src.runner import protocol as P
from src.runner.runner_process import _build_request


def _vae_decode_node_inputs(arch="ideogram4", uncond_file="/m/ideogram4_unconditional_fp8_scaled.safe"):
    """flux2_vae_decode dispatch 节点的 inputs:嵌套 latent(带合并后 model)+ vae。"""
    model = {"_type": "flux2_model",
             "spec": {"kind": "diffusion_models", "file": "/m/ideogram4_fp8_scaled.safe",
                      "device": "cuda:2", "dtype": "fp8_e4m3", "adapter_arch": arch},
             "loras": [], "offload": "cpu"}
    if uncond_file:
        model["unconditional_file"] = uncond_file  # 合并节点 ideogram4_dual_guider 挂的第二 DiT
    cond = {"_type": "flux2_conditioning",
            "clip": {"_type": "flux2_clip", "type": "qwen",
                     "encoders": [{"kind": "clip", "file": "/m/qwen3vl.safe", "dtype": "bfloat16"}]},
            "text": "a fox", "negative": ""}
    latent = {"_type": "flux2_latent", "model": model, "conditioning": cond,
              "width": 512, "height": 512, "steps": 12, "cfg_scale": 7.0, "seed": 42}
    vae = {"_type": "flux2_vae", "spec": {"kind": "vae", "file": "/m/flux2-vae.safe", "dtype": "bfloat16"}}
    return {"latent": latent, "vae": vae, "url_ttl_seconds": "3600"}


def _roundtrip(node: P.RunNode) -> P.RunNode:
    """经真 IPC wire format(默认 msgpack)往返 —— 等价主进程 → pipe → 子进程。"""
    return P.decode(P.encode(node))


def test_ideogram4_unconditional_file_survives_msgpack_ipc():
    """合并节点的 unconditional_file(嵌套 latent.model 里)经 msgpack IPC 往返后,
    _build_request 仍把它放进 diffusion_models ComponentSpec → 子进程能建第二 DiT。"""
    node = P.RunNode(task_id=1, node_id="vd", node_type="image", model_key=None,
                     inputs=_vae_decode_node_inputs())
    decoded = _roundtrip(node)
    # 往返后 inputs 结构完整(msgpack 不丢嵌套字符串)
    assert decoded.inputs["latent"]["model"]["unconditional_file"] == "/m/ideogram4_unconditional_fp8_scaled.safe"
    req = _build_request(decoded)
    assert req.components["diffusion_models"].unconditional_file == "/m/ideogram4_unconditional_fp8_scaled.safe"
    assert req.components["diffusion_models"].adapter_arch == "ideogram4"
    assert req.pipeline_class == "Ideogram4Pipeline"


def test_non_ideogram4_no_unconditional_file_after_ipc():
    """非 ideogram4(无 unconditional_file)经 IPC 往返 → ComponentSpec.unconditional_file None(零回归)。"""
    node = P.RunNode(task_id=2, node_id="vd", node_type="image", model_key=None,
                     inputs=_vae_decode_node_inputs(arch="flux2", uncond_file=None))
    req = _build_request(_roundtrip(node))
    assert req.components["diffusion_models"].unconditional_file is None
