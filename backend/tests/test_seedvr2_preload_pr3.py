"""统一引擎库 PR-3:从引擎库预热/卸载 SeedVR2 —— 协议 + runner handler + client + 端点 wiring。
CI 安全(protocol/client 顶层无 torch;runner_process/engines 用源码检查避 torch mock 边界)。"""
from __future__ import annotations

import pathlib


def test_preload_seedvr2_protocol_message():
    """PreloadSeedVR2 消息存在 + 注册进 ALL_MESSAGES + Message 联合(主进程↔runner 收发)。"""
    from src.runner import protocol as P  # noqa: PLC0415

    msg = P.PreloadSeedVR2(model_dir="/m/SEEDVR2", dit_model="dit.safetensors", vae_model="vae.safetensors")
    assert msg.kind == "preload_seedvr2"
    assert P._KIND_TO_CLASS.get("preload_seedvr2") is P.PreloadSeedVR2  # 收发反序列化注册
    assert P.PreloadSeedVR2 in P.Message.__args__  # 在联合类型里(isinstance 分派)


def test_runner_handler_wired():
    """runner_process 有 _handle_preload_seedvr2 + _pipe_reader 派发(调 get_or_load_seedvr2_adapter)。"""
    src = (pathlib.Path(__file__).parent.parent / "src/runner/runner_process.py").read_text()
    assert "async def _handle_preload_seedvr2(" in src
    assert "get_or_load_seedvr2_adapter(" in src
    assert "isinstance(msg, P.PreloadSeedVR2)" in src  # _pipe_reader 派发


def test_runner_client_has_preload_seedvr2():
    """RunnerClient.preload_seedvr2 发 PreloadSeedVR2(主进程派给 image runner)。"""
    src = (pathlib.Path(__file__).parent.parent / "src/runner/client.py").read_text()
    assert "async def preload_seedvr2(" in src
    assert "P.PreloadSeedVR2(" in src


def test_engines_routes_registered():
    """engines.py 有 /seedvr2/preload + /seedvr2/unload 端点(引擎库前端调)。"""
    src = (pathlib.Path(__file__).parent.parent / "src/api/routes/engines.py").read_text()
    assert '"/seedvr2/preload"' in src
    assert '"/seedvr2/unload"' in src
    assert "client.preload_seedvr2(" in src  # preload 派给 image runner client
    assert "sup.client.unload_model(" in src  # unload 派 UnloadModel 给持有的 runner
