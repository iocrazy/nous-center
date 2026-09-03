"""conftest 的 Popen 护栏本身要有测试 —— 护栏坏了比没有更危险(大家以为有)。

2026-09-02 事故:测试真起 `python -m vllm.entrypoints...`,把 RTX 3090 驱动跑挂。
护栏:测试进程里 argv 含推理服务入口的 Popen 一律 AssertionError,除非
NOUS_RUN_GPU_TESTS=1。
"""

import os
import subprocess
import sys

import pytest


def test_popen_blocks_inference_server_spawn(monkeypatch):
    monkeypatch.delenv("NOUS_RUN_GPU_TESTS", raising=False)
    for argv in (
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", "x"],
        [sys.executable, "-m", "sglang.launch_server"],
        ["/some/venv/bin/sgl-omni", "serve"],
    ):
        with pytest.raises(AssertionError, match="BLOCKED real inference-server spawn"):
            subprocess.Popen(argv)


def test_popen_blocks_string_argv_too(monkeypatch):
    """shell=True 的字符串 argv 同样拦。"""
    monkeypatch.delenv("NOUS_RUN_GPU_TESTS", raising=False)
    with pytest.raises(AssertionError):
        subprocess.Popen("python -m vllm.entrypoints.openai.api_server", shell=True)


def test_popen_still_works_for_ordinary_commands():
    """护栏不能误伤:普通子进程照常起(runner 测试等依赖它)。"""
    proc = subprocess.Popen([sys.executable, "-c", "print('ok')"], stdout=subprocess.PIPE, text=True)
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 0 and out.strip() == "ok"
    assert isinstance(proc, subprocess.Popen)  # 子类化保留 isinstance 语义


def test_popen_guard_can_be_lifted_explicitly(monkeypatch):
    """NOUS_RUN_GPU_TESTS=1 放行(真机 e2e 用)。用一个不存在的入口验证:放行后错误来自
    真实的 Popen(找不到可执行文件),而不是护栏。"""
    monkeypatch.setenv("NOUS_RUN_GPU_TESTS", "1")
    with pytest.raises(FileNotFoundError):
        subprocess.Popen(["/nonexistent/bin/sgl-omni", "serve"])
    assert os.environ["NOUS_RUN_GPU_TESTS"] == "1"
