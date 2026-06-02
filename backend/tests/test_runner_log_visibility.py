"""runner 子进程日志可见性(_init_runner_logging)。

runner 由 mp spawn 起,不继承父进程 logging handler → logger.info 默认丢弃,模型加载 /
组件 L1 命中 / 驱逐等 INFO 全隐身(用户「日志不统一管理」的一环)。_init_runner_logging
配一个到 stdout 的 INFO handler(stdout 已被父进程重定向进统一日志)。这里验配置 + 幂等。
"""
from __future__ import annotations

import logging

from src.runner.runner_process import _init_runner_logging


def _drop_runner_handlers(root: logging.Logger) -> None:
    for h in [h for h in root.handlers if getattr(h, "_nous_runner", False)]:
        root.removeHandler(h)


def test_init_adds_stdout_info_handler():
    root = logging.getLogger()
    _drop_runner_handlers(root)
    try:
        _init_runner_logging("image")
        runner_handlers = [h for h in root.handlers if getattr(h, "_nous_runner", False)]
        assert len(runner_handlers) == 1
        h = runner_handlers[0]
        assert isinstance(h, logging.StreamHandler)
        # 前缀含 group,便于和主进程日志区分
        assert "[runner:image]" in h.formatter._fmt
        assert root.level <= logging.INFO
    finally:
        _drop_runner_handlers(root)


def test_init_is_idempotent():
    """重复调用不叠 handler(避免日志重复 N 行)。"""
    root = logging.getLogger()
    _drop_runner_handlers(root)
    try:
        _init_runner_logging("image")
        _init_runner_logging("image")
        _init_runner_logging("tts")
        runner_handlers = [h for h in root.handlers if getattr(h, "_nous_runner", False)]
        assert len(runner_handlers) == 1, "幂等:已配过就不再加 handler"
    finally:
        _drop_runner_handlers(root)


def test_level_override_via_env(monkeypatch):
    root = logging.getLogger()
    _drop_runner_handlers(root)
    monkeypatch.setenv("NOUS_RUNNER_LOG_LEVEL", "warning")
    try:
        _init_runner_logging("image")
        assert root.level == logging.WARNING
    finally:
        _drop_runner_handlers(root)
        root.setLevel(logging.WARNING)
