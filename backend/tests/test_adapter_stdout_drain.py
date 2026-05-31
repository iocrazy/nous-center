"""round3 #1:vLLM/SGLang adapter stdout 抽干逻辑 —— 防 PIPE 填满死锁。

直接测 `_drain_stdout`(用 __new__ 绕构造,免依赖真 vllm/sglang)。
"""

from collections import deque

import pytest

from src.services.inference.llm_sglang import SGLangAdapter
from src.services.inference.llm_vllm import VLLMAdapter


class _FakeProc:
    def __init__(self, lines):
        self.stdout = iter(lines)


@pytest.mark.parametrize("cls", [VLLMAdapter, SGLangAdapter])
def test_drain_reads_all_lines(cls):
    a = cls.__new__(cls)
    a._stdout_tail = deque(maxlen=200)
    a._process = _FakeProc(["a\n", "b\n", "c\n"])
    a._drain_stdout()
    assert "".join(a._stdout_tail) == "a\nb\nc\n"


@pytest.mark.parametrize("cls", [VLLMAdapter, SGLangAdapter])
def test_drain_bounded_keeps_only_tail(cls):
    """deque(maxlen) 有界 —— 长跑日志只留尾部,不爆内存。"""
    a = cls.__new__(cls)
    a._stdout_tail = deque(maxlen=2)
    a._process = _FakeProc([f"l{i}\n" for i in range(10)])
    a._drain_stdout()
    assert list(a._stdout_tail) == ["l8\n", "l9\n"]


@pytest.mark.parametrize("cls", [VLLMAdapter, SGLangAdapter])
def test_drain_no_process_is_safe(cls):
    a = cls.__new__(cls)
    a._stdout_tail = deque()
    a._process = None
    a._drain_stdout()  # 不该抛
    assert len(a._stdout_tail) == 0
