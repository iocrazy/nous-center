import pytest

from src.services.context.gzip_compact import GzipCompactContextEngine
from src.services.context.base import ContextOverflowError


def make_msgs(user_contents: list[str], with_system: bool = False):
    msgs = []
    if with_system:
        msgs.append({"role": "system", "content": "be brief"})
    for i, c in enumerate(user_contents):
        msgs.append({"role": "user", "content": c})
        msgs.append({"role": "assistant", "content": f"ack{i}"})
    return msgs


@pytest.mark.asyncio
async def test_compress_noop_when_under_budget():
    engine = GzipCompactContextEngine()
    msgs = make_msgs(["short"])
    result, truncated = await engine.compress(messages=msgs, max_tokens=10_000)
    assert result == msgs
    assert truncated is False


@pytest.mark.asyncio
async def test_compress_drops_oldest_nonsystem_turns():
    engine = GzipCompactContextEngine()
    long = "x" * 2000
    msgs = make_msgs([long, long, long, "recent"], with_system=True)
    result, truncated = await engine.compress(messages=msgs, max_tokens=500)
    assert truncated is True
    # system preserved
    assert result[0]["role"] == "system"
    # 最后一轮 user "recent" 应在结果里
    contents = [m["content"] for m in result]
    assert "recent" in contents


@pytest.mark.asyncio
async def test_compress_preserves_last_user_turn():
    engine = GzipCompactContextEngine()
    msgs = make_msgs(["a" * 5000, "b" * 5000, "last"])
    result, _ = await engine.compress(messages=msgs, max_tokens=100)
    assert result[-1]["content"] in ("last", "ack2")
    assert any(m.get("content") == "last" for m in result)


@pytest.mark.asyncio
async def test_compress_overflow_raises():
    engine = GzipCompactContextEngine()
    huge_last = "x" * 100_000
    msgs = make_msgs([huge_last])
    with pytest.raises(ContextOverflowError):
        await engine.compress(messages=msgs, max_tokens=100)


def test_should_compress_threshold():
    engine = GzipCompactContextEngine()
    small = make_msgs(["hi"])
    big = make_msgs(["x" * 10_000])
    assert engine.should_compress(messages=small, max_tokens=10_000) is False
    assert engine.should_compress(messages=big, max_tokens=100) is True


def test_engine_name():
    assert GzipCompactContextEngine().name == "gzip-compact"


@pytest.mark.asyncio
async def test_compress_preserves_instruction_system_position():
    """round2 #9:压缩时保持原始顺序 —— instructions(system,刻意放历史之后)不被提到 agent
    提示前。survivor 历史须仍在 instructions 之前。"""
    engine = GzipCompactContextEngine()
    msgs = [
        {"role": "system", "content": "AGENT"},
        {"role": "user", "content": "x" * 4000},      # OLD,会被丢
        {"role": "assistant", "content": "oldA"},      # OLD,会被丢
        {"role": "user", "content": "KEEP1"},          # 应保留
        {"role": "system", "content": "INSTR"},        # 应仍在 KEEP1 之后
        {"role": "user", "content": "RECENT"},
    ]
    result, truncated = await engine.compress(messages=msgs, max_tokens=300)
    assert truncated
    contents = [m["content"] for m in result]
    assert "AGENT" in contents and "INSTR" in contents and "RECENT" in contents
    i_keep1 = contents.index("KEEP1")
    i_instr = contents.index("INSTR")
    # 旧逻辑(system 全提前)会让 INSTR 跑到 KEEP1 前 → i_keep1 < i_instr 失败
    assert i_keep1 < i_instr, "历史 KEEP1 应在 instructions 之前(顺序保持)"
