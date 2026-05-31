"""_build_messages 输入契约 —— 重点锁住「纯媒体无文字 prompt」合法(caption 用例)。

回归 #224 后果:_get_inputs 改精确 handle 路由后,只接 image handle 的边不再 spread
到 text/prompt;此前是那个 spread 巧合喂了 prompt,纯图节点才没报错。现在媒体存在时
允许空 prompt。
"""

import pytest

from src.services.nodes.llm import _LLMExecutionError, _build_messages


def test_image_only_no_prompt_is_allowed():
    """只连了 image、没有文字 prompt → 不报错,发纯图 content(caption)。"""
    msgs = _build_messages(
        {"model": "vl"},
        {"image": "data:image/png;base64,xxx"},
    )
    assert len(msgs) == 1
    content = msgs[0].content
    assert isinstance(content, list)
    # 纯 caption:不塞空 text part,只有 image_url
    assert all(part.get("type") != "text" for part in content)
    assert any(part.get("type") == "image_url" for part in content)


def test_image_with_prompt_includes_both():
    msgs = _build_messages(
        {"model": "vl"},
        {"prompt": "描述这张图", "image": "data:image/png;base64,xxx"},
    )
    content = msgs[-1].content
    types = [p.get("type") for p in content]
    assert "text" in types and "image_url" in types


def test_data_prompt_used_as_fallback():
    """节点静态 data.prompt 作为兜底(连入 prompt/text 缺失时)。"""
    msgs = _build_messages({"model": "m", "prompt": "你好"}, {})
    assert msgs[-1].content == "你好"


def test_no_prompt_and_no_media_still_errors():
    """既无文字也无媒体 → 仍然是真的缺输入,报错。"""
    with pytest.raises(_LLMExecutionError):
        _build_messages({"model": "m"}, {})


def test_input_prompt_beats_data_prompt():
    msgs = _build_messages({"model": "m", "prompt": "static"}, {"prompt": "wired"})
    assert msgs[-1].content == "wired"
