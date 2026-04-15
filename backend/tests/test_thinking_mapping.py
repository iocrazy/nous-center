"""Unit tests for extra_body.thinking → chat_template_kwargs.enable_thinking mapping."""

from src.api.routes.openai_compat import (
    _maybe_inject_thinking,
    _supports_thinking,
)


def test_supports_thinking_qwen3():
    assert _supports_thinking("qwen3.5-35b-a3b-gptq-int4")
    assert _supports_thinking("Qwen3-8B")


def test_does_not_support_gemma():
    assert not _supports_thinking("gemma-4-26b-awq")
    assert not _supports_thinking("voxcpm2")


def test_inject_enabled_for_whitelisted():
    body = {"messages": [], "thinking": {"type": "enabled"}}
    _maybe_inject_thinking(body, "qwen3.5-35b")
    assert "thinking" not in body  # popped either way
    assert body["chat_template_kwargs"] == {"enable_thinking": True}


def test_inject_disabled_for_whitelisted():
    body = {"messages": [], "thinking": {"type": "disabled"}}
    _maybe_inject_thinking(body, "qwen3.5-35b")
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_auto_does_not_set_kwarg():
    body = {"messages": [], "thinking": {"type": "auto"}}
    _maybe_inject_thinking(body, "qwen3.5-35b")
    assert "chat_template_kwargs" not in body
    assert "thinking" not in body  # still popped


def test_silently_dropped_for_unsupported_model():
    body = {"messages": [], "thinking": {"type": "enabled"}}
    _maybe_inject_thinking(body, "gemma-4-26b-awq")
    assert "thinking" not in body  # popped
    assert "chat_template_kwargs" not in body  # NOT set


def test_no_thinking_field_is_noop():
    body = {"messages": []}
    _maybe_inject_thinking(body, "qwen3.5-35b")
    assert body == {"messages": []}


def test_invalid_thinking_type_dropped():
    body = {"messages": [], "thinking": {"type": "bogus"}}
    _maybe_inject_thinking(body, "qwen3.5-35b")
    assert "chat_template_kwargs" not in body


def test_thinking_not_dict_dropped():
    body = {"messages": [], "thinking": "enabled"}  # string instead of dict
    _maybe_inject_thinking(body, "qwen3.5-35b")
    assert "thinking" not in body
    assert "chat_template_kwargs" not in body


def test_preserves_existing_chat_template_kwargs():
    body = {
        "messages": [],
        "thinking": {"type": "enabled"},
        "chat_template_kwargs": {"some_other_kwarg": "value"},
    }
    _maybe_inject_thinking(body, "qwen3.5-35b")
    assert body["chat_template_kwargs"] == {
        "some_other_kwarg": "value",
        "enable_thinking": True,
    }
