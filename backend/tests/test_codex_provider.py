"""CodexProvider 移植正确性 + generate 流程(mock subprocess,不碰真账号)。"""
from __future__ import annotations

import pytest

from src.services.external_providers.base import ExternalGenRequest, ProviderError
from src.services.external_providers.codex import (
    CodexProvider,
    available_feature_names,
    build_exec_args,
    extract_artifacts_from_text,
    make_creator_prompt,
    parse_codex_jsonl,
    parse_feature_list,
)


# ---- 纯函数 --------------------------------------------------------------

def test_parse_feature_list():
    raw = "image_generation   under development   false\nweb_search   stable   true\nbad line"
    feats = parse_feature_list(raw)
    names = {f["name"]: f["enabled"] for f in feats}
    assert names == {"image_generation": False, "web_search": True}


def test_available_feature_names_only_enabled():
    feats = parse_feature_list("image_generation under development false\nweb_search stable true")
    assert available_feature_names(feats) == {"web_search"}


def test_parse_codex_jsonl_extracts_agent_message_and_session():
    stdout = "\n".join([
        '{"type":"thread.started","thread_id":"th-1"}',
        '{"type":"item.completed","item":{"type":"reasoning","text":"思考"}}',
        '{"type":"item.completed","item":{"type":"agent_message","text":"最终图片 /tmp/x.png"}}',
        '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5}}',
    ])
    parsed = parse_codex_jsonl(stdout)
    assert parsed["session_id"] == "th-1"
    assert "最终图片" in parsed["text"]
    assert parsed["error"] == ""


def test_parse_codex_jsonl_captures_error():
    stdout = '{"type":"turn.failed","error":{"message":"rate limited"}}'
    assert parse_codex_jsonl(stdout)["error"] == "rate limited"


def test_extract_artifacts_from_text_markdown_and_loose():
    text = "见 ![out](/tmp/a.png) 和 https://x.com/b.jpg 还有非图 /tmp/c.txt"
    arts = extract_artifacts_from_text(text)
    assert "/tmp/a.png" in arts
    assert "https://x.com/b.jpg" in arts
    assert all(not a.endswith(".txt") for a in arts)


def test_build_exec_args_enables_image_generation_only_when_available():
    on = build_exec_args(images=[], available={"image_generation"}, image_generation=True)
    assert "--enable" in on and "image_generation" in on
    off = build_exec_args(images=[], available=set(), image_generation=True)
    assert "--enable" not in off
    assert on[-1] == "-" and off[-1] == "-"      # prompt 走 stdin


def test_build_exec_args_attaches_input_images():
    args = build_exec_args(images=["/tmp/ref.png"], available=set(), image_generation=False)
    assert "-i" in args and "/tmp/ref.png" in args


def test_make_creator_prompt_reflects_feature_availability():
    req = ExternalGenRequest(prompt="一只猫")
    assert "image_generation 工具" in make_creator_prompt(req, True)
    assert "不可用" in make_creator_prompt(req, False)


# ---- generate 流程(mock _run) ------------------------------------------

@pytest.fixture
def provider():
    return CodexProvider()


async def test_generate_collects_workspace_image(provider, monkeypatch):
    calls = {"n": 0}

    async def fake_run(args, *, timeout=120, cwd=None, stdin_data=None, env=None):
        calls["n"] += 1
        if list(args[:2]) == ["features", "list"]:
            return (0, "image_generation development true", "")  # 假装可用
        # exec:把图片写进 cwd(workspace)
        from pathlib import Path

        (Path(cwd) / "gen.png").write_bytes(b"\x89PNG fake")
        out = '{"type":"item.completed","item":{"type":"agent_message","text":"已生成 ./gen.png"}}'
        return (0, out, "")

    monkeypatch.setattr(provider, "_run", fake_run)
    result = await provider.generate(ExternalGenRequest(prompt="猫"))
    assert len(result.artifacts) == 1
    assert result.artifacts[0].title == "gen.png"


async def test_generate_errors_when_no_image_and_feature_gated(provider, monkeypatch):
    async def fake_run(args, *, timeout=120, cwd=None, stdin_data=None, env=None):
        if list(args[:2]) == ["features", "list"]:
            return (0, "image_generation under development false", "")  # 被 gate
        out = '{"type":"item.completed","item":{"type":"agent_message","text":"提示词:a cat"}}'
        return (0, out, "")

    monkeypatch.setattr(provider, "_run", fake_run)
    with pytest.raises(ProviderError) as exc:
        await provider.generate(ExternalGenRequest(prompt="猫"))
    assert "image_generation" in exc.value.message
    assert "提示词" in exc.value.message      # 文本退回带在错误里


async def test_generate_raises_on_nonzero_exit(provider, monkeypatch):
    async def fake_run(args, *, timeout=120, cwd=None, stdin_data=None, env=None):
        if list(args[:2]) == ["features", "list"]:
            return (0, "", "")
        return (1, "", '{"type":"error","message":"boom"}')

    monkeypatch.setattr(provider, "_run", fake_run)
    with pytest.raises(ProviderError):
        await provider.generate(ExternalGenRequest(prompt="x"))
