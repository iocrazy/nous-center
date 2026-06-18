"""DreaminaProvider 移植正确性 + generate 流程(mock subprocess,不碰真账号)。"""
from __future__ import annotations

import pytest

from src.services.external_providers.base import ExternalGenRequest, ProviderError
from src.services.external_providers.dreamina import (
    DreaminaProvider,
    extract_json,
    failure_reason,
    image_model_version,
    image_resolution,
    output_values,
    ratio_from_size,
    submit_id_of,
)


# ---- 纯函数移植 ----------------------------------------------------------

def test_extract_json_picks_result_object():
    text = '噪声行\n{"queue_info": {}}\n{"submit_id": "abc", "gen_status": "done"}\n尾巴'
    obj = extract_json(text)
    assert obj["submit_id"] == "abc"


def test_extract_json_leading_object_wins():
    assert extract_json('{"total_credit": 5}') == {"total_credit": 5}


def test_submit_id_nested():
    assert submit_id_of({"data": {"task_id": "t-1"}}) == "t-1"
    assert submit_id_of({"x": [{"submit_id": "s-9"}]}) == "s-9"
    assert submit_id_of({"nope": 1}) == ""


def test_failure_reason_detects_fail():
    assert "无效" in failure_reason({"gen_status": "failed", "fail_reason": "参数无效"})
    assert failure_reason({"gen_status": "done"}) == ""


def test_output_values_dedup_and_filter():
    raw = {"images": ["/output/a.png", "/output/a.png"], "note": "not media"}
    assert output_values(raw) == ["/output/a.png"]


def test_ratio_from_size_snaps_to_choice():
    assert ratio_from_size(1920, 1080) == "16:9"
    assert ratio_from_size(1024, 1024) == "1:1"
    assert ratio_from_size(0, 0) == "1:1"


def test_image_model_version_falls_back_to_default():
    assert image_model_version("4.5") == "4.5"
    assert image_model_version("garbage") == "4.0"
    # image2image 不支持 3.0 → 回退默认
    assert image_model_version("3.0", "image2image") == "4.0"


def test_image_resolution_rules():
    assert image_resolution("4.0", 1024, 1024) == "2k"
    assert image_resolution("4.0", 4096, 4096) == "4k"
    assert image_resolution("3.0", 512, 512) == "2k"       # 3.0 无 4k


# ---- generate 流程(mock _run,真实临时文件) ----------------------------

@pytest.fixture
def provider(monkeypatch):
    p = DreaminaProvider(poll_seconds=1)
    # 绕过 is_installed:_run 被整体替换,不会真的 spawn。
    return p


def _two_step_fake(download_filename: str, *, submit_json='{"submit_id": "s-1"}'):
    """真机契约:submit(text2image/image2image)只返 submit_id,不带 --download_dir;
    query_result 才把媒体下载到 --download_dir。"""
    captured = {"submit": None, "query": None}

    async def fake_run(args, *, timeout=120, cwd=None, stdin_data=None, env=None):
        from pathlib import Path

        first = str(args[0])
        if first in ("text2image", "image2image"):
            captured["submit"] = list(args)
            assert not any(str(a).startswith("--download_dir") for a in args), "submit 不该带 download_dir"
            return (0, submit_json, "")
        if first == "query_result":
            captured["query"] = list(args)
            dl = next(a.split("=", 1)[1] for a in args if str(a).startswith("--download_dir="))
            (Path(dl) / download_filename).write_bytes(b"\x89PNG fake")
            return (0, '{"gen_status": "done"}', "")
        return (0, "{}", "")

    return fake_run, captured


async def test_generate_text2image_two_step(provider, monkeypatch):
    fake, captured = _two_step_fake("result_0.png")
    monkeypatch.setattr(provider, "_run", fake)
    result = await provider.generate(ExternalGenRequest(prompt="一只猫", width=1920, height=1080))
    assert len(result.artifacts) == 1
    assert result.artifacts[0].local_path.endswith("result_0.png")
    # submit 形态:text2image + ratio 由尺寸推导,且无 download_dir
    assert "text2image" in captured["submit"]
    assert "--ratio=16:9" in captured["submit"]
    # 下载走 query_result --submit_id=s-1
    assert "query_result" in captured["query"]
    assert "--submit_id=s-1" in captured["query"]


async def test_generate_image2image_passes_images(provider, monkeypatch):
    fake, captured = _two_step_fake("edit.jpg", submit_json='{"submit_id": "s-2"}')
    monkeypatch.setattr(provider, "_run", fake)
    result = await provider.generate(
        ExternalGenRequest(prompt="改成夜景", input_images=["/tmp/ref.png"])
    )
    assert result.artifacts[0].title == "edit.jpg"
    assert "image2image" in captured["submit"]
    assert "--images" in captured["submit"] and "/tmp/ref.png" in captured["submit"]


async def test_generate_raises_on_submit_failure(provider, monkeypatch):
    async def fake_run(args, *, timeout=120, cwd=None, stdin_data=None, env=None):
        return (0, '{"gen_status": "failed", "fail_reason": "敏感词"}', "")

    monkeypatch.setattr(provider, "_run", fake_run)
    with pytest.raises(ProviderError) as exc:
        await provider.generate(ExternalGenRequest(prompt="x"))
    assert "敏感词" in exc.value.message


async def test_generate_raises_when_no_submit_id(provider, monkeypatch):
    async def fake_run(args, *, timeout=120, cwd=None, stdin_data=None, env=None):
        return (0, '{"queue_info": {}}', "")  # 无 submit_id

    monkeypatch.setattr(provider, "_run", fake_run)
    with pytest.raises(ProviderError) as exc:
        await provider.generate(ExternalGenRequest(prompt="x"))
    assert "submit_id" in exc.value.message


async def test_generate_pending_when_query_no_media(provider, monkeypatch):
    async def fake_run(args, *, timeout=120, cwd=None, stdin_data=None, env=None):
        first = str(args[0])
        if first in ("text2image", "image2image"):
            return (0, '{"submit_id": "s-9"}', "")
        return (0, '{"gen_status": "pending"}', "")  # query_result 没下到媒体

    monkeypatch.setattr(provider, "_run", fake_run)
    with pytest.raises(ProviderError) as exc:
        await provider.generate(ExternalGenRequest(prompt="x"))
    assert "s-9" in exc.value.message


async def test_generate_raises_on_nonzero_exit(provider, monkeypatch):
    async def fake_run(args, *, timeout=120, cwd=None, stdin_data=None, env=None):
        return (1, "", "boom")

    monkeypatch.setattr(provider, "_run", fake_run)
    with pytest.raises(ProviderError):
        await provider.generate(ExternalGenRequest(prompt="x"))
