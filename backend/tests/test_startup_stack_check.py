"""启动预检:推理/图像栈缺失时 banner 大声报(防 2026-07-06 裸 uv sync 裁 venv 事故复发)。

那次 venv 被裁掉 vllm/torch,表现是每次加载才静默 ModuleNotFoundError,启动无任何信号。
此预检在开机 banner 用 find_spec(不 import、不碰 CUDA)查栈,缺了黄字列出 + WARNING。
"""

from src.api.startup_banner import _inference_stack


def test_all_present_ok():
    disp, missing = _inference_stack(find_spec=lambda _m: object())  # 全在
    assert missing == []
    assert "ok" in disp.lower()


def test_missing_listed():
    present = {"torch", "transformers"}
    disp, missing = _inference_stack(
        find_spec=lambda m: object() if m in present else None
    )
    # vllm/diffusers/safetensors 缺
    assert "vllm" in missing
    assert "safetensors" in missing
    assert "torch" not in missing
    assert "uv sync --all-extras" in disp  # 给出修复命令


def test_all_missing():
    disp, missing = _inference_stack(find_spec=lambda _m: None)
    assert "vllm" in missing and "torch" in missing
    assert "uv sync --all-extras" in disp
