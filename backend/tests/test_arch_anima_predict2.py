"""arch_anima.predict2 wiring 测(CI 可跑;不真 forward —— conftest mock torch)。

CI 跑:验 module symbol 表 + 公共 API 对外。真 forward / 出图等到
`tests/manual/smoke_anima_pr1.py`(或后续 PR-anima-7 真模型 e2e)。

CLAUDE.md「conftest mock torch + 无 GPU,引擎正确性只靠 standalone smoke」。
"""
from __future__ import annotations


def test_module_files_exist():
    """arch_anima 模块文件齐全(grep 风,不触发 import / metaclass)。

    conftest mock torch + einops Rearrange 子类化 nn.Module → 真 import 会 metaclass 冲突。
    所以只验 path 上文件存在;真 forward 走 `tests/manual/smoke_anima_pr*.py`。
    """
    import pathlib  # noqa: PLC0415

    base = pathlib.Path(__file__).parent.parent / "src/services/inference/arch_anima"
    assert (base / "__init__.py").exists()
    assert (base / "predict2.py").exists()
    assert (base / "position_embedding.py").exists()
    assert (base / "anima.py").exists()  # PR-anima-2
    assert (base / "text_encoder.py").exists()  # PR-anima-3
    assert (base / "load.py").exists()  # PR-anima-4


def test_source_layout():
    """检查源文件含必需的 class 定义(grep 风,不触发 import / metaclass)。"""
    import pathlib  # noqa: PLC0415

    base = pathlib.Path(__file__).parent.parent / "src/services/inference/arch_anima"
    predict2_src = (base / "predict2.py").read_text()
    pos_src = (base / "position_embedding.py").read_text()
    anima_src = (base / "anima.py").read_text()

    for sym in [
        "class GPT2FeedForward",
        "class Attention",
        "class Timesteps",
        "class TimestepEmbedding",
        "class PatchEmbed",
        "class FinalLayer",
        "class Block",
        "class MiniTrainDIT",
        "def apply_rotary_pos_emb",
        "def _scaled_dot_product_attention",
        "def _pad_to_patch_size",
    ]:
        assert sym in predict2_src, f"predict2.py missing {sym!r}"

    for sym in [
        "class VideoPositionEmb",
        "class VideoRopePosition3DEmb",
        "class LearnablePosEmbAxis",
        "def normalize",
    ]:
        assert sym in pos_src, f"position_embedding.py missing {sym!r}"

    # PR-anima-2:Anima 主类 + LLMAdapter + 1D RoPE 路径。
    for sym in [
        "class Anima",
        "class LLMAdapter",
        "class RotaryEmbedding",
        "class _AnimaAttention",
        "class _AnimaTransformerBlock",
        "def _rotate_half",
        "def _apply_llm_rope",
    ]:
        assert sym in anima_src, f"anima.py missing {sym!r}"

    # PR-anima-3:AnimaTextEncoder wrapper(qwen3-0.6b + 可选 t5xxl)。
    te_src = (base / "text_encoder.py").read_text()
    for sym in [
        "class AnimaTextEncoder",
        "def load",
        "def encode",
        "def unload",
    ]:
        assert sym in te_src, f"text_encoder.py missing {sym!r}"

    # PR-anima-4:权重加载器 + anima-base-v1.0 config(strip 'net.' prefix)。
    load_src = (base / "load.py").read_text()
    for sym in [
        "ANIMA_BASE_V1_CONFIG",
        "def load_anima_dit_from_single_file",
        'k[4:]',  # strip 'net.' prefix 是关键
    ]:
        assert sym in load_src, f"load.py missing {sym!r}"


def test_bundle_config_present():
    """PR-anima-5:Qwen3-0.6B-Base config bundled 进 repo,免运行时联网。

    config 是 1.7KB JSON,极小;tokenizer(qwen25_tokenizer/ + t5_tokenizer/ 共 6.7M)
    暂不 bundle,留运行时通过 env var / setting 指定。
    """
    import json  # noqa: PLC0415
    import pathlib  # noqa: PLC0415

    bundle = (
        pathlib.Path(__file__).parent.parent
        / "configs/image_arch/anima/qwen3_06b_base_config.json"
    )
    assert bundle.exists(), f"qwen3 bundle config missing: {bundle}"

    with bundle.open() as fh:
        cfg = json.load(fh)
    # 关键字段验证 — 真 Qwen3-0.6B-Base 必备。
    assert cfg.get("model_type") == "qwen3"
    assert cfg.get("hidden_size") == 1024
    assert cfg.get("num_hidden_layers") == 28
    assert cfg.get("vocab_size") == 151936  # Qwen3 默认 vocab
    assert cfg.get("rms_norm_eps") == 1e-06


def test_no_comfy_imports_left():
    """port 后不该再有 comfy.* import(spec 2026-05-26-anima-port-design 决策点 = 选项 A 自包含)。"""
    import pathlib  # noqa: PLC0415

    base = pathlib.Path(__file__).parent.parent / "src/services/inference/arch_anima"
    for f in base.glob("*.py"):
        for line in f.read_text().splitlines():
            stripped = line.strip()
            # 只检 import 行,不查注释里关于「替代 torchvision」的说明文。
            if stripped.startswith("#"):
                continue
            assert "import comfy" not in stripped, f"{f.name}: 仍有 comfy import:{line!r}"
            assert "from comfy" not in stripped, f"{f.name}: 仍有 from comfy:{line!r}"
            assert "import torchvision" not in stripped, f"{f.name}: 仍 import torchvision:{line!r}"
            assert "from torchvision" not in stripped, f"{f.name}: 仍 from torchvision:{line!r}"


def test_no_transformer_options_in_signature():
    """port 删了 ComfyUI 特有的 transformer_options kwarg(nous 走 diffusers LoRA loader)。

    docstring / 注释里说「删了 transformer_options」是 OK 的;只检代码体不出现 kwarg。
    """
    import pathlib  # noqa: PLC0415

    base = pathlib.Path(__file__).parent.parent / "src/services/inference/arch_anima"
    for f in base.glob("*.py"):
        for line in f.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # docstring / quoted 也跳过(简易判断:行内含成对反引号 = 文档说明)
            if "`transformer_options`" in stripped:
                continue
            assert "transformer_options" not in stripped, (
                f"{f.name}: transformer_options 是 ComfyUI 特有 kwarg,nous port 应删干净:{line!r}"
            )
