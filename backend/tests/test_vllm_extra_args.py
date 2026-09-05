"""`params.vllm_args` —— 模型 yaml 透传任意 vLLM 长参数。

CI 安全:全程 mock,**绝不真起 vLLM / 碰 GPU**(conftest 另有 Popen 护栏)。
唯一走 load() 的用例用假 Popen(已退出、returncode=1),验的是最终 argv,不是子进程。
"""
import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from src.services.inference.llm_vllm import (
    VLLMAdapter,
    merge_vllm_args,
    normalize_vllm_flag,
    render_vllm_args,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# ---------------------------------------------------------------------------
# 渲染规则
# ---------------------------------------------------------------------------

def test_render_scalar_becomes_flag_and_value():
    assert render_vllm_args({"reasoning-parser": "qwen3"}) == ["--reasoning-parser", "qwen3"]
    assert render_vllm_args({"num-scheduler-steps": 8}) == ["--num-scheduler-steps", "8"]
    assert render_vllm_args({"swap-space": 2.5}) == ["--swap-space", "2.5"]


def test_render_bool_true_is_bare_flag_and_false_is_dropped():
    """store_true 类参数不吃值:True → 只有 flag;False → 完全不出现
    (`--flag false` 会被 vLLM 当成位置参数/报错)。"""
    assert render_vllm_args({"enforce-eager": True}) == ["--enforce-eager"]
    assert render_vllm_args({"enforce-eager": False}) == []


def test_render_dict_is_single_json_argument():
    argv = render_vllm_args({"speculative-config": {"method": "mtp", "num_speculative_tokens": 3}})
    assert argv[0] == "--speculative-config"
    assert len(argv) == 2                       # JSON 是**一个**参数值,不是被拆开的
    assert json.loads(argv[1]) == {"method": "mtp", "num_speculative_tokens": 3}


def test_render_list_is_single_json_argument():
    argv = render_vllm_args({"middleware": ["a", "b"]})
    assert argv == ["--middleware", '["a","b"]']
    assert json.loads(argv[1]) == ["a", "b"]


def test_underscore_and_dash_and_leading_dashes_all_normalize():
    assert normalize_vllm_flag("speculative_config") == "--speculative-config"
    assert normalize_vllm_flag("speculative-config") == "--speculative-config"
    assert normalize_vllm_flag("--speculative-config") == "--speculative-config"
    assert render_vllm_args({"reasoning_parser": "qwen3"}) == ["--reasoning-parser", "qwen3"]


def test_render_none_value_is_skipped():
    assert render_vllm_args({"dtype": None}) == []


def test_render_rejects_unsupported_value_type():
    with pytest.raises(ValueError, match="不支持"):
        render_vllm_args({"foo": object()})


def test_render_rejects_non_dict():
    with pytest.raises(ValueError, match="必须是 dict"):
        render_vllm_args(["--foo", "bar"])


# ---------------------------------------------------------------------------
# 安全边界:--model / --port / --device 一律拒绝(不是覆盖)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["model", "--model", "port", "device", "--device"])
def test_forbidden_keys_raise(key):
    with pytest.raises(ValueError, match="vllm_args 不接受"):
        render_vllm_args({key: "whatever"})


def test_forbidden_key_fails_at_construction_not_at_load(tmp_path):
    """构造期就炸 —— 别等 load() 预留完显存、起了子进程才发现 yaml 写错。"""
    with pytest.raises(ValueError, match="--model"):
        VLLMAdapter(paths={"main": str(tmp_path)}, vllm_port=19999,
                    vllm_args={"model": "/etc/passwd"})


def test_forbidden_device_message_names_placement(tmp_path):
    with pytest.raises(ValueError, match="_placement"):
        VLLMAdapter(paths={"main": str(tmp_path)}, vllm_port=19999,
                    vllm_args={"device": "cpu"})


# ---------------------------------------------------------------------------
# 冲突:适配器自己拼的参数被 vllm_args 覆盖(不能同名两次)
# ---------------------------------------------------------------------------

def test_merge_overrides_adapter_flag_and_warns(caplog):
    cmd = ["python", "-m", "vllm...", "--max-model-len", "32768", "--dtype", "auto"]
    with caplog.at_level("WARNING"):
        merged = merge_vllm_args(cmd, ["--max-model-len", "8192"])
    assert merged.count("--max-model-len") == 1
    assert merged[merged.index("--max-model-len") + 1] == "8192"
    assert "--dtype" in merged and merged[merged.index("--dtype") + 1] == "auto"
    assert "--max-model-len" in caplog.text


def test_merge_strips_valueless_adapter_flag():
    """`--enable-prefix-caching` 没有值,摘的时候不能顺手把后面那个 flag 吃掉。"""
    cmd = ["--enable-prefix-caching", "--dtype", "auto"]
    merged = merge_vllm_args(cmd, ["--enable-prefix-caching"])
    assert merged == ["--dtype", "auto", "--enable-prefix-caching"]


def test_merge_appends_non_colliding_args_untouched():
    cmd = ["python", "--max-model-len", "32768"]
    merged = merge_vllm_args(cmd, ["--reasoning-parser", "qwen3"])
    assert merged == ["python", "--max-model-len", "32768", "--reasoning-parser", "qwen3"]


def test_merge_tp_override_warns_about_placement(caplog):
    with caplog.at_level("WARNING"):
        merge_vllm_args(["--tensor-parallel-size", "2"], ["--tensor-parallel-size", "4"])
    assert "_placement" in caplog.text


def test_merge_no_extra_is_identity():
    cmd = ["python", "--max-model-len", "32768"]
    assert merge_vllm_args(cmd, []) is cmd


# ---------------------------------------------------------------------------
# yaml round-trip:models.d → ModelSpec.params → adapter kwargs
# ---------------------------------------------------------------------------

def test_yaml_roundtrip_params_vllm_args_reaches_spec(tmp_path):
    from src.services.inference.registry import ModelRegistry

    ypath = tmp_path / "models.yaml"
    ypath.write_text(yaml.dump({"models": [{
        "id": "m1", "type": "llm",
        "adapter": "src.services.inference.llm_vllm.VLLMAdapter",
        "path": "/m/x", "vram_mb": 0,
        "params": {"vllm_args": {"speculative-config": {"method": "mtp"},
                                 "reasoning-parser": "qwen3"}},
    }]}))
    spec = ModelRegistry(str(ypath)).get("m1")
    assert spec.params["vllm_args"]["reasoning-parser"] == "qwen3"
    assert spec.params["vllm_args"]["speculative-config"] == {"method": "mtp"}


def test_instantiate_adapter_passes_vllm_args_through(tmp_path):
    """manager 把 spec.params 直接 splat 成 adapter kwargs —— 钉住这条链路。"""
    from src.services.inference.registry import ModelSpec
    from src.services.model_manager import ModelManager

    spec = ModelSpec(
        id="m1", model_type="llm",
        adapter_class="src.services.inference.llm_vllm.VLLMAdapter",
        paths={"main": str(tmp_path)}, vram_mb=0,
        params={"vllm_port": 19999,
                "vllm_args": {"reasoning-parser": "qwen3", "enforce-eager": True}},
    )
    adapter = ModelManager._instantiate_adapter(MagicMock(), spec)
    assert isinstance(adapter, VLLMAdapter)
    assert adapter._vllm_extra_argv == ["--reasoning-parser", "qwen3", "--enforce-eager"]


def test_shipped_qwen38_yaml_declares_mtp_and_reasoning_parser():
    """真配置文件:MTP 投机解码 + reasoning-parser 都在,且 tp 与 gpus 组一致。"""
    doc = yaml.safe_load((CONFIGS / "models.d" / "qwen3_8_27b_abliterated_awq.yaml").read_text())
    args = doc["params"]["vllm_args"]
    assert args["speculative-config"] == {"method": "mtp", "num_speculative_tokens": 3}
    assert args["reasoning-parser"] == "qwen3"
    assert doc["gpus"] == [0, 2]
    assert doc["params"]["tensor_parallel_size"] == len(doc["gpus"])
    # 渲染得出来(键名/类型都合法),且 JSON 值可解析
    argv = render_vllm_args(args)
    assert json.loads(argv[argv.index("--speculative-config") + 1])["method"] == "mtp"


# ---------------------------------------------------------------------------
# 端到端 argv:走一次真的 load()(假 Popen),看最终命令行
# ---------------------------------------------------------------------------

def _dead_popen():
    proc = MagicMock()
    proc.poll.return_value = 1
    proc.returncode = 1
    proc.stdout = io.StringIO("boom\n")
    proc.pid = 424242
    return proc


async def _capture_argv(adapter):
    captured: dict = {}

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env") or {}
        return _dead_popen()

    auto_result = {
        "port": 19999, "tp": 1, "max_model_len": 32768,
        "utilization": 0.85, "quantization": None, "dtype": None,
        "max_num_seqs": 32, "gpu_idx": 0, "model_size_gb": 0.0, "cards": [1],
        "gpu_total_gb": 24.0, "gpu_free_gb": 20.0,
    }
    with patch.object(adapter, "_health_check", new_callable=AsyncMock, return_value=False), \
            patch.object(adapter, "_auto_configure", return_value=auto_result), \
            patch("subprocess.Popen", side_effect=_fake_popen), \
            patch.object(adapter, "_kill_process"):
        with pytest.raises(RuntimeError):
            await adapter.load("cuda:1")
    return captured["cmd"], captured["env"]


async def test_launch_argv_carries_valid_json_speculative_config(tmp_path):
    a = VLLMAdapter(
        paths={"main": str(tmp_path)}, vllm_port=19999,
        vllm_args={"speculative_config": {"method": "mtp", "num_speculative_tokens": 3},
                   "reasoning_parser": "qwen3"},
    )
    cmd, env = await _capture_argv(a)
    spec_val = cmd[cmd.index("--speculative-config") + 1]
    assert json.loads(spec_val) == {"method": "mtp", "num_speculative_tokens": 3}
    assert cmd[cmd.index("--reasoning-parser") + 1] == "qwen3"
    # 落卡不受 vllm_args 影响:仍由 _placement 钉住
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert "--device" not in cmd


async def test_launch_argv_has_no_duplicate_flag_when_overriding(tmp_path):
    a = VLLMAdapter(paths={"main": str(tmp_path)}, vllm_port=19999,
                    max_model_len=32768, vllm_args={"max-model-len": 8192})
    cmd, _ = await _capture_argv(a)
    assert cmd.count("--max-model-len") == 1
    assert cmd[cmd.index("--max-model-len") + 1] == "8192"


async def test_no_vllm_args_leaves_argv_unchanged(tmp_path):
    """零回归:没配 vllm_args 的模型命令行一个字节都不变。"""
    plain = VLLMAdapter(paths={"main": str(tmp_path)}, vllm_port=19999)
    cmd_a, _ = await _capture_argv(plain)
    plain2 = VLLMAdapter(paths={"main": str(tmp_path)}, vllm_port=19999, vllm_args={})
    cmd_b, _ = await _capture_argv(plain2)
    assert cmd_a == cmd_b
    assert not any(t.startswith("--speculative") or t.startswith("--reasoning") for t in cmd_a)
