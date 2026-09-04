"""模型级 GPU 组(张量并行)—— 拓扑解析 / 自动选组 / vLLM 落卡 / 覆盖 round-trip / API。

背景:机器上 cuda:0 与 cuda:2 是一对 NVLink 直连的 3090(24G × 2),cuda:1 是
RTX PRO 6000(96G)。老逻辑「装不下单卡 → tp = len(gpu_stats)」会把三张异构卡
一起拉进张量并行 —— 按最小卡算白扔 Pro 6000 的显存,实际必炸。这里把新规则钉死:
**只在同型号卡之间组队,NVLink 优先,组不成就退单卡并报清楚原因**。

CI 安全:全程 mock,**绝不真起 vLLM / 碰 GPU**(conftest 另有 Popen 护栏)。
"""
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.gpu.topology import (
    candidate_groups,
    group_is_nvlinked,
    parse_topo_matrix,
    select_tp_group,
)

# ---------------------------------------------------------------------------
# 测试用的机器画像:0/2 = 3090(NVLink 对),1 = PRO 6000
# ---------------------------------------------------------------------------
_NAMES = {
    0: "NVIDIA GeForce RTX 3090",
    1: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    2: "NVIDIA GeForce RTX 3090",
}
_PAIRS = {frozenset((0, 2))}


def _stats(free_by_idx: dict[int, int], total_by_idx: dict[int, int] | None = None):
    total_by_idx = total_by_idx or {0: 24576, 1: 98304, 2: 24576}
    return [
        {"index": i, "free_mb": free_by_idx[i], "total_mb": total_by_idx[i]}
        for i in sorted(free_by_idx)
    ]


# ---------------------------------------------------------------------------
# nvidia-smi topo -m 解析
# ---------------------------------------------------------------------------

_TOPO = """\t\x1b[4mGPU0\tGPU1\tGPU2\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m
GPU0\t X \tNODE\tNV4\t0-47\t0\t\tN/A
GPU1\tNODE\t X \tPHB\t0-47\t0\t\tN/A
GPU2\tNV4\tPHB\t X \t0-47\t0\t\tN/A

Legend:
  X    = Self
"""


def test_parse_topo_matrix_finds_nvlink_pair():
    assert parse_topo_matrix(_TOPO) == {frozenset((0, 2))}


def test_parse_topo_matrix_ignores_pcie_links():
    # NODE / PHB 都不是 NVLink —— 0-1 / 1-2 不能被当成直连
    pairs = parse_topo_matrix(_TOPO)
    assert frozenset((0, 1)) not in pairs
    assert frozenset((1, 2)) not in pairs


def test_parse_topo_matrix_survives_garbage():
    assert parse_topo_matrix("") == set()
    assert parse_topo_matrix("no gpus here\n") == set()


def test_group_is_nvlinked_requires_every_pair():
    assert group_is_nvlinked([0, 2], _PAIRS) is True
    assert group_is_nvlinked([0, 1], _PAIRS) is False
    # 三卡组:0-1 / 1-2 无链路 → 整组不算 NVLink
    assert group_is_nvlinked([0, 1, 2], _PAIRS) is False
    assert group_is_nvlinked([0], _PAIRS) is False


# ---------------------------------------------------------------------------
# select_tp_group:同型号 / NVLink 优先 / 装不下就不成组
# ---------------------------------------------------------------------------

def test_select_never_mixes_heterogeneous_cards():
    """三张卡两种型号 → 只可能选出那对 3090,绝不把 PRO 6000 拉进来。"""
    stats = _stats({0: 24000, 1: 96000, 2: 24000})
    picked = select_tp_group(stats, model_size_gb=30.0, names=_NAMES, pairs=_PAIRS)
    assert picked == [0, 2]
    assert 1 not in picked


def test_select_prefers_nvlinked_group():
    """四张同型号卡,只有 0-2 有 NVLink → 选 0-2 而不是 index 更小的 0-1。"""
    names = {i: "NVIDIA GeForce RTX 3090" for i in range(4)}
    stats = _stats(
        {0: 24000, 1: 24000, 2: 24000, 3: 24000},
        {0: 24576, 1: 24576, 2: 24576, 3: 24576},
    )
    picked = select_tp_group(stats, model_size_gb=30.0, names=names, pairs={frozenset((0, 2))})
    assert picked == [0, 2]


def test_select_returns_empty_when_group_cannot_fit():
    """两张 3090 加起来也装不下 → 不成组(调用方退单卡 + 报错说明)。"""
    stats = _stats({0: 24000, 1: 96000, 2: 24000})
    assert select_tp_group(stats, model_size_gb=80.0, names=_NAMES, pairs=_PAIRS) == []


def test_select_uses_min_free_not_sum():
    """组内一张卡被占了大半 → 分片装不下那张,整组作废(不能按 sum 算)。"""
    stats = _stats({0: 24000, 1: 96000, 2: 4000})
    assert select_tp_group(stats, model_size_gb=26.0, names=_NAMES, pairs=_PAIRS) == []


def test_select_exact_size_filters_group_size():
    names = {i: "NVIDIA GeForce RTX 3090" for i in range(4)}
    stats = _stats(
        {0: 24000, 1: 24000, 2: 24000, 3: 24000},
        {0: 24576, 1: 24576, 2: 24576, 3: 24576},
    )
    picked = select_tp_group(
        stats, model_size_gb=30.0, names=names, pairs=set(), exact_size=4
    )
    assert picked == [0, 1, 2, 3]


def test_select_single_card_of_a_model_never_groups():
    """机器上只有一张 PRO 6000 → 它不可能自己跟自己组队。"""
    stats = _stats({1: 96000}, {1: 98304})
    assert select_tp_group(stats, model_size_gb=90.0, names=_NAMES, pairs=_PAIRS) == []


# ---------------------------------------------------------------------------
# candidate_groups:给前端的候选清单
# ---------------------------------------------------------------------------

def test_candidate_groups_only_lists_same_model_combos():
    groups = candidate_groups(_stats({0: 24000, 1: 96000, 2: 24000}), names=_NAMES, pairs=_PAIRS)
    assert [g["gpus"] for g in groups] == [[0, 2]]
    g = groups[0]
    assert g["nvlink"] is True
    assert g["name"] == "NVIDIA GeForce RTX 3090"
    assert g["total_gb"] == pytest.approx(48.0, abs=0.2)


def test_candidate_groups_sorts_nvlink_first():
    names = {i: "NVIDIA GeForce RTX 3090" for i in range(3)}
    groups = candidate_groups(
        _stats({0: 24000, 1: 24000, 2: 24000}, {0: 24576, 1: 24576, 2: 24576}),
        names=names,
        pairs={frozenset((0, 2))},
    )
    assert groups[0]["gpus"] == [0, 2] and groups[0]["nvlink"] is True
    assert all(len(g["gpus"]) >= 2 for g in groups)


# ---------------------------------------------------------------------------
# VLLMAdapter:显式 gpus → CUDA_VISIBLE_DEVICES + --tensor-parallel-size
# ---------------------------------------------------------------------------

def _dead_popen():
    """假 Popen:已退出、returncode=1 —— 让 load() 走「子进程起不来」分支后抛错。
    绝不真起 vLLM。"""
    proc = MagicMock()
    proc.poll.return_value = 1
    proc.returncode = 1
    proc.stdout = io.StringIO("boom\n")
    proc.pid = 424242
    return proc


async def _capture_launch(adapter, auto: dict, device: str | None = None):
    """跑一次 load(),返回 (cmd, env) —— 子进程是假的,必抛 RuntimeError。"""
    captured: dict = {}

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env") or {}
        return _dead_popen()

    with patch.object(adapter, "_auto_configure", return_value=auto), \
            patch.object(adapter._client, "get", new_callable=AsyncMock,
                         side_effect=Exception("no server")), \
            patch("subprocess.Popen", side_effect=_fake_popen), \
            patch.object(adapter, "_kill_process"):
        with pytest.raises(RuntimeError):
            await adapter.load(device)
    return captured["cmd"], captured["env"]


def _auto(**over):
    base = {
        "port": 19999, "tp": 2, "gpus": [0, 2], "max_model_len": 4096,
        "utilization": 0.85, "quantization": None, "dtype": None,
        "max_num_seqs": 32, "gpu_idx": 0, "model_size_gb": 30.0,
        "gpu_total_gb": 24.0, "gpu_free_gb": 23.0,
        "is_multimodal": False, "is_audio": False,
    }
    base.update(over)
    return base


async def test_explicit_gpus_pin_cuda_visible_devices_and_tp(tmp_path):
    """`gpus=[0, 2]` → 子进程 env 只看得见这两张卡,且 --tensor-parallel-size 2。"""
    from src.services.inference.llm_vllm import VLLMAdapter

    a = VLLMAdapter(paths={"main": str(tmp_path)}, vllm_port=19999, gpus=[0, 2])
    cmd, env = await _capture_launch(a, _auto(), device="cuda:0")

    assert env["CUDA_VISIBLE_DEVICES"] == "0,2"
    assert "--tensor-parallel-size" in cmd
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2"
    assert a.gpu_indices == [0, 2]


async def test_auto_selected_group_also_pins_visible_devices(tmp_path):
    """自动选出来的组(adapter 没配 gpus)同样要钉 CUDA_VISIBLE_DEVICES —— 老代码
    在 tp>1 时什么都不设,子进程继承父进程环境看到全部三张卡。"""
    from src.services.inference.llm_vllm import VLLMAdapter

    a = VLLMAdapter(paths={"main": str(tmp_path)}, vllm_port=19999)
    cmd, env = await _capture_launch(a, _auto(), device="cuda:0")
    assert env["CUDA_VISIBLE_DEVICES"] == "0,2"


async def test_single_card_still_pins_one_index(tmp_path):
    from src.services.inference.llm_vllm import VLLMAdapter

    a = VLLMAdapter(paths={"main": str(tmp_path)}, vllm_port=19999)
    cmd, env = await _capture_launch(a, _auto(tp=1, gpus=None, gpu_idx=1), device="cuda:1")
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert "--tensor-parallel-size" not in cmd


async def test_explicit_tp_narrows_group(tmp_path):
    """显式 tensor_parallel_size 只能收窄组(取前 N 张),不能超过组内卡数。"""
    from src.services.inference.llm_vllm import VLLMAdapter

    a = VLLMAdapter(paths={"main": str(tmp_path)}, vllm_port=19999,
                    gpus=[0, 2], tensor_parallel_size=1)
    cmd, env = await _capture_launch(a, _auto(tp=1, gpus=[0]), device="cuda:0")
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "--tensor-parallel-size" not in cmd


# ---------------------------------------------------------------------------
# _auto_configure:自动 tp 不再 len(gpu_stats)
# ---------------------------------------------------------------------------

def _configure(adapter, stats, model_size_gb, device=None, tmp_path=None):
    """跑 _auto_configure,mock 掉磁盘 & nvidia-smi & 拓扑。"""
    with patch("src.services.gpu_monitor.poll_gpu_stats", return_value=stats), \
            patch("src.gpu.topology.gpu_names", return_value=_NAMES), \
            patch("src.gpu.topology.nvlink_pairs", return_value=_PAIRS), \
            patch.object(type(adapter), "_auto_configure",
                         type(adapter)._auto_configure):
        # 模型大小从 safetensors 体积算 —— 造一个占位文件到 tmp_path
        f = tmp_path / "model-00001.safetensors"
        f.write_bytes(b"\0" * 1024)
        with patch("pathlib.Path.glob") as g:
            fake = MagicMock()
            fake.stat.return_value = MagicMock(st_size=int(model_size_gb * 1024**3))
            g.side_effect = lambda pat: [fake] if "safetensors" in pat else []
            return adapter._auto_configure(device)


async def test_auto_tp_picks_homogeneous_pair_not_all_cards(tmp_path):
    """30G 模型装不下任一张 3090 → 选 [0, 2] 做 tp=2,**不是** tp=3(混上 PRO 6000)。"""
    from src.services.inference.llm_vllm import VLLMAdapter

    a = VLLMAdapter(paths={"main": str(tmp_path)}, vllm_port=19999)
    # 让 PRO 6000 也装不下(free 压到 10G),否则单卡就够了不会触发组队
    stats = _stats({0: 20000, 1: 10000, 2: 20000})
    auto = _configure(a, stats, model_size_gb=30.0, device="cuda:0", tmp_path=tmp_path)
    assert auto["tp"] == 2
    assert auto["gpus"] == [0, 2]


async def test_auto_tp_falls_back_to_single_card_when_no_group_fits(tmp_path):
    """没有同型号组能装下 → tp=1 退单卡最大者,而不是硬凑异构组。"""
    from src.services.inference.llm_vllm import VLLMAdapter

    a = VLLMAdapter(paths={"main": str(tmp_path)}, vllm_port=19999)
    stats = _stats({0: 8000, 1: 90000, 2: 8000})
    auto = _configure(a, stats, model_size_gb=60.0, device="cuda:0", tmp_path=tmp_path)
    assert auto["tp"] == 1
    assert auto["gpus"] is None
    assert auto["gpu_idx"] == 1  # 单卡最大者


async def test_explicit_group_budget_uses_min_of_group(tmp_path):
    """显式组的显存预算按组内**最小** free/total 算(TP 均分,最小卡先炸)。"""
    from src.services.inference.llm_vllm import VLLMAdapter

    a = VLLMAdapter(paths={"main": str(tmp_path)}, vllm_port=19999, gpus=[0, 2])
    stats = _stats({0: 20000, 1: 90000, 2: 12000})
    auto = _configure(a, stats, model_size_gb=30.0, device="cuda:0", tmp_path=tmp_path)
    assert auto["gpus"] == [0, 2]
    assert auto["tp"] == 2
    assert auto["gpu_free_gb"] == pytest.approx(12000 / 1024, abs=0.1)


# ---------------------------------------------------------------------------
# 运行时覆盖 round-trip(DB → 缓存 → ModelSpec)
# ---------------------------------------------------------------------------

async def test_override_gpus_round_trip(db_session):
    from src.models.model_runtime_override import ModelRuntimeOverride
    from src.services import runtime_override_store as store

    store.reset_cache()
    await store.set_override(db_session, "my-llm", "gpus", [0, 2])
    assert store.get_overrides()["my-llm"]["gpus"] == [0, 2]

    row = await db_session.get(ModelRuntimeOverride, "my-llm")
    assert row.gpus == [0, 2]
    assert row.to_overrides()["gpus"] == [0, 2]


async def test_setting_single_gpu_clears_the_group(db_session):
    """点「GPU 1」= 退出组;不清的话组仍优先,用户的点击等于没生效。"""
    from src.services import runtime_override_store as store

    store.reset_cache()
    await store.set_override(db_session, "my-llm", "gpus", [0, 2])
    await store.set_override(db_session, "my-llm", "gpu", 1)
    ov = store.get_overrides()["my-llm"]
    assert ov["gpu"] == 1
    assert "gpus" not in ov


async def test_override_gpus_none_clears(db_session):
    from src.services import runtime_override_store as store

    store.reset_cache()
    await store.set_override(db_session, "my-llm", "gpus", [0, 2])
    await store.set_override(db_session, "my-llm", "gpus", None)
    assert "gpus" not in store.get_overrides().get("my-llm", {})


def test_model_spec_carries_gpus():
    from src.services.inference.registry import ModelSpec

    spec = ModelSpec(id="m", model_type="llm", adapter_class="x.Y", paths={"main": "m"},
                     vram_mb=1024, gpus=[0, 2])
    assert spec.gpus == [0, 2]


# ---------------------------------------------------------------------------
# model_manager:spec.gpus → adapter kwarg + gpu_indices
# ---------------------------------------------------------------------------

def test_instantiate_adapter_injects_gpus(monkeypatch):
    from src.services.inference.registry import ModelSpec
    from src.services.model_manager import ModelManager

    spec = ModelSpec(
        id="m", model_type="llm",
        adapter_class="src.services.inference.llm_vllm.VLLMAdapter",
        paths={"main": "m"}, vram_mb=1024, gpus=[0, 2],
    )
    mgr = ModelManager.__new__(ModelManager)
    monkeypatch.setattr("src.config.load_runtime_overrides", lambda: {})
    adapter = mgr._instantiate_adapter(spec)
    assert adapter._gpus == [0, 2]


def test_instantiate_adapter_skips_gpus_for_single_card():
    from src.services.inference.registry import ModelSpec
    from src.services.model_manager import ModelManager

    spec = ModelSpec(
        id="m", model_type="llm",
        adapter_class="src.services.inference.llm_vllm.VLLMAdapter",
        paths={"main": "m"}, vram_mb=1024, gpu=1,
    )
    mgr = ModelManager.__new__(ModelManager)
    adapter = mgr._instantiate_adapter(spec)
    assert adapter._gpus is None


# ---------------------------------------------------------------------------
# allocator:组预留 / 释放
# ---------------------------------------------------------------------------

def test_allocator_group_reservation_blocks_concurrent_picks():
    from src.services.gpu_allocator import GPUAllocator

    stats = _stats({0: 20000, 1: 20000, 2: 20000})
    alloc = GPUAllocator(poll_fn=lambda: stats, hardware_config={})
    alloc.reserve_gpus([0, 2], 15000)
    # 0/2 各被预留 15G → 只剩 5G,一个 10G 的模型只能落到 1
    assert alloc.get_best_gpu(10000, reserve=False) == 1
    alloc.release_gpus([0, 2], 15000)
    assert alloc._pending == {}


# ---------------------------------------------------------------------------
# vllm_scanner:识别多卡进程
# ---------------------------------------------------------------------------

def test_scanner_reports_tensor_parallel_size():
    from src.services.inference import vllm_scanner

    ps = (
        "  123 python -m vllm.entrypoints.openai.api_server --model /models/big "
        "--port 8100 --tensor-parallel-size 2\n"
        "  124 python -m vllm.entrypoints.openai.api_server --model /models/small "
        "--port 8101\n"
    )
    with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout=ps)), \
            patch("httpx.get", side_effect=Exception("down")):
        found = {c["port"]: c for c in vllm_scanner.scan_running_vllm()}
    assert found[8100]["tensor_parallel_size"] == 2
    assert found[8101]["tensor_parallel_size"] == 1


# ---------------------------------------------------------------------------
# detector:组里每张卡都过 display-GPU 检查
# ---------------------------------------------------------------------------

def test_get_device_for_engine_warns_for_every_display_card_in_group(caplog):
    from src.gpu import detector

    fake = [detector.GPUInfo(index=i, name="RTX 3090", vram_total_gb=24.0,
                             compute_capability=(8, 6)) for i in range(3)]
    with patch.object(detector, "get_gpus", return_value=fake), \
            patch.object(detector, "get_display_gpu_indices", return_value={2}):
        with caplog.at_level("WARNING"):
            dev = detector.get_device_for_engine({"gpus": [0, 2]})
    assert dev == "cuda:0"  # 主卡 = 组内第一张
    assert "GPU group" in caplog.text and "[2]" in caplog.text


def test_get_device_for_engine_group_beats_single_gpu():
    from src.gpu import detector

    fake = [detector.GPUInfo(index=i, name="RTX 3090", vram_total_gb=24.0,
                             compute_capability=(8, 6)) for i in range(3)]
    with patch.object(detector, "get_gpus", return_value=fake), \
            patch.object(detector, "get_display_gpu_indices", return_value=set()):
        assert detector.get_device_for_engine({"gpu": 1, "gpus": [0, 2]}) == "cuda:0"


# ---------------------------------------------------------------------------
# API:GET /api/v1/gpu/groups + PATCH /api/v1/engines/{name}/gpu
# ---------------------------------------------------------------------------

async def test_gpu_groups_endpoint(db_client):
    with patch("src.services.gpu_monitor.poll_gpu_stats",
               return_value=_stats({0: 24000, 1: 96000, 2: 24000})), \
            patch("src.gpu.topology.gpu_names", return_value=_NAMES), \
            patch("src.gpu.topology.nvlink_pairs", return_value=_PAIRS):
        resp = await db_client.get("/api/v1/gpu/groups")
    assert resp.status_code == 200
    groups = resp.json()["groups"]
    assert [g["gpus"] for g in groups] == [[0, 2]]
    assert groups[0]["nvlink"] is True


def _first_engine_name() -> str:
    from src.config import load_model_configs
    return next(iter(load_model_configs()))


async def test_patch_gpu_accepts_group(db_client):
    from src.services import runtime_override_store as store

    name = _first_engine_name()
    store.reset_cache()
    with patch("src.gpu.topology.gpu_names", return_value=_NAMES), \
            patch("src.gpu.topology.nvlink_pairs", return_value=_PAIRS):
        resp = await db_client.patch(f"/api/v1/engines/{name}/gpu", json={"gpus": [2, 0]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["gpus"] == [0, 2]  # 排序归一
    assert body["gpu"] == 0        # 主卡 = 组内第一张
    assert store.get_overrides()[name]["gpus"] == [0, 2]


async def test_patch_gpu_rejects_heterogeneous_group(db_client):
    name = _first_engine_name()
    with patch("src.gpu.topology.gpu_names", return_value=_NAMES), \
            patch("src.gpu.topology.nvlink_pairs", return_value=_PAIRS):
        resp = await db_client.patch(f"/api/v1/engines/{name}/gpu", json={"gpus": [0, 1]})
    assert resp.status_code == 400
    assert "同型号" in resp.text


async def test_patch_gpu_rejects_one_card_group(db_client):
    name = _first_engine_name()
    with patch("src.gpu.topology.gpu_names", return_value=_NAMES):
        resp = await db_client.patch(f"/api/v1/engines/{name}/gpu", json={"gpus": [0, 0]})
    assert resp.status_code == 400


async def test_patch_gpu_single_card_still_works(db_client):
    """老路径不能坏:`?gpu=1` 查询参数照旧,且会清掉已有的组。"""
    from src.services import runtime_override_store as store

    name = _first_engine_name()
    store.reset_cache()
    with patch("src.gpu.topology.gpu_names", return_value=_NAMES), \
            patch("src.gpu.topology.nvlink_pairs", return_value=_PAIRS):
        await db_client.patch(f"/api/v1/engines/{name}/gpu", json={"gpus": [0, 2]})
        resp = await db_client.patch(f"/api/v1/engines/{name}/gpu?gpu=1")
    assert resp.status_code == 200
    assert resp.json() == {"name": name, "gpu": 1, "gpus": None, "applied": False,
                           "hint": "需重新加载模型生效(unload + load)"}
    assert "gpus" not in store.get_overrides()[name]
