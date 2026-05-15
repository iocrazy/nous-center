from src.services.gpu_allocator import GPUAllocator

def _fake_stats():
    return [
        {"index": 0, "free_mb": 8000, "total_mb": 24000, "used_mb": 16000},
        {"index": 1, "free_mb": 20000, "total_mb": 24000, "used_mb": 4000},
    ]

def test_pick_best_gpu():
    alloc = GPUAllocator(poll_fn=_fake_stats)
    assert alloc.get_best_gpu(required_vram_mb=4000) == 1

def test_pick_gpu_insufficient():
    alloc = GPUAllocator(poll_fn=_fake_stats)
    assert alloc.get_best_gpu(required_vram_mb=25000) == -1

def test_pick_gpu_no_gpus():
    alloc = GPUAllocator(poll_fn=lambda: [])
    assert alloc.get_best_gpu(required_vram_mb=1000) == -1

def test_get_free_mb():
    alloc = GPUAllocator(poll_fn=_fake_stats)
    assert alloc.get_free_mb(0) == 8000
    assert alloc.get_free_mb(1) == 20000
    assert alloc.get_free_mb(99) == 0


# ---- Lane A: group-aware API ----

_2GPU_CFG = {
    "groups": [
        {"id": "llm-tp", "gpus": [0, 1], "nvlink": True, "role": "llm", "vram_gb": 48},
    ]
}
_3GPU_CFG = {
    "groups": [
        {"id": "image", "gpus": [2], "nvlink": False, "role": "image", "vram_gb": 96},
        {"id": "llm-tp", "gpus": [0, 1], "nvlink": True, "role": "llm", "vram_gb": 48},
        {"id": "tts", "gpus": [3], "nvlink": False, "role": "tts", "vram_gb": 24},
    ]
}


def test_groups_parsed_from_hardware_config():
    alloc = GPUAllocator(poll_fn=_fake_stats, hardware_config=_2GPU_CFG)
    groups = alloc.groups()
    assert len(groups) == 1
    g = groups[0]
    assert g.id == "llm-tp"
    assert g.gpus == [0, 1]
    assert g.nvlink is True
    assert g.role == "llm"
    assert g.vram_gb == 48


def test_runner_count_follows_groups():
    """runner 数 = groups 数量，不写死（spec §3.2）。"""
    assert GPUAllocator(poll_fn=_fake_stats, hardware_config=_2GPU_CFG).runner_count() == 1
    assert GPUAllocator(poll_fn=_fake_stats, hardware_config=_3GPU_CFG).runner_count() == 3


def test_group_for_role():
    alloc = GPUAllocator(poll_fn=_fake_stats, hardware_config=_3GPU_CFG)
    assert alloc.group_for_role("image").id == "image"
    assert alloc.group_for_role("llm").id == "llm-tp"
    assert alloc.group_for_role("tts").id == "tts"
    assert alloc.group_for_role("nonexistent") is None


def test_group_by_id():
    alloc = GPUAllocator(poll_fn=_fake_stats, hardware_config=_3GPU_CFG)
    assert alloc.group_by_id("llm-tp").gpus == [0, 1]
    assert alloc.group_by_id("missing") is None


def test_llm_group_gpus():
    """Lane E 的 vLLM 选卡数据源：role:llm group 的 gpus。"""
    assert GPUAllocator(poll_fn=_fake_stats, hardware_config=_3GPU_CFG).llm_group_gpus() == [0, 1]
    assert GPUAllocator(poll_fn=_fake_stats, hardware_config=_2GPU_CFG).llm_group_gpus() == [0, 1]
    # 无 llm group → 空列表
    img_only = {"groups": [{"id": "image", "gpus": [0], "nvlink": False,
                            "role": "image", "vram_gb": 24}]}
    assert GPUAllocator(poll_fn=_fake_stats, hardware_config=img_only).llm_group_gpus() == []


def test_nvlink_groups_only():
    """tensor-parallel 模型校验用：只返回 nvlink:true 的 group。"""
    alloc = GPUAllocator(poll_fn=_fake_stats, hardware_config=_3GPU_CFG)
    nvlink_ids = {g.id for g in alloc.nvlink_groups()}
    assert nvlink_ids == {"llm-tp"}


def test_empty_hardware_config_falls_back_to_detected_gpus(monkeypatch):
    """hardware.yaml 为空 → 按 detect_gpus() 每卡一个单卡 group。"""
    from src.gpu.detector import GPUInfo

    monkeypatch.setattr(
        "src.services.gpu_allocator.detect_gpus",
        lambda: [
            GPUInfo(index=0, name="RTX 3090", vram_total_gb=24.0, compute_capability=(8, 6)),
            GPUInfo(index=1, name="RTX 3090", vram_total_gb=24.0, compute_capability=(8, 6)),
        ],
    )
    alloc = GPUAllocator(poll_fn=_fake_stats, hardware_config={"groups": []})
    groups = alloc.groups()
    assert len(groups) == 2
    assert {g.id for g in groups} == {"gpu0", "gpu1"}
    assert all(g.nvlink is False for g in groups)
    assert groups[0].gpus == [0]
    assert alloc.runner_count() == 2


def test_empty_config_no_gpus_returns_empty(monkeypatch):
    """hardware.yaml 空 + 无 GPU（CI）→ groups 空，runner_count 0，不抛异常。"""
    monkeypatch.setattr("src.services.gpu_allocator.detect_gpus", lambda: [])
    alloc = GPUAllocator(poll_fn=lambda: [], hardware_config={"groups": []})
    assert alloc.groups() == []
    assert alloc.runner_count() == 0
    assert alloc.group_for_role("llm") is None
    assert alloc.llm_group_gpus() == []
