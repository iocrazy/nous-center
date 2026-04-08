import pytest
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
