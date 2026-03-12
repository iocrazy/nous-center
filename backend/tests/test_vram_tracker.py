from src.gpu.vram_tracker import VRAMTracker


def test_tracker_init():
    tracker = VRAMTracker(gpu_count=2, vram_per_gpu_gb=24)
    assert tracker.get_free(0) == 24.0
    assert tracker.get_free(1) == 24.0


def test_allocate_and_release():
    tracker = VRAMTracker(gpu_count=2, vram_per_gpu_gb=24)

    assert tracker.allocate(gpu=0, model_name="sdxl", vram_gb=10.0) is True
    assert tracker.get_free(0) == 14.0

    assert tracker.allocate(gpu=0, model_name="big_model", vram_gb=20.0) is False
    assert tracker.get_free(0) == 14.0

    tracker.release(gpu=0, model_name="sdxl")
    assert tracker.get_free(0) == 24.0


def test_get_loaded_models():
    tracker = VRAMTracker(gpu_count=2, vram_per_gpu_gb=24)
    tracker.allocate(gpu=0, model_name="sdxl", vram_gb=10.0)
    tracker.allocate(gpu=1, model_name="cosyvoice2", vram_gb=3.0)

    loaded = tracker.get_loaded_models()
    assert loaded[0] == [("sdxl", 10.0)]
    assert loaded[1] == [("cosyvoice2", 3.0)]


def test_release_all():
    tracker = VRAMTracker(gpu_count=2, vram_per_gpu_gb=24)
    tracker.allocate(gpu=0, model_name="sdxl", vram_gb=10.0)
    tracker.allocate(gpu=1, model_name="cosyvoice2", vram_gb=3.0)

    tracker.release_all()
    assert tracker.get_free(0) == 24.0
    assert tracker.get_free(1) == 24.0
