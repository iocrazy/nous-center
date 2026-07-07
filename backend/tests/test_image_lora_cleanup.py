"""回归:_apply_loras 空列表分支必须删掉旧 LoRA,否则 [loraA]→[] 转换下权重常驻 → VRAM 累积。

审查发现:空列表分支 set_adapters([]) 只停用不释放,且提前 return 跳过下面的
delete_adapters 清理。
"""
from src.services.inference import image_modular


class _FakeLoraPipe:
    def __init__(self, active=None):
        self._active = list(active or [])
        self.deleted = []

    def get_active_adapters(self):
        return list(self._active)

    def set_adapters(self, names, adapter_weights=None):
        self._active = list(names)

    def delete_adapters(self, names):
        self.deleted.extend(names)


def _backend():
    return image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")


def test_apply_loras_empty_deletes_stale_and_clears():
    be = _backend()
    be._pipe = _FakeLoraPipe(active=["loraA"])
    be._loaded_loras = {"loraA"}

    be._apply_loras([])  # 本次请求不带 LoRA

    assert "loraA" in be._pipe.deleted, "空 LoRA 请求没删掉旧 LoRA → 权重常驻 pipe(VRAM 累积)"
    assert be._loaded_loras == set(), "_loaded_loras 没收敛到空"


def test_apply_loras_empty_noop_when_none_loaded():
    """没装过 LoRA 时空请求不该炸、不该乱删。"""
    be = _backend()
    be._pipe = _FakeLoraPipe(active=[])
    be._loaded_loras = set()
    be._apply_loras([])
    assert be._pipe.deleted == []
