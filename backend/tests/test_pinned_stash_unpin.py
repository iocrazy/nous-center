"""回归:unpin 某 ptr 注销失败时,剩余 ptr 仍要尝试(不跳出),账目只减成功项。

审查发现:早先 per-ptr try/finally(非 except)——中间 ptr 抛异常跳出循环,后面 ptr
泄漏;finally 还对失败项减字节。
"""

import src.services.inference.pinned_stash as ps


class _FakeCudart:
    def __init__(self, fail_ptr):
        self.calls = []
        self._fail = fail_ptr

    def cudaHostUnregister(self, ptr):
        self.calls.append(ptr)
        if ptr == self._fail:
            raise RuntimeError("cudaHostUnregister failed")
        return 0


def test_unpin_attempts_all_ptrs_despite_middle_failure(monkeypatch):
    import torch  # noqa: PLC0415
    fake = _FakeCudart(fail_ptr=20)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: fake, raising=False)
    ps._total_pinned_bytes = 300

    ps.unpin([(10, 100), (20, 100), (30, 100)])

    # 关键:中间 ptr(20)失败不该跳出 → 30 仍被尝试注销
    assert fake.calls == [10, 20, 30], "中间 ptr 失败后剩余 ptr 未尝试(泄漏)"
    # 仅成功的 10、30 减账;失败的 20 不减 → 300-100-100 = 100
    assert ps._total_pinned_bytes == 100


def test_unpin_all_success_decrements_all(monkeypatch):
    import torch  # noqa: PLC0415
    fake = _FakeCudart(fail_ptr=None)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: fake, raising=False)
    ps._total_pinned_bytes = 200

    ps.unpin([(10, 100), (20, 100)])
    assert fake.calls == [10, 20]
    assert ps._total_pinned_bytes == 0
