"""Pinned-memory stash 搬运(RAM stash spec 2026-06-12 PR-4,借鉴 ComfyUI pin_memory)。

restore 慢的两个根因(2026-06-12 真机 spike):
1. ``Module.to()`` 逐 tensor 同步拷贝 —— 每个 tensor 一次隐式 synchronize;
2. pageable 内存 H2D 19.4 GB/s,pinned(cudaHostRegister 原地页锁定)53.3 GB/s(2.8×)。

本模块提供:stash 后对 CPU 权重**原地** ``cudaHostRegister``(零拷贝 pin,带全局预算
``NOUS_STASH_PIN_BUDGET_GB`` 默认 64);restore 用 per-tensor ``non_blocking`` 批量 issue
+ 单次 synchronize(pinned 源真异步)。

安全边界(对齐 ComfyUI PINNING_ALLOWED_TYPES 思路):
- 只 pin ``type(t) is torch.Tensor`` 的连续 CPU tensor —— torchao Float8Tensor 等
  subclass ``data_ptr()=0`` 且不支持 ``empty_like/copy_``(真机验证),走同步 ``.to()`` 兜底;
- restore 必须 **持住 CPU tensor 引用直到 synchronize 完、unregister 完**再放 ——
  否则 GPU 拷贝替换 ``p.data`` 后 CPU tensor 被 gc,而 cuda 注册还在 = 悬挂注册(UB);
- 任何一步异常 → 调用方回退整体 ``.to()``(零回归)。
"""
from __future__ import annotations

import itertools
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# 全局已 pin 字节数(runner 进程内单线程串行队列,无需加锁)。
_total_pinned_bytes = 0

# 外部 pinned 占用账本(spec 2026-06-12 ram-pinned-linkage PR-1):diffusers 流式分块的
# 预 pin 走 `tensor.pin_memory()`(cudaHostAlloc 拷贝),不经本模块 cudaHostRegister,
# 历史上完全绕过账本(~37G pinned RAM 隐身)。挂载方实测/估算字节后 register_external
# 入账,卸载时 release_external 出账 —— total_pinned_bytes() 自此 = 真实 pinned 总量,
# 预算(NOUS_STASH_PIN_BUDGET_GB)对两类占用统一生效,不会叠到远超物理合理值。
_external_pinned: dict[int, int] = {}
_next_external_handle = 1


def _pin_budget_bytes() -> int:
    return int(float(os.getenv("NOUS_STASH_PIN_BUDGET_GB", "64")) * 1e9)


def pin_budget_bytes() -> int:
    """pinned 内存预算上限(字节)。调用方(RAM 门禁)据此判降级,与本模块预算口径一致。"""
    return _pin_budget_bytes()


def register_external(nbytes: int) -> int:
    """外部 pinned 占用入账(流式分块预 pin)。返回 handle,释放时传给 release_external。"""
    global _next_external_handle
    handle = _next_external_handle
    _next_external_handle += 1
    _external_pinned[handle] = max(0, int(nbytes))
    return handle


def release_external(handle: int | None) -> None:
    """外部 pinned 占用出账。handle 为 None / 重复释放均安全(no-op)。"""
    if handle is not None:
        _external_pinned.pop(handle, None)


def _tensors_of(module: Any):
    for t in itertools.chain(module.parameters(), module.buffers()):
        yield t


def pin_module_inplace(module: Any) -> list[tuple[int, int]]:
    """对 module(已在 CPU)的常规权重 tensor 原地 cudaHostRegister。

    返回注册清单 [(ptr, nbytes)],调用方保存、销毁/restore 后用 `unpin` 注销。
    预算外/subclass/非连续/注册失败 → 跳过该 tensor(降级 pageable,无害)。
    无 GPU(CI mock torch)→ 空清单。
    """
    global _total_pinned_bytes
    regs: list[tuple[int, int]] = []
    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            return regs
        cudart = torch.cuda.cudart()
        for p in _tensors_of(module):
            t = p.data
            if type(t) is not torch.Tensor:  # subclass(Float8Tensor 等)不可 pin
                continue
            if (not t.device.type == "cpu") or (not t.is_contiguous()) or t.is_pinned():
                continue
            size = t.nbytes
            # 预算按真实总量(stash 原地 pin + 外部流式 pin)统一卡 —— 旧口径只看 stash
            # 自己,流式占 37G 后 stash 还能再 pin 满 64G,叠到远超物理合理值。
            if size <= 0 or total_pinned_bytes() + size > _pin_budget_bytes():
                continue
            ptr = t.data_ptr()
            if ptr == 0:
                continue
            # 1 = cudaHostRegisterPortable
            if int(cudart.cudaHostRegister(ptr, size, 1)) == 0:
                regs.append((ptr, size))
                _total_pinned_bytes += size
    except Exception as e:  # noqa: BLE001 — pin 是纯优化,失败不挡 stash
        logger.warning("pin_module_inplace 部分失败(降级 pageable):%s", e)
    return regs


def unpin(regs: list[tuple[int, int]]) -> None:
    """注销 cudaHostRegister。调用方必须保证对应内存还活着(持引用)。"""
    global _total_pinned_bytes
    if not regs:
        return
    try:
        import torch  # noqa: PLC0415
        cudart = torch.cuda.cudart()
        for ptr, size in regs:
            try:
                cudart.cudaHostUnregister(ptr)
            finally:
                _total_pinned_bytes -= size
    except Exception as e:  # noqa: BLE001
        logger.warning("unpin 失败(泄漏 %d 条注册):%s", len(regs), e)


def restore_module_fast(module: Any, device: str, regs: list[tuple[int, int]]) -> None:
    """stashed module 搬回 device:常规 tensor non_blocking 批量 issue + 单次 sync
    (pinned 源走 DMA 真异步);subclass 同步 ``.to()``。完成后注销 pin。

    异常时也先 unpin(CPU 引用仍由 module/holds 持着)再 raise —— 调用方回退重建。
    """
    import torch  # noqa: PLC0415
    holds: list[Any] = []  # CPU tensor 引用,撑到 unregister 完成(防悬挂注册)
    try:
        for p in _tensors_of(module):
            src = p.data
            if type(src) is torch.Tensor and src.device.type == "cpu":
                dst = torch.empty_like(src, device=device)
                dst.copy_(src, non_blocking=True)
                holds.append(src)
                p.data = dst
            else:
                p.data = src.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
    finally:
        unpin(regs)
        holds.clear()


def total_pinned_bytes() -> int:
    """真实 pinned 总量 = stash 原地注册 + 外部入账(流式预 pin)。"""
    return _total_pinned_bytes + sum(_external_pinned.values())
