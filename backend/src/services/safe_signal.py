"""Broadcast-safe process signalling.

线上事故根因(2026-07-05):测试/清理代码里的 `os.killpg(os.getpgid(pid), sig)`
在 `pgid <= 1` 时等价于 `kill(-1, sig)` / `kill(0, sig)` —— 向调用用户能触及的
**所有进程**广播信号。bpftrace 抓到 pytest 把 SIGTERM 打给了宿主的 mihomo /
sshd-session / systemd user@1000。触发链:被清理的目标是孤儿(被 init 收养)或
PID 已死被回收 → `getpgid` 返回 1 / 0 / 指向无关进程 → `killpg` 一锅端。

统一经本模块发信号:
- `safe_killpg`:解析 pgid,**pgid <= 1 一律拒绝**;可选 `verify` 回调在发信号前
  复核 PID 身份(防回收误杀)。
- `safe_kill`:**pid <= 1 或 pid == 0 一律拒绝**(0 → 调用者进程组;1 → init;负数
  → 进程组广播)。

两者都返回 bool(True=已发信号),吞掉 ProcessLookupError(目标已消失)。
"""
from __future__ import annotations

import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)


def _proc_cmdline_contains(pid: int, needle: str) -> bool:
    """True iff /proc/<pid>/cmdline 存在且包含 needle(小写匹配)。用于 verify 回调。"""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return needle.lower().encode() in fh.read().lower()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def safe_killpg(pid: int, sig: int, *, verify: Callable[[int], bool] | None = None) -> bool:
    """向 pid 所在进程组发 sig,但拒绝广播。

    - `verify`:发信号前调用 `verify(pid)`,返回 False → 拒绝(PID 回收防护)。
    - 解析出的 `pgid <= 1` → 拒绝并记 error(广播防护,本函数存在的核心理由)。
    返回 True 当且仅当信号真正发出。ProcessLookupError(进程已没)→ False。
    """
    if pid <= 1:
        logger.error("safe_killpg refused: pid=%d <= 1 (broadcast guard)", pid)
        return False
    if verify is not None and not verify(pid):
        logger.warning("safe_killpg refused pid=%d: verify() failed (recycled PID?)", pid)
        return False
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return False  # 已消失
    except PermissionError:
        logger.error("safe_killpg: getpgid(%d) permission denied", pid)
        return False
    if pgid <= 1:
        logger.error(
            "safe_killpg REFUSED broadcast: pid=%d resolved pgid=%d (<=1). "
            "killpg would signal every process the user can reach. Skipping.",
            pid, pgid,
        )
        return False
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False


def safe_kill(pid: int, sig: int) -> bool:
    """向单个 pid 发 sig,拒绝 pid<=1 / pid==0(0→调用者进程组,1→init,负→广播)。

    返回 True 当且仅当信号发出。ProcessLookupError → False。
    """
    if pid <= 1:
        logger.error("safe_kill refused: pid=%d (<=1 / broadcast guard)", pid)
        return False
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return False
