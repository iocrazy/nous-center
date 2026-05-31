"""Admin 登录限流 —— 防密码/TOTP 在线暴力(bug hunt round2 #3)。

admin 登录端点(/sys/admin/login、/totp/login、/passkey/login/finish)在 `/api/` gate 之外、
原本零限流。TOTP 6 位码 + valid_window=1 ≈ 3/10⁶/次,无限流下脚本可数小时内命中拿到 admin
session(等价 root)。本模块按客户端 IP 做失败计数 + 指数退避锁(纯内存,单 admin 够用)。

IP 取真实客户端:cloudflared 转发 CF-Connecting-IP / X-Forwarded-For;裸跑回退 request.client。
成功登录清零该 IP。锁定期间所有登录尝试 429。
"""
from __future__ import annotations

import threading
import time

_MAX_FAILS = 5            # 连续失败几次后开始锁
_BASE_LOCK_S = 30.0       # 第 1 次锁 30s,之后指数翻倍
_LOCK_CAP_S = 3600.0      # 锁封顶 1h
_GC_AFTER_S = 7200.0      # 条目空闲 2h 清理,防 map 无限增长

# ip → (consecutive_fails, locked_until_monotonic, last_seen_monotonic)
_STATE: dict[str, tuple[int, float, float]] = {}
_MUTEX = threading.Lock()


def _client_ip(request) -> str:
    h = request.headers
    cf = h.get("cf-connecting-ip")
    if cf:
        return cf.strip()
    xff = h.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _gc(now: float) -> None:
    """惰性清理空闲条目(在持锁内调)。"""
    stale = [ip for ip, (_f, _lu, seen) in _STATE.items() if now - seen > _GC_AFTER_S]
    for ip in stale:
        _STATE.pop(ip, None)


def remaining_lock_seconds(request) -> float:
    """该 IP 当前被锁的剩余秒数;未锁返回 0。"""
    ip = _client_ip(request)
    now = time.monotonic()
    with _MUTEX:
        fails, locked_until, _seen = _STATE.get(ip, (0, 0.0, now))
        _STATE[ip] = (fails, locked_until, now)  # 刷 last_seen
        return max(0.0, locked_until - now)


def record_failure(request) -> None:
    """记一次登录失败;达阈值后设指数退避锁。"""
    ip = _client_ip(request)
    now = time.monotonic()
    with _MUTEX:
        _gc(now)
        fails, _locked_until, _seen = _STATE.get(ip, (0, 0.0, now))
        fails += 1
        locked_until = 0.0
        if fails >= _MAX_FAILS:
            lock_s = min(_BASE_LOCK_S * (2 ** (fails - _MAX_FAILS)), _LOCK_CAP_S)
            locked_until = now + lock_s
        _STATE[ip] = (fails, locked_until, now)


def record_success(request) -> None:
    """登录成功 —— 清该 IP 的失败计数与锁。"""
    ip = _client_ip(request)
    with _MUTEX:
        _STATE.pop(ip, None)
