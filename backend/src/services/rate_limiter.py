"""In-process sliding-window rate limiter for service instances.

Single-worker scope only. Multi-worker / multi-host needs Redis (v2).

## 准入即占坑(reserve-at-admission),不是"先查后记"

旧实现把限流拆成 check(准入时只读)+ record(推理后才追加)。两步之间有 TOCTOU
窗口:N 个并发请求在 record 发生前都看到 `len(window) < rpm` → 全部放行,RPM 被绕过
(rpm=10 时并发 1000 个全过)。而且 record 挂在记账路径上,客户端断连/非 LLM(TTS、
workflow)/出错的请求根本不 record → 这些请求完全不计入 RPM。

现在 RPM 与 TPM 用两个独立窗口:
- **RPM(请求数)**:在 `reserve()` 的同一把锁、同一临界区里**检查 + 追加时间戳**。
  并发请求彼此可见,准入瞬间就占住名额 → 关掉 TOCTOU。请求一旦准入就计数,
  无论后续成功/失败/断连/是否产 token,RPM 都算 —— 这才是"每分钟请求数"的正确语义,
  也顺带堵了"拿失败请求刷量绕过"。
- **TPM(token 数)**:token 数只有推理完才知道,天然无法在准入前精确预占。沿用
  `record(tokens)` 在推理后追加,准入时 TPM 检查看的是**已结算的尾随窗口**(当前请求
  贡献 0 直到结算)。这是滑动窗口限流的固有近似:超额最多溢出"正在并发的这一批",
  下一批被尾随窗口挡住。
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque

from src.errors import RateLimitError

WINDOW_SECONDS = 60


class InstanceRateLimiter:
    def __init__(self):
        # instance_id -> deque[ts] —— RPM 请求时间戳,reserve() 准入时追加
        self._req_events: dict[int, deque] = defaultdict(deque)
        # instance_id -> deque[(ts, tokens)] —— TPM token 数,record() 推理后追加
        self._tok_events: dict[int, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()

    @staticmethod
    def _prune(q: deque, now: float) -> None:
        cutoff = now - WINDOW_SECONDS
        while q and q[0][0] < cutoff:
            q.popleft()

    @staticmethod
    def _prune_ts(q: deque, now: float) -> None:
        """RPM 窗口元素是裸时间戳(非元组),单独一个 prune。"""
        cutoff = now - WINDOW_SECONDS
        while q and q[0] < cutoff:
            q.popleft()

    async def reserve(
        self,
        instance_id: int,
        rpm_limit: int | None,
        tpm_limit: int | None,
    ) -> None:
        """准入检查 + RPM 占坑(原子)。超限抛 RateLimitError,否则占住一个 RPM 名额。

        必须在请求**被准入的那一刻**调用(鉴权通过后、真正干活前),每个准入路径都要调,
        包括 M:N key 解析出目标 instance 之后 —— 否则那条路径就是限流漏洞。
        """
        if not rpm_limit and not tpm_limit:
            return
        now = time.monotonic()
        async with self._lock:
            rq = self._req_events[instance_id]
            self._prune_ts(rq, now)
            if rpm_limit and len(rq) >= rpm_limit:
                raise RateLimitError(
                    f"instance RPM limit ({rpm_limit}/min) exceeded",
                    code="rate_limit_rpm",
                )
            if tpm_limit:
                tq = self._tok_events[instance_id]
                self._prune(tq, now)
                used = sum(tok for _, tok in tq)
                if used >= tpm_limit:
                    raise RateLimitError(
                        f"instance TPM limit ({tpm_limit}/min) exceeded; "
                        f"current window: {used}",
                        code="rate_limit_tpm",
                    )
            # 检查通过 —— 在同一临界区里立刻占坑,关掉 check-then-act 的并发窗口。
            rq.append(now)

    async def record(self, instance_id: int, tokens: int) -> None:
        """推理完成后回填本次 token 数(供 TPM 尾随窗口计量)。RPM 已在 reserve 占过坑。"""
        now = time.monotonic()
        async with self._lock:
            tq = self._tok_events[instance_id]
            self._prune(tq, now)
            tq.append((now, max(0, int(tokens or 0))))


_limiter: InstanceRateLimiter | None = None


def get_rate_limiter() -> InstanceRateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = InstanceRateLimiter()
    return _limiter
