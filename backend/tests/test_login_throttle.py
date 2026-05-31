"""admin 登录限流 —— bug hunt round2 #3。"""
from src.api import login_throttle as lt


class _Req:
    def __init__(self, ip="1.2.3.4", headers=None):
        self.headers = headers or {}
        self.client = type("C", (), {"host": ip})()


def _fresh():
    lt._STATE.clear()


def test_locks_after_max_consecutive_fails():
    _fresh()
    r = _Req("9.9.9.9")
    assert lt.remaining_lock_seconds(r) == 0
    for _ in range(lt._MAX_FAILS):
        lt.record_failure(r)
    assert lt.remaining_lock_seconds(r) > 0  # 达阈值 → 锁


def test_success_clears_lock():
    _fresh()
    r = _Req("8.8.8.8")
    for _ in range(lt._MAX_FAILS):
        lt.record_failure(r)
    assert lt.remaining_lock_seconds(r) > 0
    lt.record_success(r)
    assert lt.remaining_lock_seconds(r) == 0  # 成功清零


def test_per_ip_independent():
    _fresh()
    a, b = _Req("1.1.1.1"), _Req("2.2.2.2")
    for _ in range(lt._MAX_FAILS):
        lt.record_failure(a)
    assert lt.remaining_lock_seconds(a) > 0
    assert lt.remaining_lock_seconds(b) == 0  # 另一 IP 不受影响


def test_keys_on_real_client_ip_behind_tunnel():
    """cloudflared 后:按 CF-Connecting-IP(真实攻击者)计数,不是隧道 IP。"""
    _fresh()
    attacker = _Req("tunnel-ip", headers={"cf-connecting-ip": "6.6.6.6"})
    for _ in range(lt._MAX_FAILS):
        lt.record_failure(attacker)
    # 同隧道 IP、不同真实 IP 的合法用户不被连带锁
    legit = _Req("tunnel-ip", headers={"cf-connecting-ip": "7.7.7.7"})
    assert lt.remaining_lock_seconds(attacker) > 0
    assert lt.remaining_lock_seconds(legit) == 0
