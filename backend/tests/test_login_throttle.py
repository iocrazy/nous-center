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
    """cloudflared(本机回环 peer)后:按 CF-Connecting-IP(真实攻击者)计数,不是隧道 IP。"""
    _fresh()
    attacker = _Req("127.0.0.1", headers={"cf-connecting-ip": "6.6.6.6"})
    for _ in range(lt._MAX_FAILS):
        lt.record_failure(attacker)
    # 同隧道(loopback peer)、不同真实 IP 的合法用户不被连带锁
    legit = _Req("127.0.0.1", headers={"cf-connecting-ip": "7.7.7.7"})
    assert lt.remaining_lock_seconds(attacker) > 0
    assert lt.remaining_lock_seconds(legit) == 0


def test_direct_connect_forged_cf_header_ignored():
    """round6 安全:直连(peer 非 loopback)时无视伪造的 CF-Connecting-IP,按真实 peer 计数 ——
    否则每请求换一个伪造头就永远锁不上、暴破旁路。"""
    _fresh()
    # 攻击者直连 :8000(peer=5.5.5.5),每次带不同伪造 cf-connecting-ip
    for i in range(lt._MAX_FAILS):
        lt.record_failure(_Req("5.5.5.5", headers={"cf-connecting-ip": f"9.9.9.{i}"}))
    # 伪造头被无视 → 全记在真实 peer 5.5.5.5 上 → 已锁
    assert lt.remaining_lock_seconds(_Req("5.5.5.5", headers={"cf-connecting-ip": "1.1.1.1"})) > 0
