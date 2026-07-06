"""Broadcast-kill guard tests (线上事故: pytest killpg(pgid<=1) 广播 SIGTERM 到
宿主 mihomo/sshd/systemd)。safe_killpg/safe_kill 必须在 pgid<=1 / pid<=1 时拒绝发信号。"""
import os
import signal
import pytest

from src.services.safe_signal import safe_killpg, safe_kill


class TestSafeKillpg:
    def test_refuses_pgid_1(self, monkeypatch):
        """getpgid 返回 1 → killpg(1) == kill(-1) 广播,必须拒绝且不调用 killpg。"""
        monkeypatch.setattr(os, "getpgid", lambda pid: 1)
        called = []
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: called.append((pgid, sig)))
        assert safe_killpg(3041793, signal.SIGTERM) is False
        assert called == []

    def test_refuses_pgid_0(self, monkeypatch):
        """getpgid 返回 0 → killpg(0) 杀调用者自己整个进程组,拒绝。"""
        monkeypatch.setattr(os, "getpgid", lambda pid: 0)
        called = []
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: called.append((pgid, sig)))
        assert safe_killpg(999, signal.SIGKILL) is False
        assert called == []

    def test_sends_when_pgid_valid(self, monkeypatch):
        monkeypatch.setattr(os, "getpgid", lambda pid: 4242)
        called = []
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: called.append((pgid, sig)))
        assert safe_killpg(4242, signal.SIGTERM) is True
        assert called == [(4242, signal.SIGTERM)]

    def test_verify_callback_false_refuses(self, monkeypatch):
        """PID 回收防护:verify 返回 False(cmdline 不再是目标进程)→ 不发信号。"""
        monkeypatch.setattr(os, "getpgid", lambda pid: 4242)
        called = []
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: called.append((pgid, sig)))
        assert safe_killpg(4242, signal.SIGTERM, verify=lambda pid: False) is False
        assert called == []

    def test_dead_pid_returns_false(self, monkeypatch):
        def _boom(pid):
            raise ProcessLookupError()
        monkeypatch.setattr(os, "getpgid", _boom)
        assert safe_killpg(4242, signal.SIGTERM) is False


class TestSafeKill:
    @pytest.mark.parametrize("bad_pid", [0, 1, -1, -5])
    def test_refuses_nonpositive_and_one(self, monkeypatch, bad_pid):
        called = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: called.append((pid, sig)))
        assert safe_kill(bad_pid, signal.SIGTERM) is False
        assert called == []

    def test_sends_when_pid_valid(self, monkeypatch):
        called = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: called.append((pid, sig)))
        assert safe_kill(4242, signal.SIGKILL) is True
        assert called == [(4242, signal.SIGKILL)]

    def test_dead_pid_returns_false(self, monkeypatch):
        def _boom(pid, sig):
            raise ProcessLookupError()
        monkeypatch.setattr(os, "kill", _boom)
        assert safe_kill(4242, signal.SIGTERM) is False


class TestHarnessBroadcastGuard:
    """证明 conftest 模块级护栏:pytest 进程永远无法对 pgid<=1 / pid<=1 投递真实信号
    (2026-07-05 事故 —— pytest 广播 SIGTERM 打死宿主 mihomo/sshd/systemd)。"""

    def test_killpg_broadcast_pgid_1_raises(self):
        with pytest.raises(AssertionError, match="broadcast"):
            os.killpg(1, signal.SIGTERM)

    def test_killpg_broadcast_pgid_0_raises(self):
        with pytest.raises(AssertionError):
            os.killpg(0, signal.SIGKILL)

    def test_kill_pid_1_raises(self):
        with pytest.raises(AssertionError):
            os.kill(1, signal.SIGTERM)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_kill_nonpositive_raises(self, bad):
        with pytest.raises(AssertionError):
            os.kill(bad, signal.SIGTERM)


class TestOrphanScanKillRefusesBroadcast:
    """主进程启动清理孤儿 vLLM 的路径(main.py):目标解析出 pgid<=1 时必须拒绝,
    不得广播。用 safe_killpg 组合验证(getpgid→1 → 不投递)。"""

    def test_orphan_kill_pgid_1_no_delivery(self, monkeypatch):
        import src.services.safe_signal as ss
        monkeypatch.setattr(os, "getpgid", lambda pid: 1)
        delivered = []
        # 直接盯真实 killpg 是否被触达(safe_killpg 内部用 os.killpg)
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: delivered.append((pgid, sig)))
        # verify 通过(装成还是 vllm),仅靠 pgid<=1 守卫拦截
        ok = ss.safe_killpg(3041793, signal.SIGKILL, verify=lambda p: True)
        assert ok is False
        assert delivered == []
