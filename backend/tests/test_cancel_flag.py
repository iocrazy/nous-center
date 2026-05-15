"""Lane G: CancelFlag —— threading.Event 薄包装，cancel 信号穿 to_thread 的载体。"""
import threading

from src.services.inference.cancel_flag import CancelFlag


def test_initial_state_not_set():
    flag = CancelFlag()
    assert flag.is_set() is False
    assert flag.reason is None


def test_set_records_reason():
    flag = CancelFlag()
    flag.set("user requested")
    assert flag.is_set() is True
    assert flag.reason == "user requested"


def test_set_default_reason():
    flag = CancelFlag()
    flag.set()
    assert flag.is_set() is True
    assert flag.reason == "cancelled"


def test_set_is_idempotent_first_reason_wins():
    """多处可能都 set（pipe-reader Abort + wait_for timeout 竞态）；
    第一个 reason 留下，后续 set 不覆盖 —— 便于事后判定真正触发源。"""
    flag = CancelFlag()
    flag.set("node timeout")
    flag.set("user requested")
    assert flag.reason == "node timeout"


def test_clear_resets():
    flag = CancelFlag()
    flag.set("x")
    flag.clear()
    assert flag.is_set() is False
    assert flag.reason is None


def test_visible_across_threads():
    """worker 线程里 set，主线程能看到 —— 这是它存在的全部理由
    （to_thread 工作线程 set / 主线程 poll，或反之）。"""
    flag = CancelFlag()
    seen = []

    def worker():
        flag.set("from worker")
        seen.append(flag.is_set())

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen == [True]
    assert flag.is_set() is True
    assert flag.reason == "from worker"
