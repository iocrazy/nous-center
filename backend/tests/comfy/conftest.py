"""comfy 测试的全局状态隔离。

`comfy_bridge` 的渲染信号量 `_SEM` 和 `_running_task_id` 是**模块级**的(生产上靠它
保证 sidecar 一次只服务一个渲染)。测试里这会串味:某个用例的后台渲染任务如果没跑完
就结束(断言失败/超时/gate 没 set),信号量不释放,后面的用例全部拿不到 → 随机超时红。
2026-08-12 实测:单跑绿、全量跑偶发红一条 cancel 用例,就是这个。

每个用例前后重置这两个全局,让顺序无关。
"""
import asyncio

import pytest

import src.services.nodes.comfy_bridge as nb


@pytest.fixture(autouse=True)
def _isolate_bridge_globals():
    nb._SEM = asyncio.Semaphore(1)
    nb._running_task_id = None
    nb._running_since = None
    yield
    nb._SEM = asyncio.Semaphore(1)
    nb._running_task_id = None
    nb._running_since = None
