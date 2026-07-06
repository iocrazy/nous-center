"""性能二轮:vllm_metrics 共享 client(E-G)。

注:E-H(execution_task 大 JSON 列 deferred)已撤销 —— deferred 让 async 里 commit 后
访问 webhook_events 触发惰性加载 MissingGreenlet,且 webhook 触发路径会读它,生产有
风险。不值得。本 PR 只保留 status 缓存 + 共享 client。
"""


def test_vllm_metrics_shared_client_singleton():
    import src.services.vllm_metrics as vm
    vm._client = None
    c1 = vm._get_client()
    c2 = vm._get_client()
    assert c1 is c2  # 复用同一 client
