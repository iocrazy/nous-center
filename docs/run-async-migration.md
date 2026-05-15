# /run 异步契约迁移指引（V1.5 D17）

## 变更摘要

V1.5 起，workflow 执行端点从「同步阻塞到完成」改为「纯异步入队」：

| 端点 | V1（旧） | V1.5（新） |
|---|---|---|
| `POST /v1/instances/{id}/run` | 同步执行，阻塞到 workflow 跑完，body 直接是 result | 入队 → 立即返回 `202 {"task_id": "..."}` |
| `POST /api/v1/workflows/execute` | 同上 | 同上 |

**单次 LLM 调用不受影响** —— OpenAI/Anthropic/Ollama/Responses compat 路由本来就直连
vLLM HTTP、本来就同步，行为不变。D17 只改多节点 workflow 端点。

## 上游（mediahub 等）迁移步骤

旧代码（同步等结果）：

```python
resp = httpx.post(f"{base}/v1/instances/{iid}/run",
                  json={"inputs": ...}, headers=auth)
result = resp.json()          # V1：body 直接是 result
```

新代码（拿 task_id → 轮询）：

```python
resp = httpx.post(f"{base}/v1/instances/{iid}/run",
                  json={"inputs": ...}, headers=auth)
assert resp.status_code == 202
task_id = resp.json()["task_id"]

# 方式 A：轮询
while True:
    t = httpx.get(f"{base}/api/v1/tasks/{task_id}").json()
    if t["status"] in ("completed", "failed", "cancelled"):
        break
    time.sleep(0.5)
result = t["result"]          # status=completed 时

# 方式 B：订阅 WS（实时进度）
#   ws connect {base}/ws/workflow/{instance_id}
#   收 node_start / node_complete / complete 事件
```

## 迁移期兼容（可选）

如上游一时改不动，可在服务端加一个 `?wait=true` 兼容 flag：服务端代为轮询
task 直到终态再返回 result。该 flag 标记 **deprecated**，仅为过渡——长期
所有上游都应走 task_id + 轮询/WS。

（注：`?wait=true` 兼容 flag 本 Lane 不实现；若 mediahub 迁移确实需要缓冲期，
另开 PR 加，并从落地起就标 deprecated。）
