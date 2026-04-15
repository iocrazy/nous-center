# Step 6 · 接入点限流 + 临时 API Key

## Context

接入点当前只有 active/inactive 状态，无限流、无临时凭据。Ark 的接入点有 RPM/TPM，且可发临时 Key 给第三方 SDK。这步补齐。

## 决策

1. **限流存储**：进程内 `dict[instance_id, deque[timestamps]]` + `asyncio.Lock`。单 worker 够；多 worker 上 Redis 再说（v2）。
2. **双维度**：`rate_limit_rpm`（每分钟请求数）、`rate_limit_tpm`（每分钟 token 数，用上一次请求的 usage 作计数口径，不做预估）。Null 表示不限。
3. **触发位置**：`verify_bearer_token` 之后立即查；RPM 超限抛 `RateLimitError`（HTTP 429, code=`rate_limit_rpm`）；TPM 同理（code=`rate_limit_tpm`）。
4. **临时 Key**：`InstanceApiKey` 增 `expires_at` nullable。`verify_bearer_token` 里命中 key 后判过期 → 401 `code=api_key_expired`。
5. **新端点**：`POST /api/v1/instances/{id}/keys/temporary { label, duration_seconds }` → 返回明文 key（只此一次）+ 元数据。`duration_seconds` 范围 `[60, 30*86400]`。
6. **管控面 PATCH**：已有 `PATCH /api/v1/instances/{id}` 扩展 body 接受 `rate_limit_rpm`/`rate_limit_tpm`。
7. **清理**：临时 key 过期后不自动删，鉴权层拒绝即可；后台 loop 每 6h 硬删 `expires_at < now - 7d` 的残骸（可选，先不做）。

## Schema 改动

```python
# service_instance.py
rate_limit_rpm = Column(Integer, nullable=True)
rate_limit_tpm = Column(Integer, nullable=True)

# instance_api_key.py
expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
```

## 限流器接口

```python
class InstanceRateLimiter:
    async def check(self, instance_id, rpm_limit, tpm_limit) -> None
    async def record(self, instance_id, tokens) -> None
```
- `check`：滑动 60s 窗口内统计请求数/总 token；超限抛 RateLimitError。
- `record`：请求完成后调用，写入当前时间戳 + 本次 token 数。

## 验证

```bash
# 1. 临时 key
curl -X POST /api/v1/instances/$IID/keys/temporary -d '{"label":"test","duration_seconds":60}'
# → {"key":"sk-tmp-xxx", "expires_at": ...}
# 等 61s 后用 → 401 code=api_key_expired

# 2. 限流
PATCH /api/v1/instances/$IID {"rate_limit_rpm": 5}
# 连发 6 个 → 第 6 个 429 code=rate_limit_rpm

# 3. TPM
PATCH /api/v1/instances/$IID {"rate_limit_tpm": 100}
# 一次大 prompt 产生 >100 tokens → 下一次请求 429 code=rate_limit_tpm
```

## 不做的事

- 不做 UI（下一轮）
- 不做 Redis（v2 再换）
- 不做 per-key 限流（接入点粒度足够）
