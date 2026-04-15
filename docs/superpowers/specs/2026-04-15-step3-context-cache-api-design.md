# Step 3 · Context Cache API（common_prefix 模式）

## Context

vLLM 引擎本身有 prefix caching：相同前缀的 prompt 第二次进入时不重新计算 KV cache，命中部分以接近零延迟服务。但这个能力对 nous-center 的外部用户**不可见、不可控**：

- 用户不知道哪些请求命中了 cache
- vLLM 的 prefix cache 是 LRU，长 system prompt 在并发流量下可能被挤掉
- `usage.prompt_tokens_details.cached_tokens` 字段虽然 vLLM 返回，但客户端没法主动管理 cache 生命周期

火山方舟的 Context Cache API 暴露了这层能力：用户先创建 cache（`ctx-xxx`），后续请求带上 `context_id`，服务端前置 cache 内容、记录命中、按需续期。这是 nous-center 升级到"准平台级"体验的一块关键能力。

**典型场景：** 一个 Agent 应用有 3KB 的 system prompt + 工具定义。一天调用 10000 次：
- 不用 Cache：每次重新预填充 3KB，~750 input tokens × 10000 = 7.5M tokens 的算力浪费
- 用 Cache：3KB 预热一次，后续 9999 次几乎零成本

我们这一层做的不是新算法，而是把 vLLM 已有的 prefix caching 显式管理化（生命周期 + 计数 + 跨进程持久），并通过 OpenAI SDK 兼容的 `extra_body.context_id` 暴露。

## 决策记录

| 决策 | 选择 | 理由 |
|---|---|---|
| 元数据存储 | PostgreSQL 持久化（`context_caches` 表） | 重启不丢；可查询统计；元数据访问不是热路径（vLLM GPU cache 才是） |
| 端点形态 | 复用 `/v1/chat/completions` + 独立管理端点 | OpenAI SDK 客户端无需切 endpoint；管理操作分离更清晰 |
| 触发字段 | `extra_body.context_id` | 非 OpenAI 标准字段，按 Ark 惯例放 extra_body |
| 缓存模式 | 仅 `common_prefix`（不做 `session`） | session 与 Step 4 Responses API 重复；prefix 是不可替代的核心 |
| TTL 行为 | 每次 use 重置 expires_at | 热 cache 永不过期，冷 cache 自动淘汰，符合直觉 |
| 清理策略 | lazy delete（access 时）+ 后台 task（1h 扫一次） | 双保险；lazy 兜底，后台清僵尸 |
| 权限范围 | 绑定 `instance_id`（同接入点共享） | dev/staging/prod 多个 key 可复用同一前缀 |
| 创建预热 | 创建时调一次 vLLM 实际预热 prefix cache | 防止首次 chat 之前被 LRU 挤掉；返回真实 prompt_tokens |
| `hit_count` 字段 | 加，每次 use +1 | 几乎免费；将来可视化热点、可扩展 LFU 淘汰 |
| Cache 失败时 | hard 404（不静默 fallback） | 用户显式指定 context_id，找不到必须报错，否则用户以为命中实则在重算 |

## 架构

```
┌──────────────────── POST /v1/context/create ─────────────────────┐
│  Bearer sk-xxx                                                   │
│  body: {model, messages: [...], ttl?: 86400}                     │
│         ▼                                                        │
│  1. verify_bearer_token → instance, api_key                      │
│  2. resolve engine adapter, get base_url                         │
│  3. POST {base_url}/v1/chat/completions with max_tokens=1        │
│     → vLLM预热 prefix cache, 返回 usage.prompt_tokens             │
│  4. INSERT context_caches (id="ctx-...", messages_json, ...)     │
│         ▼                                                        │
│  ◀── { id: "ctx-...", model, ttl, expires_at, prompt_tokens,    │
│        usage: {prompt_tokens, prompt_tokens_details: {           │
│              cached_tokens: 0 }} }                               │
└──────────────────────────────────────────────────────────────────┘

┌──────────────── POST /v1/chat/completions (with cache) ──────────┐
│  body: { messages: [{user}], extra_body: {context_id: "ctx-..."} │
│         ▼                                                        │
│  1. extract context_id from body (top-level OR extra_body)       │
│  2. SELECT context_caches WHERE id=ctx-... AND expires_at>now    │
│     - not found / expired / wrong instance → NotFoundError       │
│  3. body["messages"] = cache.messages + body["messages"]         │
│  4. UPDATE context_caches SET hit_count+=1,                       │
│     expires_at=now+ttl, last_used_at=now (background task)       │
│  5. proxy to vLLM as usual                                       │
│         ▼                                                        │
│  ◀── normal chat response with usage.prompt_tokens_details       │
│      .cached_tokens reflecting vLLM's actual hit                 │
└──────────────────────────────────────────────────────────────────┘

┌─── Background cleanup task (asyncio task launched in lifespan) ──┐
│  every 3600s:                                                    │
│  DELETE FROM context_caches WHERE expires_at < now               │
│  log count                                                       │
└──────────────────────────────────────────────────────────────────┘
```

## Schema

新增表 `context_caches`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | `varchar(64)` PK | `ctx-{base58 of snowflake bigint}` |
| `instance_id` | `bigint` FK→`service_instances.id`, indexed | 权限边界，cascade delete |
| `api_key_id` | `bigint` FK→`instance_api_keys.id`, nullable | 创建者，仅审计 |
| `model` | `varchar(128)` | 创建时的 engine_name 快照（防止 instance source 改名） |
| `mode` | `varchar(32)` default `'common_prefix'` | 预留扩展（暂只用一种） |
| `messages_json` | `jsonb` | 前缀 messages 数组 |
| `prompt_tokens` | `int` | vLLM 在创建时返回的实际 token 数 |
| `ttl_seconds` | `int` default `86400` | 创建时设定，影响每次 use 重置后的 expires_at |
| `expires_at` | `timestamptz`, indexed | use 时 = `now() + ttl_seconds` |
| `hit_count` | `int` default `0` | 每次 use 在背景 +1 |
| `created_at` | `timestamptz` default `now()` | |
| `last_used_at` | `timestamptz` nullable | use 时更新 |

约束：
- `ttl_seconds`: 检查 `>= 60 AND <= 604800`（1 分钟到 7 天，对齐 Ark）
- `messages_json`: 必须是数组，且元素至少有 `role` + `content`
- 索引：`(expires_at)` 用于清理；`(instance_id, expires_at)` 用于权限+查询

## API 契约

### 1. `POST /v1/context/create`

**Request:**
```json
{
  "model": "qwen3.5-35b-a3b-gptq-int4",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant. <long preamble>"}
  ],
  "ttl": 86400
}
```

**Response 200:**
```json
{
  "id": "ctx-9P3kVxAm7tZ",
  "model": "qwen3.5-35b-a3b-gptq-int4",
  "mode": "common_prefix",
  "ttl": 86400,
  "expires_at": "2026-04-16T15:00:00Z",
  "usage": {
    "prompt_tokens": 754,
    "completion_tokens": 0,
    "total_tokens": 754,
    "prompt_tokens_details": {"cached_tokens": 0}
  }
}
```

**Errors:**
- 400 `invalid_request_error` — `messages` 非数组、`ttl` 越界、`model` 缺失
- 401 / 403 — 鉴权失败
- 503 `api_error` — model 未加载
- 502 `api_error` — vLLM 预热调用失败

### 2. `POST /v1/chat/completions` 增加 `context_id` 支持

**新增 body 字段（位置二选一）：**
```json
{
  "model": "qwen3.5-35b",
  "context_id": "ctx-9P3kVxAm7tZ",   // 顶层
  "messages": [{"role": "user", "content": "hello"}]
}
```
或：
```json
{
  "model": "qwen3.5-35b",
  "messages": [{"role": "user", "content": "hello"}],
  "extra_body": {"context_id": "ctx-9P3kVxAm7tZ"}
}
```

后端读取顺序：top-level `context_id` → `extra_body.context_id`。

**行为：**
1. 查 cache，校验 `instance_id` 一致 + `expires_at > now`
2. 在内存里 `body.messages = cache.messages + body.messages`
3. **`model` 必须匹配** cache 的 model（否则 prefix cache 在 vLLM 端不会命中，违反语义）—— 不匹配抛 400
4. 后台任务（fire-and-forget）：`UPDATE hit_count+=1, expires_at=now+ttl, last_used_at=now`
5. 后续流程不变（vLLM 调用、usage 记录）

**Errors（额外）：**
- 404 `not_found_error` `code=context_not_found` — id 不存在或已过期
- 400 `invalid_request_error` `code=context_model_mismatch` — 请求 model ≠ cache.model
- 403 `permission_error` `code=context_wrong_instance` — cache 属于另一个 instance

### 3. `GET /v1/context/{id}`

**Response 200:**
```json
{
  "id": "ctx-9P3kVxAm7tZ",
  "model": "qwen3.5-35b-a3b-gptq-int4",
  "mode": "common_prefix",
  "ttl": 86400,
  "expires_at": "2026-04-16T15:00:00Z",
  "created_at": "2026-04-15T15:00:00Z",
  "last_used_at": "2026-04-15T18:23:14Z",
  "hit_count": 47,
  "prompt_tokens": 754,
  "messages_preview": [
    {"role": "system", "content": "You are a helpful assistant. <long..."}
  ]
}
```

`messages_preview`: 每条 content 截断到前 200 字符，避免响应过大。完整 messages 故意不返回（拿来就能复刻 prefix 没意义；要复刻请 create 新的）。

**Errors:**
- 404 — 不存在/过期
- 403 — 不是同 instance

### 4. `DELETE /v1/context/{id}`

**Response 204** — 删除成功（包括"已经不存在"也返回 204，幂等）。

**Errors:**
- 403 — 不是同 instance（不允许删别人的）

## 实现细节

### Cache ID 生成

复用现有 snowflake：`f"ctx-{base58_encode(snowflake_id())}"`。base58 选择是为了无 `0/O/I/l` 易混字符。如果项目暂无 base58 工具，用 `secrets.token_urlsafe(12)` 兜底（`ctx-` 前缀 + 12 字符）。

### 预热调用细节

```python
# pseudo
warm_body = {
    "model": "",  # vLLM uses internal path
    "messages": cache_messages,
    "max_tokens": 1,
    "temperature": 0,
    "stream": False,
}
async with httpx.AsyncClient(timeout=60) as client:
    resp = await client.post(f"{base_url}/v1/chat/completions", json=warm_body)
    resp.raise_for_status()
    data = resp.json()
    prompt_tokens = data["usage"]["prompt_tokens"]
```

**重要：** 预热不计入 `record_llm_usage`（不是真实业务请求，只是 cache 预热成本）。但 `prompt_tokens` 要存进 cache 行用于展示。

### 后台 hit_count 更新

不在请求关键路径里做 `UPDATE`，避免增加 chat 延迟。用 fire-and-forget：
```python
asyncio.create_task(_increment_cache_hit(cache.id, cache.ttl_seconds))
```
失败也不影响请求（只是统计偶尔丢一次）。

### 清理任务

`backend/src/services/context_cache_cleaner.py`：
```python
async def cleanup_loop(interval_seconds: int = 3600):
    while True:
        try:
            async with sessionmaker() as s:
                stmt = delete(ContextCache).where(
                    ContextCache.expires_at < datetime.now(timezone.utc)
                )
                result = await s.execute(stmt)
                await s.commit()
                if result.rowcount:
                    logger.info("cleaned %d expired context caches", result.rowcount)
        except Exception:
            logger.exception("context cache cleanup error")
        await asyncio.sleep(interval_seconds)
```

在 `main.py` 的 `lifespan` 启动时 `asyncio.create_task(cleanup_loop())`。

### 并发与一致性

- DELETE + concurrent chat 使用同一 cache：chat 读到 cache → DELETE 删除行 → chat 已经把 messages 装进 body，调用照常；后台 hit_count UPDATE 会找不到行（影响为零，除了一次无效 UPDATE）。可接受。
- 两个 chat 同时 use：两个并发 UPDATE expires_at（最大值生效，不冲突）。可接受。
- 创建时 vLLM 失败：事务回滚，PG 不写入。

## 不做的事（YAGNI）

- ❌ `session` 模式（Step 4 Responses API 覆盖）
- ❌ `truncation_strategy`（仅 session 才用）
- ❌ Cache 跨 instance 共享（隐私边界，复杂权限模型）
- ❌ Cache 共享给其他 user（多租户暂不需要）
- ❌ 预热失败的重试（让用户重 create）
- ❌ Context preview 的完整 messages 返回（YAGNI + 体积）
- ❌ 列表 API `GET /v1/contexts`（Step 6/7 用量查询会覆盖类似需求）
- ❌ Frontend UI（开发者 API，前端不需要单独页面）
- ❌ Streaming context create（无意义，create 是一次性的）

## 验证

```bash
# 1. 创建 cache
RESP=$(curl -s --noproxy '*' -X POST http://localhost:8000/v1/context/create \
  -H "Authorization: Bearer sk-xxx" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-35b","messages":[{"role":"system","content":"<long...>"}]}')
echo "$RESP" | jq .
CTX_ID=$(echo "$RESP" | jq -r .id)

# 2. 用 cache 调 chat（首次应有 cached_tokens > 0，因为 create 已预热）
curl -s --noproxy '*' -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-xxx" -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3.5-35b\",\"context_id\":\"$CTX_ID\",
       \"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" | jq .usage

# 3. 元数据查询
curl -s --noproxy '*' -H "Authorization: Bearer sk-xxx" \
  http://localhost:8000/v1/context/$CTX_ID | jq .hit_count
# 预期: 1

# 4. 错误：找不到的 ctx
curl -s --noproxy '*' -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-xxx" \
  -d '{"model":"qwen3.5-35b","context_id":"ctx-doesnotexist",
       "messages":[{"role":"user","content":"hi"}]}'
# 预期: 404 + {"error":{"type":"not_found_error","code":"context_not_found",...}}

# 5. model 不匹配
curl -s --noproxy '*' -X POST http://localhost:8000/v1/chat/completions \
  -d "{\"model\":\"gemma-4-26b\",\"context_id\":\"$CTX_ID\",
       \"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"
# 预期: 400 + {"error":{"code":"context_model_mismatch",...}}

# 6. 删除
curl -s --noproxy '*' -X DELETE -H "Authorization: Bearer sk-xxx" \
  http://localhost:8000/v1/context/$CTX_ID -w "\nHTTP %{http_code}\n"
# 预期: HTTP 204
```

## 回滚

- 改动局限：1 个新表 + 4 个新端点 + chat_completions 内一段读取/前置逻辑 + 1 个后台 task
- `git revert` + `DROP TABLE context_caches` 即可完全恢复
- 现有 chat_completions 的请求路径在 `context_id` 缺失时完全不变 → 零回归风险
