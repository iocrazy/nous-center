# Step 4 · Responses API（服务端会话管理）

## Context

`/v1/chat/completions` 是无状态的：客户端每次请求都要把完整对话历史塞进 `messages` 数组。一段 20 轮对话（每轮 ~10KB）每次发送都是 200KB 上行流量 + 同等的 vLLM 重新预填充开销——尽管 vLLM 的 prefix caching 能缓解 KV 重算，传输和序列化成本仍然存在，且客户端必须维护历史状态。

OpenAI Responses API（火山方舟同名 API 抄过来）的核心改进是：**服务端管会话**。客户端只传 `previous_response_id` + 本轮新 `input`，服务端拼装完整历史发给模型。配合 Step 3 的 Context Cache，可以做到：
- **固定前缀**（system prompt / 工具定义）走 Context Cache
- **变化历史**（用户与助手往来）走 Responses chain
- **本轮新输入**走 `input` 字段

这是 nous-center 真正"准平台级"差异化能力，也是 Step 1-3 设施投资的最终兑现：错误标准化、用量记录、context cache 在这里全部复用。

## 决策记录

| 决策 | 选择 | 理由 |
|---|---|---|
| 历史存储 | flattened snapshot + gzip bytea | 1 次 PG 读拿全；gzip 压缩 5-6x；写放大可控；实现极简 |
| 与 chat/completions 关系 | 并存独立路由 | 与 OpenAI SDK 命名空间一致（`client.responses.*` vs `client.chat.completions.*`）；底层共享 vLLM 调用 helper |
| 流式 SSE 协议 | 语义事件（response.created / output_text.delta / completed） | 与 OpenAI Responses SDK 兼容；客户端能按事件类型分支 |
| 多模态输入 | 文本 + 图片 URL/base64；`file_id` 预留分支抛 501 | vLLM 已支持 vision；Step 5 接 Files API 时只填 file_id 分支 |
| Context Cache 共存 | 允许，文档约定第一轮用 context_id，后续轮不传 | 不做隐式自动检测；同时给只 warn 不报错 |
| 拼装顺序 | `MESSAGES_ORDER = ["context", "chain", "current_input"]` 显式常量 | 代码意图直观 |
| `store` 默认 | `true` | 与 Ark 一致 |
| `expire_at` 默认 / 上限 | 创建时 + 72h；上限 7 天 (604800s) | 与 Ark 一致 |
| `instructions` | 不继承 previous，每轮独立指定 | 与 Ark 一致 |
| `text.format` | 透传到 vLLM `response_format` | vLLM 原生支持 |
| `thinking` / `reasoning.effort` | 复用 Step 2 thinking 映射；effort 暂忽略（TODO） | 一致性 |
| 鉴权 | Bearer API Key | 与全栈一致，不引入 Access Key |
| 权限范围 | 绑定 `instance_id` | 与 Step 3 一致 |
| 端点 | POST + GET + DELETE + LIST(cursor 分页)；不做 input_items | flattened 方案下 input_items 是冗余；offset 在高写入下漂移 |

## 架构

```
┌────────────── POST /v1/responses ──────────────────────────┐
│  body: { model, input, previous_response_id?, context_id?, │
│          instructions?, thinking?, reasoning?, store?,     │
│          expire_at?, stream?, text? }                      │
│                                                            │
│  1. verify_bearer_token → instance, api_key                │
│  2. normalize input: str -> [{type:input_text,text:...}]   │
│  3. resolve image dispatch (file_id → 501)                  │
│  4. assemble messages = [                                   │
│       (context_id ? cache.messages : []),                  │
│       (previous_response_id ? prev.messages_full : []),    │
│       (instructions ? [{role:system,content:instructions}] : []), │
│       *current input items                                 │
│     ]                                                      │
│  5. inject thinking via Step 2 helper                       │
│  6. proxy to vLLM /v1/chat/completions (stream or not)     │
│  7. record_llm_usage (Step 1+ pattern)                      │
│  8. if store=true:                                          │
│       compress(messages + assistant_reply) -> bytea         │
│       INSERT response_record                                │
└────────────────────────────────────────────────────────────┘

┌────────────── GET /v1/responses/{id} ──────────────────────┐
│  decompress messages_full → return                         │
└────────────────────────────────────────────────────────────┘

┌────────────── GET /v1/responses?limit=N&after=resp-xxx ────┐
│  cursor pagination (after = exclusive cutoff by id)        │
└────────────────────────────────────────────────────────────┘

┌────────────── DELETE /v1/responses/{id} ───────────────────┐
│  idempotent                                                │
└────────────────────────────────────────────────────────────┘

┌── Background: response_records cleanup loop (lifespan) ────┐
│  every hour: DELETE WHERE expire_at < now                  │
└────────────────────────────────────────────────────────────┘
```

## Schema

新增表 `response_records`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | `varchar(64)` PK | `resp-{base64url(snowflake)}` |
| `instance_id` | `bigint` FK→`service_instances.id` ON DELETE CASCADE, indexed | 权限边界 |
| `api_key_id` | `bigint` nullable | 创建者，仅审计 |
| `model` | `varchar(128)` | engine_name 快照 |
| `previous_response_id` | `varchar(64)` nullable, indexed | 上一轮 id（chain 头节点） |
| `context_cache_id` | `varchar(64)` nullable | 创建本轮用的 cache id（仅本轮记录，不溯源）|
| `messages_compressed` | `LargeBinary` (PG bytea) | gzip(json.dumps(messages_full)) |
| `usage_json` | `JSON().with_variant(JSONB, "postgresql")` | `{prompt_tokens, completion_tokens, total_tokens, prompt_tokens_details}` |
| `reasoning_json` | `JSON().with_variant(JSONB, "postgresql")` nullable | thinking summary（如有） |
| `instructions` | `Text` nullable | 本轮 instructions（不参与下一轮） |
| `text_format` | `JSON().with_variant(JSONB, "postgresql")` nullable | text.format 请求参数（如 json_schema） |
| `store` | `Boolean` default `true` | 是否持久化（false 时本行不该存在；保留字段便于将来扩展） |
| `expire_at` | `timestamptz`, indexed | 默认 now+72h，上限 +7d |
| `created_at` | `timestamptz` default now | |

约束：
- CHECK `expire_at > created_at AND expire_at <= created_at + interval '7 days'`
- Index `(expire_at)` 用于清理
- Index `(instance_id, created_at desc)` 用于 LIST 端点
- Index `(instance_id, id)` 用于 cursor pagination + 权限校验

## API 契约

### 1. `POST /v1/responses`

**请求字段**（精简版，对齐 Ark）：
```json
{
  "model": "qwen3.5-35b-a3b-gptq-int4",
  "input": "你好",
  "previous_response_id": null,
  "context_id": null,
  "instructions": null,
  "thinking": {"type": "auto"},
  "reasoning": {"effort": "medium"},
  "store": true,
  "expire_at": null,
  "stream": false,
  "text": {"format": {"type": "text"}}
}
```

`input` 可为 string OR array：
```json
"input": [
  {"type": "input_text", "text": "describe this"},
  {"type": "input_image", "image_url": "data:image/png;base64,...", "detail": "auto"}
]
```

**响应（非流式）：**
```json
{
  "id": "resp-aB3xKp...",
  "object": "response",
  "created_at": 1776220176,
  "model": "qwen3.5-35b-a3b-gptq-int4",
  "previous_response_id": null,
  "instructions": null,
  "store": true,
  "expire_at": 1776479376,
  "output": [
    {
      "type": "message",
      "id": "msg-...",
      "role": "assistant",
      "content": [
        {"type": "output_text", "text": "你好啊", "annotations": []}
      ]
    }
  ],
  "usage": {
    "input_tokens": 32,
    "output_tokens": 7,
    "total_tokens": 39,
    "input_tokens_details": {"cached_tokens": 0}
  }
}
```

**响应（流式 stream=true）—— SSE 语义事件：**
```
event: response.created
data: {"type":"response.created","response":{"id":"resp-x","status":"in_progress",...}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","response_id":"resp-x","delta":"你好"}

event: response.output_text.delta
data: {"type":"response.output_text.delta","response_id":"resp-x","delta":"啊"}

event: response.completed
data: {"type":"response.completed","response":{...full response object...}}

data: [DONE]
```

错误中途出现时（复用 Step 1 SSE wrapper 的格式）：
```
event: error
data: {"type":"error","error":{"message":"...","type":"...","code":"..."}}

data: [DONE]
```

**Errors:**
- 400 `invalid_request_error`：`input` 缺失 / 格式错；`expire_at` 越界
- 400 `context_model_mismatch`：`previous_response_id` 链上的 model 与本次 model 不匹配
- 400 `not_implemented` `code=image_file_id`：`input_image` 用了 `file_id`（Step 5 才开放）
- 401 / 403 / 404 / 429 / 500：常规
- 404 `previous_response_not_found`：`previous_response_id` 不存在或已过期
- 403 `previous_response_wrong_instance`：链上 response 属于另一个 instance

### 2. `GET /v1/responses/{id}`

返回与 POST 响应相同的完整 response object（解压 `messages_compressed`，重构 `output` 字段）。

**Errors:** 404 / 403（同 Context Cache）。

### 3. `GET /v1/responses?limit=20&after=resp-xxx&model=qwen3.5`

Cursor 分页：`after` 是上一页最后一条的 id（exclusive）。返回最近的 N 条 response（按 `created_at desc`）。

```json
{
  "data": [
    {"id":"resp-y","created_at":...,"model":"...","status":"completed",...},
    ...
  ],
  "has_more": true,
  "first_id": "resp-y",
  "last_id": "resp-z"
}
```

注：`data[]` 内**不含 messages**（节省传输）；要细节走 GET by id。

### 4. `DELETE /v1/responses/{id}`

204；幂等。删除会**阻断 chain**——后续以此为 previous_response_id 的请求会 404。**不做级联 chain 删除**（避免误操作影响多个会话）。

## 实现细节

### `messages_compressed` 编解码

```python
import gzip, json

def encode_messages(messages: list[dict]) -> bytes:
    return gzip.compress(json.dumps(messages, ensure_ascii=False).encode("utf-8"))

def decode_messages(data: bytes) -> list[dict]:
    return json.loads(gzip.decompress(data).decode("utf-8"))
```

预期压缩比 5-6x；20 轮 200KB 的 JSON 压到 ~35KB。

### 拼装顺序常量

```python
# src/services/responses_service.py
MESSAGES_ORDER = ("context", "chain", "current_input")  # documentation + linter signal
```

### `input` 字段归一化

```python
def normalize_input(input_field) -> list[dict]:
    """Accept string or array, return list of OpenAI-shape message items."""
    if isinstance(input_field, str):
        return [{"role": "user", "content": [
            {"type": "input_text", "text": input_field}
        ]}]
    if isinstance(input_field, list):
        # Already an array of items; wrap in a user message if items are content-typed
        if all(it.get("type", "").startswith("input_") for it in input_field):
            return [{"role": "user", "content": input_field}]
        # Otherwise assume it's an array of message objects already
        return input_field
    raise InvalidRequestError("input must be a string or array",
                               param="input", code="invalid_input")
```

### `input_image` dispatch

```python
def resolve_image(item: dict) -> dict:
    """Convert input_image item to OpenAI chat-format image_url message content."""
    if item.get("file_id"):
        raise NousError("file_id input not supported until Step 5 (Files API)",
                        code="image_file_id_not_implemented")
        # NousError subclass with http_status=501; add it to errors.py
    if item.get("image_url"):
        return {
            "type": "image_url",
            "image_url": {
                "url": item["image_url"],
                "detail": item.get("detail", "auto"),
            },
        }
    raise InvalidRequestError("input_image requires image_url or file_id",
                               param="input_image", code="invalid_image_input")
```

### Streaming SSE wrapper

新增 `responses_sse_wrapper` 而不是复用 chat 的：事件类型不同。但仍**走 Step 1 的 `sse_with_error_envelope` 风格**（finally 保证 `[DONE]`）。

```python
async def responses_sse_envelope(inner, response_id: str):
    try:
        # First emit response.created
        yield ("response.created",
               {"type":"response.created","response":{"id":response_id,"status":"in_progress"}})
        async for chunk in inner:  # inner yields ("event_type", payload_dict)
            yield chunk
    except NousError as e:
        yield ("error", {"type":"error","error": e.to_dict()["error"]})
    except Exception:
        logger.exception("responses stream failure")
        from src.errors import APIError
        err = APIError("Internal server error", code="internal_error")
        yield ("error", {"type":"error","error": err.to_dict()["error"]})
    finally:
        yield None  # signal terminator
```

实际 `StreamingResponse` 的字节流由一个外层 formatter 把 `(event, payload)` 元组转成 SSE wire format（`event: X\ndata: {...}\n\n`），`None` 转成 `data: [DONE]\n\n`。

### chain walking (虽然存的是 flattened，但需要查上一轮)

```python
async def get_previous_messages(
    session: AsyncSession,
    prev_id: str,
    instance_id: int,
) -> tuple[list[dict], str | None, str]:
    """Returns (messages_full, context_cache_id_if_any, model)."""
    row = await session.get(ResponseRecord, prev_id)
    if row is None or row.expire_at < now() or row.instance_id != instance_id:
        # 404 / 403 mapping
        raise ...
    return decode_messages(row.messages_compressed), row.context_cache_id, row.model
```

### Cursor pagination

```python
async def list_responses(
    session, instance_id, *, limit=20, after=None, model_filter=None,
):
    stmt = select(ResponseRecord).where(
        ResponseRecord.instance_id == instance_id,
        ResponseRecord.expire_at > now(),
    )
    if after:
        # subselect to get cursor row's created_at
        anchor = await session.get(ResponseRecord, after)
        if anchor is None:
            raise InvalidRequestError("invalid cursor", param="after", code="invalid_cursor")
        stmt = stmt.where(
            (ResponseRecord.created_at, ResponseRecord.id)
            < (anchor.created_at, anchor.id)
        )
    stmt = stmt.order_by(
        ResponseRecord.created_at.desc(), ResponseRecord.id.desc()
    ).limit(limit + 1)  # +1 to detect has_more
    rows = (await session.execute(stmt)).scalars().all()
    has_more = len(rows) > limit
    rows = rows[:limit]
    return rows, has_more
```

### 后台清理

复用 Step 3 的 cleanup loop 风格：每小时一次 DELETE WHERE expire_at < now。可以合到同一个 cleanup task 里减少 task 数。

### 关键复用

- `record_llm_usage`（Step 1+） — 每次 vLLM 调用后记录
- `_maybe_inject_thinking`（Step 2） — `thinking` 字段映射
- `sse_with_error_envelope` 思路（Step 1） — SSE 错误兜底
- `verify_bearer_token` — 鉴权
- `fetch_active_cache`（Step 3） — context_id 解析
- `JsonColumn = JSON().with_variant(JSONB, "postgresql")`（Step 3） — JSON 字段在 SQLite/PG 兼容

## 不做的事（YAGNI）

- ❌ `input_items` 端点（flattened 方案下 GET 整个 response 已包含全部 input）
- ❌ Access Key / IAM 鉴权（与全栈对齐，仅 Bearer）
- ❌ `file_id` 输入（Step 5 来了再做，预留 501 stub）
- ❌ `truncation` / `max_output_tokens` 自动管理（用户自己控制 max_tokens）
- ❌ 自动 chain 删除（DELETE 只删本条，不影响其他）
- ❌ 跨 instance cache 共享
- ❌ 多语言 SSE event 名（统一英文）
- ❌ Frontend UI（开发者 API）
- ❌ `effort` 字段（vLLM 当前无对应参数；记 TODO）
- ❌ Function Calling / Tools 字段（巨大子项目，单独 Step）

## 验证

```bash
KEY=sk-qwen-xxx

# 1. 简单一轮
curl --noproxy '*' -X POST http://localhost:8000/v1/responses \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5","input":"你好","store":true}' | jq .
# → resp-xxx, output[0].content[0].text

# 2. 多轮（chain）
RESP1=$(curl --noproxy '*' -X POST .../v1/responses \
  -d '{"model":"qwen3.5","input":"我叫小明","store":true}' | jq -r .id)
curl --noproxy '*' -X POST .../v1/responses \
  -d "{\"model\":\"qwen3.5\",\"input\":\"我叫什么？\",\"previous_response_id\":\"$RESP1\"}"
# → 应回答"小明"

# 3. 流式
curl --noproxy '*' -N -X POST .../v1/responses \
  -d '{"model":"qwen3.5","input":"讲个笑话","stream":true}'
# → event: response.created / output_text.delta * N / completed / [DONE]

# 4. context_id + chain（首轮用 context，二轮用 chain）
CTX=$(curl ... /v1/context/create -d '{...JSON only system...}' | jq -r .id)
R1=$(curl ... /v1/responses -d "{\"model\":\"...\",\"input\":\"hi\",\"context_id\":\"$CTX\"}" | jq -r .id)
curl ... /v1/responses -d "{\"model\":\"...\",\"input\":\"again\",\"previous_response_id\":\"$R1\"}"
# 二轮不传 context_id；按文档约定 chain 已包含全历史

# 5. 多模态
curl ... /v1/responses -d '{"model":"qwen3.5-vl","input":[
  {"type":"input_text","text":"What is in this image?"},
  {"type":"input_image","image_url":"https://..."}
]}'

# 6. file_id stub
curl ... /v1/responses -d '{"model":"...","input":[
  {"type":"input_image","file_id":"file-xxx"}
]}'
# → 501 + code=image_file_id_not_implemented

# 7. LIST 分页
curl ... "/v1/responses?limit=5"     # 第 1 页
curl ... "/v1/responses?limit=5&after=resp-y"  # 下一页

# 8. DELETE + chain 阻断
curl ... -X DELETE /v1/responses/$R1
# 之后用 R1 做 previous_response_id → 404
```

## 回滚

- 一个新表 + 一个新路由文件 + main.py 注册一行 + lifespan cleanup loop 一段
- `git revert` + `DROP TABLE response_records` 全恢复
- 不修改 chat/completions / context cache —— 零回归风险
