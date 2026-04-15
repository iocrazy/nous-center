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
| 历史存储 | event-sourcing：`response_sessions` + `response_turns`（每轮 gzip） | 借鉴 Claude Code 的 JSONL 模式但落地到 PG；写放大归零；天然支持部分读 / FTS / psql 调试 |
| API 兼容字段名 | 对外保留 `previous_response_id` (= `resp-xxx` turn id)；内部 `session_id` 隐藏 | 与 OpenAI Responses SDK 一致；service 层做 turn→session 映射 |
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
| 自动 compaction | 拼装 messages 时按 token 计数，超 `max_history_tokens` 丢最旧轮（保留 system / context_id 注入的内容） | 借鉴 Claude Code，避免长会话撞 vLLM `max_model_len` 直接报错 |
| Session 总预算 | `response_sessions.total_input_tokens` + `total_output_tokens` 每轮累加；超 `MAX_SESSION_TOKENS`（默认 200000，可改）拒绝新轮 | 防止 chain 跑飞、bug 重试爆 token；与 OpenAI 现有 model 一致量级 |
| 停止原因 | 顶层 `status: "completed" \| "incomplete"` + `incomplete_details: {reason}` | 对齐 OpenAI Responses API；区分 `max_output_tokens` / `content_filter` / `history_truncated` / `session_budget_exceeded` / `connection_closed` |
| 流式中断行为 | 写入 partial assistant turn + `status="incomplete"` + `incomplete_details.reason="connection_closed"` | 客户端断连后能查到部分内容；下次用 previous_response_id 继续不会失败 |
| 并发写冲突 | UNIQUE(session_id, turn_idx) 撞了 → 409 `ConflictError` `code="session_concurrent_write"` | 需要 Step 1 errors.py 加 `ConflictError`（http_status=409） |
| 单 input 超模型上限 | compaction 后仍超 → `InvalidRequestError code="input_too_long_for_model"` | 不发 vLLM；明确错误信息给用户 |
| Gzip 解压上限 | `gzip.decompress(data, max_length=10_000_000)` | 防 zip bomb（虽然攻击面窄，零代价加固） |
| `response.created` emit 时机 | 拼装 + vLLM 首字节成功后再发 | 失败时只发 error event，避免客户端 state machine 看到 created→error 序列困惑 |
| PII 数据保留 | 文档显式警告 72h 保留期；redaction 列 TODO | 私有/小团队场景接受默认 expire_at + DELETE 端点；将来加 message-level redaction |

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

两张新表（event-sourcing）：

### `response_sessions` — 一行一个会话

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | `varchar(64)` PK | `session-{base64url(snowflake)}`，**仅内部使用，不出现在 API** |
| `instance_id` | `bigint` FK→`service_instances.id` ON DELETE CASCADE, indexed | 权限边界 |
| `api_key_id` | `bigint` nullable | 创建者审计 |
| `model` | `varchar(128)` | engine_name 快照（一个 session 内不可改 model） |
| `context_cache_id` | `varchar(64)` nullable | 首轮 context_id 记录（用于审计，不溯源） |
| `total_input_tokens` | `bigint` default 0 | 累计输入 token（含 cache 命中部分） |
| `total_output_tokens` | `bigint` default 0 | 累计输出 token |
| `expire_at` | `timestamptz`, indexed | 默认 now+72h，上限 +7d |
| `created_at` | `timestamptz` default now | |

### `response_turns` — 一行一轮（user 一行 + assistant 一行）

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | `varchar(64)` PK | `resp-{base64url(snowflake)}` —— **对外 API 暴露的 response id** |
| `session_id` | `varchar(64)` FK→`response_sessions.id` ON DELETE CASCADE, indexed | |
| `turn_idx` | `int` | 0, 1, 2... session 内单调递增；UNIQUE(session_id, turn_idx) |
| `role` | `varchar(20)` | `user` / `assistant` / `system` |
| `content_compressed` | `LargeBinary` (PG bytea) | gzip(json.dumps(content_array)) — 单轮内容，体积小 |
| `usage_json` | `JsonColumn` nullable | 仅 assistant turn 有：`{input_tokens, output_tokens, ...}` |
| `reasoning_json` | `JsonColumn` nullable | 仅 assistant：thinking summary |
| `instructions` | `Text` nullable | 仅 assistant：本轮 instructions（不参与下一轮） |
| `text_format` | `JsonColumn` nullable | 仅 assistant：text.format 参数 |
| `created_at` | `timestamptz` default now | |

约束：
- `response_sessions.expire_at` CHECK：`> created_at`（仅基础约束；上限 7 天在 application 层强制，避免 PG `interval` 语法在 SQLite test fixture 失败）
- `UNIQUE(session_id, turn_idx)`
- `INDEX(session_id, turn_idx)` —— 拼装历史的主查询
- `INDEX(expire_at)` 在 `response_sessions` 上，用于清理
- `INDEX(instance_id, created_at desc)` 在 `response_sessions` 上，用于 LIST

### API 与内部映射

- `previous_response_id` (传入) = `response_turns.id` ("resp-xxx")
- 拿到 turn 行 → 通过 `session_id` 找到 session → 同 session 拉所有 turns（按 turn_idx 升序）→ 构造 messages 数组
- 一轮新对话 = 新建 session **或** 复用 session（如果传了 previous_response_id）
- 每次 `POST /v1/responses` 写入：1 个新 user turn + 1 个新 assistant turn（成功响应后）；`session` 行只在新 session 时插入

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
  "status": "completed",
  "incomplete_details": null,
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
    "input_tokens_details": {"cached_tokens": 0},
    "session_total_input_tokens": 132,
    "session_total_output_tokens": 41
  },
  "history_truncated": false
}
```

**`status` 与 `incomplete_details`：**
| status | incomplete_details.reason | 触发条件 |
|---|---|---|
| `completed` | `null` | 模型自然停止（finish_reason=stop） |
| `incomplete` | `max_output_tokens` | 模型达到 max_tokens 截断（finish_reason=length） |
| `incomplete` | `content_filter` | vLLM/safety 触发 |
| `incomplete` | `history_truncated` | 拼装时 compaction 丢了旧轮（warning 信号） |

**`history_truncated: true`** 是顶层 boolean，方便客户端快速检测；详细原因看 `incomplete_details`。

**Session 预算超限：** 在新轮**开始前**检查 `total_input_tokens + estimated_new_input > MAX_SESSION_TOKENS`，超则不调用模型，直接返回：
```json
{"error": {
  "message": "Session token budget exceeded (200000)",
  "type": "rate_limit_error",
  "code": "session_budget_exceeded",
  "request_id": "..."
}}
```
HTTP 429。客户端应该开新 session（不传 previous_response_id）。

**响应（流式 stream=true）—— SSE 语义事件：**

字段对齐 OpenAI Responses SDK 的事件 schema（`item_id`, `output_index`, `content_index` 必须有，否则 SDK 不识别）：
```
event: response.created
data: {"type":"response.created","response":{"id":"resp-x","status":"in_progress","model":"...","created_at":...}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg-x","type":"message","role":"assistant","content":[]}}

event: response.content_part.added
data: {"type":"response.content_part.added","item_id":"msg-x","output_index":0,"content_index":0,"part":{"type":"output_text","text":""}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","item_id":"msg-x","output_index":0,"content_index":0,"delta":"你好"}

event: response.output_text.delta
data: {"type":"response.output_text.delta","item_id":"msg-x","output_index":0,"content_index":0,"delta":"啊"}

event: response.completed
data: {"type":"response.completed","response":{...full response object with output[]...}}

data: [DONE]
```

`item_id` / `output_index` / `content_index` 现在都填 0 / `msg-x`（单 output item, 单 content part），但 schema 提前留好；将来 tool calls / multi-output 时这些索引才有意义。

**`request_id` 注入**：wrapper 自动给每个 event payload 的顶层加 `request_id` 字段（来自 RequestIdMiddleware），客户端可对接日志追踪。

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

返回与 POST 响应相同的完整 response object（解压 `content_compressed`，重构 `output` 字段）。

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

204；幂等。

**语义：删除整个会话**（不只是这一轮）。在 event-sourcing 模型下，单独删一个 turn 会让 chain 出现"洞"导致历史拼装错乱；而通常用户的意图就是"我不要这段对话了"。

实现：通过 turn id 找到 session_id → `DELETE FROM response_sessions WHERE id = X`（FK ON DELETE CASCADE 自动清掉所有 turns）。

**这是与 OpenAI Responses API 的语义偏差**——它的 DELETE 只删一行 response。我们因为存储模型不同，必须级联。文档里要明确说清。

## 实现细节

### 单轮内容编解码

每个 `response_turns.content_compressed` 存的是**单轮 content 数组** —— 比 snapshot 小一个量级（typically 几百字节到几 KB）。

```python
import gzip, json

def encode_content(content: list[dict]) -> bytes:
    return gzip.compress(json.dumps(content, ensure_ascii=False).encode("utf-8"))

def decode_content(data: bytes, max_size: int = 10_000_000) -> list[dict]:
    # NOTE: gzip.decompress() in Py3.12 has NO max_length kwarg.
    # Use streaming GzipFile + bounded read to enforce the cap.
    import io
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
        out = gz.read(max_size + 1)
    if len(out) > max_size:
        from src.errors import InvalidRequestError
        raise InvalidRequestError(
            "decompressed payload too large",
            code="payload_too_large",
        )
    return json.loads(out.decode("utf-8"))
```

短 content（< 500 字节）gzip 收益不大但也不亏，统一压便于 schema 一致。

### 历史拼装（注意 instructions 处理）

**关键约束：** `instructions` 字段每轮独立，**不继承** previous（决策记录）。
所以拼装时**只取 user / assistant turns**；instructions 只用本轮请求里传的（如有）放在最前。
绝对不要把 previous turn 的 instructions（存在 `response_turns.instructions`）拼回 messages。

```python
async def assemble_history_for_response(
    session: AsyncSession, prev_resp_id: str | None, instance_id: int,
) -> tuple[list[dict], "ResponseSession | None"]:
    """Walk backwards from prev_resp_id to assemble messages for the next request."""
    if not prev_resp_id:
        return [], None
    prev_turn = await session.get(ResponseTurn, prev_resp_id)
    if prev_turn is None:
        raise NotFoundError("previous_response_not_found", code="previous_response_not_found")
    sess = await session.get(ResponseSession, prev_turn.session_id)
    if sess is None or sess.expire_at < now() or sess.instance_id != instance_id:
        # 404 or 403
        raise ...
    # Single SELECT ordered by turn_idx
    stmt = (
        select(ResponseTurn)
        .where(ResponseTurn.session_id == sess.id)
        .order_by(ResponseTurn.turn_idx.asc())
    )
    rows = (await session.execute(stmt)).scalars().all()
    messages = []
    for r in rows:
        if r.role not in ("user", "assistant"):
            # Defense-in-depth: skip any non-conversational rows
            continue
        messages.append({
            "role": r.role,
            "content": decode_content(r.content_compressed),
        })
    return messages, sess
```

PG 单 SELECT 排序扫描，毫秒级；行级 gzip 解压在 Python 侧并行（用 async 内的 thread executor 也行，通常不需要）。

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

新增 `responses_sse_envelope` 而不是复用 chat 的：事件类型不同。**关键三件事：**

1. **wrapper 直接 yield 完整 SSE 字符串**（不是 tuple），与 Step 1 的 `sse_with_error_envelope` 输出格式保持一致 —— Starlette `StreamingResponse` 直接吃 string/bytes 生成器，不需要外层 formatter。
2. **`response.created` 时机：** 由 `inner` 在拿到 vLLM 首字节后才 yield，wrapper 不主动注入。失败时只有 `error` 事件，避免 `created→error` 的混乱状态机。
3. **`asyncio.CancelledError` 持久化：** 取消语义下，await PG INSERT **可能本身被立即取消**。不能依赖 except CancelledError 分支里 await。改用**模块级 background task queue**（在 lifespan 启动时拉起 worker）接收 partial-write 任务。

```python
import json
from src.errors import APIError, NousError

# Module-level queue + worker — survives request cancellation
_partial_write_queue: asyncio.Queue | None = None

async def _partial_write_worker():
    """Started in lifespan. Drains partial-write requests serially."""
    while True:
        item = await _partial_write_queue.get()
        if item is None:  # shutdown sentinel
            break
        try:
            persist_fn, args = item
            await persist_fn(*args)
        except Exception:
            logger.exception("partial-write worker failed")
        finally:
            _partial_write_queue.task_done()

def schedule_partial_write(persist_fn, *args):
    """Fire-and-forget; survives request task cancellation."""
    if _partial_write_queue is not None:
        try:
            _partial_write_queue.put_nowait((persist_fn, args))
        except asyncio.QueueFull:
            logger.error("partial-write queue full, dropping")


def _sse_format(evt: str, payload: dict) -> str:
    return f"event: {evt}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def responses_sse_envelope(inner, persist_partial_fn, request_id: str | None):
    """Wrap an async-generator that yields ('event_type', dict) tuples and
    serialize to SSE wire format. Always emits exactly one `data: [DONE]\\n\\n`.

    Args:
      inner: async iter of (evt_type, payload_dict). MUST emit response.created
             AFTER first vLLM byte (not at start).
      persist_partial_fn: async callable(text_so_far, status, reason); enqueued
             onto background worker queue on cancellation.
      request_id: injected into every event payload as 'request_id' field.
    """
    accumulated_text = ""
    sent_done = False
    try:
        async for evt_type, payload in inner:
            payload = dict(payload)  # don't mutate caller's dict
            if request_id and "request_id" not in payload:
                payload["request_id"] = request_id
            if evt_type == "response.output_text.delta":
                accumulated_text += payload.get("delta", "")
            yield _sse_format(evt_type, payload)
    except asyncio.CancelledError:
        # Client disconnected mid-stream. Hand off to background worker because
        # awaiting PG inside this except block is unreliable under cancellation.
        schedule_partial_write(
            persist_partial_fn,
            accumulated_text,
            "incomplete",
            "connection_closed",
        )
        raise
    except NousError as e:
        err_payload = {"type": "error", "error": e.to_dict()["error"]}
        if request_id:
            err_payload["request_id"] = request_id
        yield _sse_format("error", err_payload)
    except Exception:
        logger.exception("responses stream failure")
        err = APIError("Internal server error", code="internal_error")
        err_payload = {"type": "error", "error": err.to_dict()["error"]}
        if request_id:
            err_payload["request_id"] = request_id
        yield _sse_format("error", err_payload)
    finally:
        if not sent_done:
            yield "data: [DONE]\n\n"
            sent_done = True
```

**lifespan 集成：** `main.py` 的 `lifespan` 启动时初始化 queue 和 worker，shutdown 时发 sentinel + await worker：
```python
import src.api.routes.responses as responses_routes
responses_routes._partial_write_queue = asyncio.Queue(maxsize=1000)
worker = asyncio.create_task(responses_routes._partial_write_worker())
try:
    yield
finally:
    await responses_routes._partial_write_queue.put(None)  # sentinel
    await asyncio.wait_for(worker, timeout=5.0)
```

实际 `StreamingResponse` 的字节流由一个外层 formatter 把 `(event, payload)` 元组转成 SSE wire format（`event: X\ndata: {...}\n\n`），`None` 转成 `data: [DONE]\n\n`。

### 自动 compaction（防止撞 max_model_len）

历史拼装后、调用 vLLM 前，检查累计 token 数。超阈值则丢最旧的非系统轮：

```python
# src/services/responses_service.py

def _approx_tokens(messages: list[dict]) -> int:
    """Quick OVER-estimate of token count, kept conservative.

    Calibration: BPE tokenizers vary, but for safety we assume the worst:
    - English-heavy text: ~1 token per 4 chars (so len/4 underestimates;
      we use len/2 to over-estimate by ~2x).
    - Chinese text: ~1.5 tokens per char. len/2 is still safe (overshoots
      by ~25%).
    - Mixed text: covered by len/2.

    Result: we may compact MORE aggressively than necessary, which means
    the model gets less context but never crashes on context_length_exceeded.
    For an exact count, fetch from adapter.tokenizer when available.
    """
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c) // 2 + 4  # +4 for role/format overhead
        elif isinstance(c, list):
            for item in c:
                t = item.get("text", "")
                if isinstance(t, str):
                    total += len(t) // 2 + 4
                else:
                    total += 200  # image / other content placeholder (vision tokens vary)
    return total

def compact_messages(
    messages: list[dict],
    *,
    max_history_tokens: int,
    keep_system: bool = True,
) -> tuple[list[dict], bool]:
    """Drop oldest turns until token estimate fits.
    Always keeps the first system message + the most recent user/assistant pair.
    Returns (compacted_messages, was_truncated).
    """
    if _approx_tokens(messages) <= max_history_tokens:
        return messages, False

    system_msgs = [m for m in messages if m.get("role") == "system"] if keep_system else []
    rest = [m for m in messages if m.get("role") != "system"]

    # Keep dropping oldest non-system turns until under budget
    while rest and _approx_tokens(system_msgs + rest) > max_history_tokens:
        rest.pop(0)
    # Safety: if even system + last turn overflows, force-keep last 1 turn anyway
    if not rest and len(messages) > 0:
        rest = messages[-1:]
    return system_msgs + rest, True
```

`max_history_tokens` 默认 = `adapter.max_model_len - 2048`（留 2K 给输出）。可改成请求级 query param 但 YAGNI。

**单 input 超限兜底：** compaction 之后再做一次估算，如果**单轮 input alone** 超 `max_history_tokens`，**不发 vLLM**，直接抛：
```python
if _approx_tokens(messages_after_compaction) > max_history_tokens:
    raise InvalidRequestError(
        f"input alone ({_approx_tokens(messages)} tokens) exceeds "
        f"max_history_tokens ({max_history_tokens})",
        code="input_too_long_for_model",
        param="input",
    )
```
否则 vLLM 会回 context_length_exceeded，客户端拿到含糊的 upstream error。

`history_truncated=True` 时：
- 顶层 response 字段 `history_truncated: true`
- 如果同时模型也截断输出 → `status="incomplete"`, `incomplete_details.reason="history_truncated"`
- 否则 status 仍可为 `completed`（只是历史变短了），但 `history_truncated` 提示用户

### Session 预算检查

```python
SESSION_TOKEN_BUDGET = 200_000  # roughly 6x typical model context

async def check_session_budget(
    session: AsyncSession, sess: ResponseSession, estimated_new: int,
) -> None:
    projected = sess.total_input_tokens + sess.total_output_tokens + estimated_new
    if projected > SESSION_TOKEN_BUDGET:
        raise RateLimitError(
            f"Session token budget exceeded ({SESSION_TOKEN_BUDGET})",
            code="session_budget_exceeded",
        )

async def update_session_usage(
    session: AsyncSession, sess: ResponseSession,
    input_tokens: int, output_tokens: int,
) -> None:
    """Atomic UPDATE post-vLLM call."""
    stmt = update(ResponseSession).where(ResponseSession.id == sess.id).values(
        total_input_tokens=ResponseSession.total_input_tokens + input_tokens,
        total_output_tokens=ResponseSession.total_output_tokens + output_tokens,
    ).execution_options(synchronize_session=False)
    await session.execute(stmt)
    await session.commit()
```

### 写入新轮（一次 POST 完成 user + assistant）

```python
async def write_response_turn(
    session: AsyncSession, *, sess: "ResponseSession", user_content: list[dict],
    assistant_content: list[dict], usage: dict, reasoning: dict | None,
    instructions: str | None, text_format: dict | None,
) -> tuple["ResponseTurn", "ResponseTurn"]:
    """Insert two new turns (user + assistant) and return them."""
    # Find next turn_idx (lock-free; UNIQUE catches concurrent writes)
    last_idx = (await session.execute(
        select(func.max(ResponseTurn.turn_idx))
        .where(ResponseTurn.session_id == sess.id)
    )).scalar() or -1
    user_turn = ResponseTurn(
        id=_new_turn_id(), session_id=sess.id, turn_idx=last_idx + 1,
        role="user", content_compressed=encode_content(user_content),
    )
    asst_turn = ResponseTurn(
        id=_new_turn_id(), session_id=sess.id, turn_idx=last_idx + 2,
        role="assistant", content_compressed=encode_content(assistant_content),
        usage_json=usage, reasoning_json=reasoning, instructions=instructions,
        text_format=text_format,
    )
    session.add_all([user_turn, asst_turn])
    try:
        await session.commit()
    except IntegrityError as e:
        # UNIQUE(session_id, turn_idx) collision: another request grabbed the
        # same idx between our SELECT max() and INSERT. Surface as 409 so the
        # client can refetch the head and retry.
        await session.rollback()
        if "turn_idx" in str(e).lower() or "uniq" in str(e).lower():
            raise ConflictError(
                "concurrent write to the same session; refetch and retry",
                code="session_concurrent_write",
            )
        raise
    return user_turn, asst_turn
```

**对外返回的 `id` 是 `asst_turn.id`** —— 客户端拿这个作为下一轮的 `previous_response_id`。

**Step 1 反向追加要求：** 在 `backend/src/errors.py` 加：
```python
class ConflictError(NousError):
    type = "invalid_request_error"  # match _HTTP_STATUS_TO_ERROR[409]
    http_status = 409
```
不需要改 `main.py`。`_nous` exception handler 已经从 `exc.http_status` 取状态码，409 会正确返回。`code` 在 raise 处指定（如 `code="session_concurrent_write"`），不在 class 上设默认值。

### Cursor pagination

LIST 端点列出**最近的 assistant turn**（每个 session 的最后一轮，按 created_at desc）。Cursor 是 `resp-xxx` (turn id)。

```python
async def list_responses(
    session, instance_id, *, limit=20, after=None, model_filter=None,
):
    # Join turn -> session for instance scope
    stmt = (
        select(ResponseTurn).join(
            ResponseSession, ResponseTurn.session_id == ResponseSession.id
        ).where(
            ResponseSession.instance_id == instance_id,
            ResponseSession.expire_at > now(),
            ResponseTurn.role == "assistant",
        )
    )
    if after:
        anchor = await session.get(ResponseTurn, after)
        if anchor is None:
            raise InvalidRequestError("invalid cursor", param="after", code="invalid_cursor")
        # IMPORTANT: must use sqlalchemy.tuple_ for row-constructor comparison.
        # Bare Python tuple comparison is NOT translated to SQL — it would
        # compare Column objects in Python, producing nonsense.
        from sqlalchemy import tuple_
        stmt = stmt.where(
            tuple_(ResponseTurn.created_at, ResponseTurn.id)
            < tuple_(anchor.created_at, anchor.id)
        )
    stmt = stmt.order_by(
        ResponseTurn.created_at.desc(), ResponseTurn.id.desc()
    ).limit(limit + 1)
    rows = (await session.execute(stmt)).scalars().all()
    has_more = len(rows) > limit
    return rows[:limit], has_more
```

### 后台清理

复用 Step 3 的 cleanup loop 风格：每小时一次 DELETE WHERE expire_at < now。可以合到同一个 cleanup task 里减少 task 数。

### 抽取共享 context 解析 helper（避免代码重复）

`openai_compat.py:132-179` 现有的 `context_id` 处理和 Step 4 即将写的几乎一样（约 50 行）。**重构到 `services/context_cache_service.py`**：

```python
async def resolve_for_request(
    session: AsyncSession,
    *,
    context_id: str | None,
    instance_id: int,
    engine_name: str,
) -> tuple[list[dict] | None, int | None]:
    """Common helper: lookup context cache, validate, return (messages, ttl).
    Raises NotFoundError / PermissionError / InvalidRequestError on failure.
    Used by both /v1/chat/completions and /v1/responses.
    """
    if not context_id:
        return None, None
    cache = await fetch_active_cache(session, context_id, instance_id)
    if cache is None:
        other = await fetch_cache_any_instance(session, context_id)
        if other is not None and other.instance_id != instance_id:
            raise PermissionError("...", code="context_wrong_instance")
        raise NotFoundError("Context cache not found or expired",
                            code="context_not_found")
    if cache.model != engine_name:
        raise InvalidRequestError(
            f"Cache was created for '{cache.model}', not '{engine_name}'",
            code="context_model_mismatch",
            param="model",
        )
    return list(cache.messages_json), cache.ttl_seconds
```

Step 4 的 chat 集成调用这个 helper；Step 1 实施时把 openai_compat.py 内联那段也替换为调用 helper。**减重约 50 行**。

### 关键复用

- `record_llm_usage`（Step 1+） — 每次 vLLM 调用后记录
- `_maybe_inject_thinking`（Step 2） — `thinking` 字段映射
- `sse_with_error_envelope` 思路（Step 1） — SSE 错误兜底
- `verify_bearer_token` — 鉴权
- `fetch_active_cache`（Step 3） — context_id 解析
- `JsonColumn = JSON().with_variant(JSONB, "postgresql")`（Step 3） — JSON 字段在 SQLite/PG 兼容

## 数据保留与 PII 警告

`response_turns.content_compressed` 存了用户与模型完整对话内容（gzip 压缩，**未加密**）。
默认 `expire_at = created_at + 72h`，到期由 cleanup loop 物理删除。

**用户应知道的：**
- 对话内容在 PG 里保留至多 7 天（`expire_at` 上限）
- 任何持有 instance API Key 的人都能 GET 历史 response（即使是别人的 chat）
- **对话中请勿包含真实凭据 / API Keys / PII** —— 当前没有 message-level redaction

**响应文档：** 4 个端点的对外文档（README / OpenAPI）必须显式说明这一点。

**TODO（不在本 spec 范围内）：**
- Message-level redaction: pre-write 扫描 content，把疑似 secrets 替换为 `[REDACTED]`
- 加密静态存储（PG bytea + AES）—— 需要 KMS 集成
- 立即清空入口（`POST /v1/sessions/clear-all`）—— 紧急合规场景

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

# 9. Auto-compaction 触发（短模型 + 长 chain）
# 在 max_model_len ~ 4096 的小模型上跑 30 轮长 prompt，
# 第 ~15 轮起响应应有 history_truncated: true
# 但 status 仍为 completed（除非输出也被 length 截断）

# 10. Session 预算超限
# 累计跑超 200K token 的 chain
# 下一个 POST → 429 + code=session_budget_exceeded

# 11. 流式中断保留 partial
# stream=true 调用，客户端在第 50 个 delta 时 SIGINT
# 服务端日志应有 "partial assistant turn written, status=incomplete"
# 之后 GET /v1/responses/{id} 应返回 status="incomplete",
#   incomplete_details.reason="connection_closed", output 含已收到的文本

# 12. 单 input 超模型上限
# 构造 messages 含 100K token 的单条 user input（mock_model_len=4096 时）
# POST → 400 + code=input_too_long_for_model
# vLLM 不应被调用（log 里不该出现 vLLM upstream call）

# 13. 并发写撞 UNIQUE
# 用同一 previous_response_id 并发发 2 个 POST
# 一个返回 200, 另一个返回 409 + code=session_concurrent_write
```

## 回滚

- 一个新表 + 一个新路由文件 + main.py 注册一行 + lifespan cleanup loop 一段
- `git revert` + `DROP TABLE response_records` 全恢复
- 不修改 chat/completions / context cache —— 零回归风险
