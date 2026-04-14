# Step 1 · 错误码规范化（OpenAI 风格）

## Context

当前后端错误响应混用三种格式：
- FastAPI 默认 `{"detail": "..."}`（多数路由 `raise HTTPException`）
- 部分自定义 `{"error": "..."}`（某些路由手动构造）
- 裸字符串 / 字典（少量边缘路径）

前端 `apiFetch` 捕获错误时不得不做 `detail || error || String(e)` 兜底；toast 只能显示原始文本，无法按类型做差异化处理（如 rate limit 应倒计时，auth 失败应跳登录）。

用 OpenAI SDK 调用我们 `/v1/chat/completions` 的客户端会期望 `{error: {message, type, code}}` 结构，当前我们不符合。

目标：全栈（`/v1/*` + `/api/v1/*`）统一成 OpenAI 风格错误 payload，加 `X-Request-Id` 方便追踪。现有路由代码**不需要改**，通过全局 Exception Handler 在边界处统一转换。

这是后续所有升级（Context Cache / Responses API / Files API 等）的基础——它们都会复用同一套错误类型。

## 决策记录

| 决策 | 选择 | 理由 |
|---|---|---|
| 错误 schema | OpenAI 风格 `{error: {message, type, code, param, request_id}}` | 对齐 OpenAI SDK，前端能按 `type` 做分支 |
| 改造范围 | 全栈 `/v1/*` + `/api/v1/*` | 一套规则，前端适配一次 |
| 实现机制 | 全局 `ExceptionHandler` + RequestID `Middleware` | 零侵入，现有 `raise HTTPException` 自动转换 |
| 错误类型枚举 | 6 类（OpenAI 标准） | 覆盖 95% 场景：`invalid_request_error` / `authentication_error` / `permission_error` / `not_found_error` / `rate_limit_error` / `api_error` |
| `code` 字段 | 自由填（snake_case 约定） | 不预注册 enum；用于文档和前端跳转逻辑 |
| HTTP 状态码 | 由 error type 固定决定 | 400/401/403/404/429/500 |
| Request ID | UUID；header 读写 `X-Request-Id` | 追踪调用链；错误 payload 带上 |

## 架构

```
┌───────────────── Request ─────────────────┐
│  X-Request-Id: <UUID>  (可选)             │
└───────────────────┬───────────────────────┘
                    ▼
        ┌──── RequestIdMiddleware ────┐
        │ request.state.request_id    │
        └──────────────┬──────────────┘
                       ▼
              ┌─── Route handler ───┐
              │  raises:            │
              │  - NousError (新)   │
              │  - HTTPException    │
              │  - RequestValidation│
              │    Error (Pydantic) │
              │  - Exception 兜底   │
              └──────────┬──────────┘
                         ▼
     ┌───── 4 个 ExceptionHandler (全局) ─────┐
     │  1. NousError → payload.to_dict()     │
     │  2. HTTPException → 按 status 映射    │
     │  3. RequestValidationError → 400      │
     │  4. Exception → 500 + 日志 traceback  │
     │                                       │
     │  统一注入 request_id                  │
     │  统一设置 X-Request-Id header         │
     │  500 case: log → app_logs table       │
     │           response: generic message    │
     └──────────────┬────────────────────────┘
                    ▼
┌──────── Response (JSON 或 SSE) ────────┐
│ X-Request-Id: <UUID>                   │
│ { "error": { ...OpenAI shape... } }    │
│                                        │
│ (SSE 流式错误: data: {"error":{...}}\n │
│              + data: [DONE]\n\n)       │
└────────────────────────────────────────┘
```

## 关键文件

### 后端

| 路径 | 说明 |
|---|---|
| `backend/src/errors.py`（新） | `NousError` 基类 + 6 个子类 |
| `backend/src/api/middleware/request_id.py`（新） | 读/生成 `X-Request-Id`，写入 `request.state` + response header |
| `backend/src/api/main.py`（修改） | 注册 middleware + 3 个 exception handler（NousError / HTTPException / Exception 兜底） |

**`errors.py` 主体：**

```python
class NousError(Exception):
    type: str = "api_error"
    http_status: int = 500

    def __init__(self, message: str, *, code: str | None = None,
                 param: str | None = None, request_id: str | None = None):
        self.message = message
        self.code = code
        self.param = param
        self.request_id = request_id
        super().__init__(message)

    def to_dict(self) -> dict:
        err: dict = {"message": self.message, "type": self.type}
        if self.code: err["code"] = self.code
        if self.param: err["param"] = self.param
        if self.request_id: err["request_id"] = self.request_id
        return {"error": err}


class InvalidRequestError(NousError):  type = "invalid_request_error"; http_status = 400
class AuthenticationError(NousError):  type = "authentication_error"; http_status = 401
class PermissionError(NousError):      type = "permission_error";     http_status = 403
class NotFoundError(NousError):        type = "not_found_error";      http_status = 404
class RateLimitError(NousError):       type = "rate_limit_error";     http_status = 429
class APIError(NousError):             type = "api_error";            http_status = 500
```

**HTTPException → NousError 映射（在 handler 里）：**

| HTTP status | 转成的 NousError type |
|---|---|
| 400 | `invalid_request_error` |
| 401 | `authentication_error` |
| 403 | `permission_error` |
| 404 | `not_found_error` |
| 429 | `rate_limit_error` |
| 其他 4xx | `invalid_request_error` |
| 5xx | `api_error` |

**`HTTPException.detail` 可能是 str 或 list**（Pydantic 错误时 FastAPI 会塞 list）。handler 需：
```python
if isinstance(exc.detail, str):
    message = exc.detail
elif isinstance(exc.detail, list):
    # 通常是 [{"loc":[...], "msg":"...", "type":"..."}]
    message = "; ".join(e.get("msg", str(e)) for e in exc.detail)
else:
    message = str(exc.detail)
```

**`RequestValidationError` 单独处理**（Pydantic 请求体校验失败，不会走 HTTPException 路径）：
```python
@app.exception_handler(RequestValidationError)
async def validation_exc(request, exc):
    errors = exc.errors()
    first = errors[0] if errors else {}
    loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
    err = InvalidRequestError(
        message=first.get("msg", "Invalid request"),
        code="validation_error",
        param=loc or None,
    )
    err.request_id = getattr(request.state, "request_id", None)
    return JSONResponse(err.to_dict(), status_code=400,
                        headers={"X-Request-Id": err.request_id or ""})
```

**500 兜底 handler 的 traceback 策略（安全）：**
- 响应 payload **只返 generic message**：`"Internal server error"`（绝不泄露 traceback / 数据库错误细节 / 内部路径）
- 完整 traceback **只写进 `logs.db` 的 `app_logs` 表**（复用现有 schema：`level="ERROR"`, `module="api.exception_handler"`, `message=f"{request_id} {path} | {type(exc).__name__}: {exc}"`, `location=traceback.format_exc()[-2000:]`）
- 支持线上排查：用户拿到 `request_id` 找客服 → 后端按 request_id 查 app_logs → 看 traceback

### 前端

| 路径 | 说明 |
|---|---|
| `frontend/src/api/errors.ts`（新） | `NousApiError` class |
| `frontend/src/api/client.ts`（**必须同批改**） | `apiFetch` 抛出 `NousApiError`；现有 `body.detail` 逻辑失效 |
| `frontend/src/stores/toast.ts`（可选，后续） | 按 `error.type` 做差异化 UI |

**⚠️ Breaking change：** 后端改完后，现有 `apiFetch` 的 `body.detail || ...` 会落到 fallback（所有错误 toast 变成 "API error: <status>"）。前端 client.ts 必须**和后端同一次 commit** 一起改。

**`frontend/src/api/errors.ts`（新建）：**

```typescript
export class NousApiError extends Error {
  type: string;
  code?: string;
  param?: string;
  requestId?: string;
  httpStatus: number;

  constructor(payload: any, httpStatus: number, fallbackRequestId?: string) {
    const err = payload?.error ?? {};
    super(err.message ?? `HTTP ${httpStatus}`);
    this.name = 'NousApiError';
    this.type = err.type ?? 'api_error';
    this.code = err.code;
    this.param = err.param;
    this.requestId = err.request_id ?? fallbackRequestId;
    this.httpStatus = httpStatus;
  }
}
```

**`frontend/src/api/client.ts`（改写）：**

```typescript
import { NousApiError } from './errors'

const BASE = ''

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  })
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}))
    const reqId = resp.headers.get('x-request-id') ?? undefined
    throw new NousApiError(body, resp.status, reqId)
  }
  if (resp.status === 204) return undefined as T
  return resp.json()
}
```

**现有消费者兼容性：** `error.message` 访问依然有效（`NousApiError extends Error`），所以所有 `${error.message}` 的 toast 仍能正常显示。按 `type` 分支是新能力，现有代码无需改。

### 流式响应错误（SSE）

`/v1/chat/completions` + `stream=true` 的错误不能走 JSON handler。OpenAI SSE 协议：

```
data: {"error": {"message": "...", "type": "api_error", "request_id": "..."}}\n\n
data: [DONE]\n\n
```

**实现：** `backend/src/api/routes/openai_compat.py` 的流式循环里 `try/except NousError` + `except Exception`，捕获后 yield 上面格式的 2 行，再正常关闭流。调用端（OpenAI SDK）会在 `chunk` 里看到 `chunk.get('error')` —— 符合 OpenAI 标准，客户端不用改。

## 日志集成（复用现有 `logs.db`）

不新建表，复用 `backend/src/services/log_db.py` 已有的 `app_logs` schema：

```sql
app_logs (id, timestamp, level, module, message, location)
```

500 handler 写法（复用已有的 `DbLogHandler` — 它捕获 stdlib logging，无需额外 API）：
```python
import logging, traceback
logger = logging.getLogger(__name__)

@app.exception_handler(Exception)
async def unexpected_exc(request, exc):
    req_id = getattr(request.state, "request_id", None)
    # logger.exception() 会自动带 traceback；DbLogHandler 已注册，自动落盘 app_logs
    logger.exception(
        "unhandled exception | req_id=%s | %s %s",
        req_id, request.method, request.url.path,
    )
    err = APIError("Internal server error", code="internal_error", request_id=req_id)
    return JSONResponse(err.to_dict(), status_code=500,
                        headers={"X-Request-Id": req_id or ""})
```

这样前端 `LogsOverlay` 查看 app_logs（level=ERROR）就能看到 `req_id` + 完整 traceback（存在 `message` 里，由 `logger.exception` 格式化后附加）。`location` 字段由 DbLogHandler 自动填 `filename:lineno`。

## 不做的事（YAGNI）

- 不预定义 `code` 枚举（每次新加错误不需要改 enum）
- 不做多语言 error message（后端只返英文/固定 message，前端需要本地化时按 `code` 自己映射）
- 不改现有路由代码的 `raise HTTPException`（handler 自动转换；将来新代码用 `NousError` 子类）
- 不加错误码文档页面（等真的有外部用户时再生成）
- 不改 `logs.db` schema（request_id 只存在错误日志里即可）

## 验证

```bash
# 1. HTTPException 自动转换
curl --noproxy '*' -sw "\n%{http_code}\n" http://localhost:8000/api/v1/instances/999999999
# 预期: {"error":{"message":"Instance not found","type":"not_found_error","request_id":"<uuid>"}} + 404

# 1a. Pydantic validation error (RequestValidationError path)
curl --noproxy '*' -sw "\n%{http_code}\n" -X POST http://localhost:8000/api/v1/instances \
  -H "Content-Type: application/json" -d '{"name":"no_source_type"}'
# 预期: {"error":{"message":"Field required","type":"invalid_request_error","code":"validation_error","param":"source_type","request_id":"..."}} + 400

# 1b. 500 兜底：traceback 不泄露
# (触发 500 的方式：临时在某路由 raise RuntimeError 测试)
# 预期响应: {"error":{"message":"Internal server error","type":"api_error","code":"internal_error","request_id":"..."}}
# logs.db 的 app_logs 表里应有完整 traceback

# 1c. 流式 SSE 错误
curl --noproxy '*' -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-xxx" -H "Content-Type: application/json" \
  -d '{"model":"nonexistent","messages":[{"role":"user","content":"hi"}],"stream":true}'
# 预期: data: {"error":{"message":"...","type":"not_found_error","request_id":"..."}}
#       data: [DONE]

# 2. 无效 API Key
curl --noproxy '*' -sw "\n%{http_code}\n" -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-invalid-xxx" -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-35b","messages":[{"role":"user","content":"hi"}]}'
# 预期:
# {"error":{"message":"Invalid API key","type":"authentication_error","code":"invalid_api_key","request_id":"..."}}
# 401

# 3. Request ID 回传
curl --noproxy '*' -D- http://localhost:8000/api/v1/instances -o /dev/null 2>&1 | grep -i x-request-id
# 预期: x-request-id: <uuid>

# 4. 客户端传入 Request ID 被回传
curl --noproxy '*' -D- -H "X-Request-Id: my-trace-abc" http://localhost:8000/api/v1/instances -o /dev/null 2>&1 | grep -i x-request-id
# 预期: x-request-id: my-trace-abc

# 5. 前端验证
# 打开 DevTools → 手动 fetch 一个失败请求，console 里应看到 NousApiError with type/code/requestId
```

## 回滚

改动局限在 3 个新文件 + `main.py` 注册逻辑 + 前端 `client.ts` 调整。`git revert` 单次 commit 即可完全恢复。现有路由代码不被触碰。
