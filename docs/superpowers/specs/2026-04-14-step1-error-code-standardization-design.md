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
│                                           │
│  X-Request-Id: <UUID>                     │
│                                           │
└───────────────────┬───────────────────────┘
                    ▼
        ┌──── RequestIdMiddleware ────┐
        │ request.state.request_id    │
        └──────────────┬──────────────┘
                       ▼
              ┌─── Route handler ───┐
              │  raises:            │
              │  - NousError (new)  │
              │  - HTTPException    │
              │  - Exception 兜底   │
              └──────────┬──────────┘
                         ▼
        ┌─── ExceptionHandler (全局) ───┐
        │  转 OpenAI payload            │
        │  注入 request_id              │
        │  设置 HTTP status             │
        │  写 logs.db (含 traceback)    │
        └──────────────┬────────────────┘
                       ▼
┌───────────────── Response ────────────────┐
│ X-Request-Id: <UUID>                      │
│ Content-Type: application/json            │
│ { "error": { ...OpenAI shape... } }       │
└───────────────────────────────────────────┘
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

| HTTP status | 转成的 NousError |
|---|---|
| 400 | `InvalidRequestError` |
| 401 | `AuthenticationError` |
| 403 | `PermissionError` |
| 404 | `NotFoundError` |
| 429 | `RateLimitError` |
| 其他 4xx | `InvalidRequestError` |
| 5xx | `APIError` |

### 前端

| 路径 | 说明 |
|---|---|
| `frontend/src/api/client.ts`（修改） | `apiFetch` 解析 OpenAI 错误 payload，抛出 `NousApiError` |
| `frontend/src/api/errors.ts`（新） | `NousApiError` class：`message` / `type` / `code` / `param` / `requestId` |
| `frontend/src/stores/toast.ts`（修改，可选） | 按 `error.type` 做差异化 UI（rate_limit 倒计时、auth 跳登录） |

**`NousApiError`：**

```typescript
export class NousApiError extends Error {
  type: string;
  code?: string;
  param?: string;
  requestId?: string;
  httpStatus: number;

  constructor(payload: any, httpStatus: number) {
    const err = payload?.error ?? {};
    super(err.message ?? `HTTP ${httpStatus}`);
    this.type = err.type ?? 'api_error';
    this.code = err.code;
    this.param = err.param;
    this.requestId = err.request_id;
    this.httpStatus = httpStatus;
  }
}
```

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
# 预期输出:
# {"error":{"message":"Instance not found","type":"not_found_error","request_id":"<uuid>"}}
# 404

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
