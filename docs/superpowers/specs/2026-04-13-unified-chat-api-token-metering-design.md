# Unified Chat API + Token Metering

## Goal

Make nous-center a private "Volcengine Ark" — expose a standard OpenAI-compatible chat API with per-request token metering. Personal/small team use case.

## Architecture

```
Client POST /v1/chat/completions (Bearer sk-xxx)
  → verify_instance_key() — validate Key, resolve ServiceInstance
  → ServiceInstance.source_type="model", source_id=engine_name
  → ModelManager finds loaded adapter, gets base_url (e.g. localhost:47683)
  → Proxy request to vLLM/SGLang (inject stream_options.include_usage for streaming)
  → Extract usage (prompt_tokens, completion_tokens) from response
  → Write LLMUsage record + update api_key counters
  → Return standard OpenAI format to client
```

## Components

### 1. LLMUsage Database Table

New SQLAlchemy model for per-request token tracking.

```python
class LLMUsage(Base):
    __tablename__ = "llm_usage"
    id: int              # Snowflake ID
    instance_id: int     # FK to ServiceInstance
    api_key_id: int      # FK to InstanceApiKey
    model: str           # engine name (e.g. "qwen3_5_35b_a3b_gptq_int4")
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_ms: int
    created_at: datetime
```

Indexes: `(model, created_at)`, `(instance_id, created_at)`, `(api_key_id, created_at)`

### 2. OpenAI-Compatible Chat Endpoint

Extend `openai_compat.py` with:

**POST `/v1/chat/completions`**
- Bearer token auth via existing `verify_instance_key()`
- Resolve ServiceInstance → engine name → loaded adapter → base_url
- Non-streaming: proxy POST to vLLM, extract `usage` from response JSON, record, return
- Streaming: proxy SSE stream, inject `stream_options: {include_usage: true}`, extract usage from final chunk, record, relay chunks to client
- Error handling: if model not loaded → 503; if adapter has no base_url → 500

Request validation:
- `model` field in request is optional (ServiceInstance already binds to a specific model)
- If provided, must match the bound engine (or ignore it)
- `max_tokens` clamped to adapter's `max_model_len - 512`

### 3. ServiceInstance Extension

Add `source_type="model"` support:
- `source_id` stores engine name string (not numeric ID)
- `ServiceInstanceCreate` schema: add `source_type="model"` option
- When `source_type="model"`: `source_id` is treated as engine name string, validated against ModelRegistry
- Endpoint path auto-generated as `/v1/chat/completions` (shared, differentiated by API key)

### 4. Model Page: "Create Endpoint" Action

In ModelsOverlay's context menu, add "Create API Endpoint" option for loaded models:
- Creates a ServiceInstance with `source_type="model"`, `source_id=engine_name`
- Auto-generates an API key
- Shows the key once (copy dialog)
- Displays connection info: `base_url + api_key + model_name`

### 5. Dashboard Data Endpoints

**GET `/api/v1/usage/summary`**
- Returns: `today_calls`, `today_tokens`, `total_calls`, `total_tokens`
- Aggregated from LLMUsage + TTSUsage tables
- Fills Dashboard's "Today Calls" and "Token Usage" cards

**GET `/api/v1/usage/by-model`**
- Returns per-model breakdown: `[{model, calls, prompt_tokens, completion_tokens}]`
- Optional `?since=` parameter for time range

### 6. Wire Existing TTSUsage

Currently `record_tts_usage()` exists but is never called. Wire it into:
- `/v1/audio/speech` endpoint
- TTS workflow node executor
- Instance service synthesize endpoint

## What's NOT Changed

- Existing API Key verification logic (bcrypt + prefix)
- Existing ServiceInstance CRUD endpoints
- Existing TTS `/v1/audio/speech` endpoint (only adds usage recording)
- ModelManager / VLLMAdapter internals
- Workflow executor (token recording added as wrapper, not internal change)

## Files to Create

| File | Purpose |
|------|---------|
| `backend/src/models/llm_usage.py` | LLMUsage SQLAlchemy model |
| `backend/src/services/usage_service.py` | Record + query usage (LLM + TTS) |

## Files to Modify

| File | Change |
|------|--------|
| `backend/src/api/routes/openai_compat.py` | Add `/v1/chat/completions` with token metering |
| `backend/src/api/routes/instances.py` | Support `source_type="model"` |
| `backend/src/models/schemas.py` | Update ServiceInstanceCreate for model source type |
| `backend/src/api/main.py` | Import llm_usage model for table creation |
| `backend/src/api/routes/monitor.py` | Add `/api/v1/usage/summary` and `/api/v1/usage/by-model` |
| `frontend/src/components/overlays/ModelsOverlay.tsx` | "Create API Endpoint" in context menu |
| `frontend/src/components/overlays/DashboardOverlay.tsx` | Wire Today Calls / Token Usage cards |

## Token Counting Strategy

- **Non-streaming**: Read `response.usage.prompt_tokens` and `response.usage.completion_tokens` from vLLM JSON response
- **Streaming**: Inject `stream_options: {"include_usage": true}` into request. vLLM returns usage in the final SSE chunk. Parse and record after stream completes
- **No local tokenizer needed** — vLLM provides accurate counts natively

## Error Cases

| Scenario | Response |
|----------|----------|
| Invalid API key | 401 Unauthorized |
| Model not loaded | 503 Service Unavailable ("Model not loaded. Load it from the management page.") |
| Model not found in registry | 404 Not Found |
| vLLM returns error | Proxy the error status + message |
| max_tokens exceeds model limit | Auto-clamp, proceed |
