# Logs System Design

> Date: 2026-04-09
> Status: Approved
> Scope: 4-tab logging system (Request, Application, Frontend, Audit)

---

## Architecture

Separate `logs.db` SQLite database (WAL mode) for all log storage, isolated from business data. Collection via middleware (Request + Audit), Python logging handler (Application), and dedicated API endpoint (Frontend).

Retention: 7 days OR 10,000 records per table, whichever triggers first. Background cleanup runs hourly.

## Data Models

All tables live in `data/logs.db` (separate from `data/nous.db`).

### request_logs

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK AUTOINCREMENT | |
| timestamp | DATETIME | Request time (UTC) |
| method | VARCHAR(10) | GET/POST/PUT/DELETE/PATCH/OPTIONS |
| path | VARCHAR(500) | Request path |
| status | INTEGER | HTTP status code |
| duration_ms | INTEGER | Response time in ms |
| ip | VARCHAR(45) | Client IP |
| user_agent | VARCHAR(500) | User-Agent header |

### app_logs

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK AUTOINCREMENT | |
| timestamp | DATETIME | Log time (UTC) |
| level | VARCHAR(10) | DEBUG/INFO/WARNING/ERROR/CRITICAL |
| module | VARCHAR(100) | Logger name (e.g. `src.services.model_manager`) |
| message | TEXT | Log message |
| location | VARCHAR(200) | `filename:lineno` |

### frontend_logs

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK AUTOINCREMENT | |
| timestamp | DATETIME | Error time (UTC) |
| type | VARCHAR(30) | `network` / `unhandled_rejection` / `error` / `console_error` |
| message | TEXT | Error message |
| page | VARCHAR(500) | Page path where error occurred |
| stack | TEXT | Stack trace (optional, nullable) |

### audit_logs

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK AUTOINCREMENT | |
| timestamp | DATETIME | Action time (UTC) |
| action | VARCHAR(50) | Derived from method + path (e.g. `load_model`, `publish_app`) |
| path | VARCHAR(500) | API path |
| method | VARCHAR(10) | HTTP method |
| ip | VARCHAR(45) | Client IP |
| detail | TEXT | Request body as JSON string |

## Collection Methods

### Request Logs

Modify existing `RequestLoggingMiddleware` to write each request to `logs.db` via `log_db.insert_request_log()`. Skip `/health`, `/favicon.ico`, and `/api/v1/logs/` paths to avoid recursive logging.

### Application Logs

Create `DbLogHandler(logging.Handler)` that intercepts all log records from `src.*` loggers. Buffers writes and flushes every 1 second or 50 records (whichever comes first) to avoid per-log DB overhead.

### Frontend Logs

`POST /api/v1/logs/frontend` accepts:
```json
{
  "type": "network",
  "message": "GET /api/v1/search — Request failed",
  "page": "/models",
  "stack": null
}
```

Frontend installs global error handlers on app init:
- `window.addEventListener('error', ...)` — JS errors
- `window.addEventListener('unhandledrejection', ...)` — Promise rejections
- `fetch` wrapper that catches network errors

### Audit Logs

New `AuditMiddleware` that runs after authentication. Captures requests where the admin token was provided (matches `require_admin` pattern). Records method, path, IP, and request body.

Action name derived from path: `/api/v1/engines/{name}/load` → `load_engine`, `/api/v1/workflows/{id}/publish-app` → `publish_app`.

## Retention & Cleanup

Background task runs every hour:

```python
async def cleanup_logs():
    cutoff = datetime.utcnow() - timedelta(days=7)
    for table in ['request_logs', 'app_logs', 'frontend_logs', 'audit_logs']:
        # Delete old records
        DELETE FROM {table} WHERE timestamp < cutoff
        # Enforce row limit
        DELETE FROM {table} WHERE id NOT IN (
            SELECT id FROM {table} ORDER BY id DESC LIMIT 100000
        )
```

## API Endpoints

All GET endpoints require no auth (dashboard data). POST frontend endpoint requires no auth (browser reports).

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/logs/requests` | Query request logs |
| GET | `/api/v1/logs/app` | Query application logs |
| GET | `/api/v1/logs/frontend` | Query frontend logs |
| POST | `/api/v1/logs/frontend` | Report frontend error |
| GET | `/api/v1/logs/audit` | Query audit logs |

### Common Query Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| limit | int | 50 | Max rows (max 500) |
| offset | int | 0 | Pagination offset |
| search | str | null | Full-text search on message/path |
| level | str | null | Filter by level (app_logs only) |
| type | str | null | Filter by type (frontend_logs only) |
| method | str | null | Filter by HTTP method |
| status | str | null | Filter by status code range (e.g. "4xx", "5xx") |
| since | str | null | ISO datetime, only return records after this time |

### Response Format

```json
{
  "total": 1234,
  "items": [
    { "id": 1, "timestamp": "2026-04-09T10:00:00Z", ... }
  ]
}
```

## Frontend UI

### Navigation

Add log icon to IconRail (bottom section, near settings). Opens `LogsOverlay` as a full-page overlay.

### LogsOverlay Layout

```
[Request Logs] [App Logs] [Frontend Logs] [Audit Logs]

[Search...] [Filter ▾] [15m] [1h] [24h] [3d] [7d]  [● Live]

┌─────────┬────────┬─────────────────┬────────┬────────┬──────────┐
│ Time    │ Method │ Path            │ Status │ Time   │ IP       │
├─────────┼────────┼─────────────────┼────────┼────────┼──────────┤
│ 20:51:37│ GET    │ /api/v1/tasks   │ 200    │ 102ms  │ 127.0.0.1│
│ ...     │ ...    │ ...             │ ...    │ ...    │ ...      │
```

- Method column: colored badges (GET=green, POST=blue, DELETE=red)
- Status column: green for 2xx, yellow for 4xx, red for 5xx
- Level column (app_logs): INFO=cyan, WARNING=yellow, ERROR=red
- Auto-refresh every 3 seconds when "Live" toggle is on
- Time range buttons filter by `since` parameter
- Click row to expand and show full details

### Files

**Backend:**

| File | Responsibility |
|------|----------------|
| `backend/src/services/log_db.py` | logs.db connection, table creation, insert/query/cleanup functions |
| `backend/src/services/log_collector.py` | `DbLogHandler` for Python logging, buffered writes |
| `backend/src/api/middleware.py` | Modify `RequestLoggingMiddleware` + add `AuditMiddleware` |
| `backend/src/api/routes/logs.py` | Log query + frontend report endpoints |

**Frontend:**

| File | Responsibility |
|------|----------------|
| `frontend/src/api/logs.ts` | React Query hooks for all log endpoints |
| `frontend/src/components/overlays/LogsOverlay.tsx` | 4-tab log viewer with search, filter, time range |
| `frontend/src/utils/errorReporter.ts` | Global error handlers + POST to `/api/v1/logs/frontend` |
