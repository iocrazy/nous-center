#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Starting nous-core (Rust) ==="
(cd "$REPO_ROOT/nous-core" && cargo run) &
CORE_PID=$!

echo "=== Starting backend (FastAPI) ==="
(cd "$REPO_ROOT/backend" && uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000) &
BACKEND_PID=$!

echo "=== Starting frontend (Vite) ==="
(cd "$REPO_ROOT/frontend" && npm run dev) &
FRONTEND_PID=$!

echo ""
echo "Services running:"
echo "  backend:   http://localhost:8000"
echo "  nous-core: http://localhost:8001"
echo "  frontend:  http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop all."

trap "kill $CORE_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
