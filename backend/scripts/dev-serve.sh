#!/usr/bin/env bash
# Dev 启动脚本:source .env(uv 不自动 load)+ uvicorn,stdout/stderr 落
# backend/logs/backend-dev.log(带简单 size 轮转)。生产用 systemd → journald
# (journalctl -u nous-backend -f),**不要**用这个脚本起生产。
#
#   用法: backend/scripts/dev-serve.sh            # 前台
#         backend/scripts/dev-serve.sh &          # 后台
#   看日志: tail -f backend/logs/backend-dev.log
#
# 注:结构化请求/审计/前端日志仍进 log_db(中间件写,与 stdout 去向无关),
# 可经 /api/v1/logs/* 或前端 LogsOverlay 查看。本脚本只规整进程 stdout 去向。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # backend/
cd "$HERE"

if [[ ! -f .env ]]; then
  echo "[dev-serve] backend/.env 不存在 —— 先 gen-admin-secrets + 配 .env" >&2
  exit 1
fi

LOG_DIR="$HERE/logs"
LOG="$LOG_DIR/backend-dev.log"
mkdir -p "$LOG_DIR"

# size 轮转:>50MB 就转存 .1(只留一份历史),避免无限增长。
MAX_BYTES=$((50 * 1024 * 1024))
if [[ -f "$LOG" ]]; then
  size=$(wc -c < "$LOG" 2>/dev/null || echo 0)
  if (( size > MAX_BYTES )); then
    mv -f "$LOG" "$LOG.1"
  fi
fi

HOST="${NOUS_DEV_HOST:-127.0.0.1}"
PORT="${NOUS_DEV_PORT:-8000}"

# uv 不 load .env(dev_env_gotchas)→ 显式 source 并 export。
set -a
# shellcheck disable=SC1091
. ./.env
set +a

echo "[dev-serve] uvicorn $HOST:$PORT → $LOG ($(date '+%F %T'))" | tee -a "$LOG"
exec uv run uvicorn src.api.main:app --host "$HOST" --port "$PORT" 2>&1 | tee -a "$LOG"
