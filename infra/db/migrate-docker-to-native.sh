#!/usr/bin/env bash
# ============================================================================
# nous_center DB 迁移:Docker Desktop postgres → 原生 pg17
# spec: docs/superpowers/specs/2026-06-18-native-pg-systemd-stack-design.md (PR-1)
#
# 安全优先:dump → restore → 逐表 count 校验 → 不一致立即中止(源库原封不动)。
# 迁移期双 pg 并存:docker@5432(源) + 原生@5433(目标),跨端口迁。校验通过后才由
# README 的运维步骤把原生切回 5432。本脚本**只搬数据 + 校验**,不动端口/不停容器。
#
# 幂等:可反复跑 —— 每次重新 dump → 把目标库 schema drop&recreate(--clean)再灌,
# 再校验。源库只读,永不被改。
#
# 用法:
#   NOUS_DB_PASSWORD=*** ./migrate-docker-to-native.sh precheck      # 只查前置
#   NOUS_DB_PASSWORD=*** ./migrate-docker-to-native.sh init-target   # 建 role+库(sudo -u postgres)
#   NOUS_DB_PASSWORD=*** ./migrate-docker-to-native.sh migrate       # dump+restore
#   NOUS_DB_PASSWORD=*** ./migrate-docker-to-native.sh verify        # 逐表 count 比对
#   NOUS_DB_PASSWORD=*** ./migrate-docker-to-native.sh all           # 上面四步串起来
#
# 关键 env(均有默认):
#   SRC_HOST/SRC_PORT   源(docker)  默认 127.0.0.1:5432
#   DST_HOST/DST_PORT   目标(原生)  默认 127.0.0.1:5433
#   DB / ROLE           默认 nous_center / nous_heygo
#   NOUS_DB_PASSWORD    role 密码(建库 + dump/restore 鉴权)。未给则从 backend/.env 抠。
# ============================================================================
set -euo pipefail

SRC_HOST="${SRC_HOST:-127.0.0.1}"; SRC_PORT="${SRC_PORT:-5432}"
DST_HOST="${DST_HOST:-127.0.0.1}"; DST_PORT="${DST_PORT:-5433}"
DB="${DB:-nous_center}"; ROLE="${ROLE:-nous_heygo}"
DUMP="${DUMP:-/tmp/${DB}-migrate.dump}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

c_red()  { printf '\033[31m%s\033[0m\n' "$*"; }
c_grn()  { printf '\033[32m%s\033[0m\n' "$*"; }
c_ylw()  { printf '\033[33m%s\033[0m\n' "$*"; }
die()    { c_red "ERROR: $*" >&2; exit 1; }

# 密码:env 优先,否则从 backend/.env 的 DATABASE_URL 抠(postgresql+asyncpg://user:PASS@...)
resolve_password() {
  if [[ -n "${NOUS_DB_PASSWORD:-}" ]]; then return; fi
  local envf="$REPO_ROOT/backend/.env"
  if [[ -f "$envf" ]]; then
    local url; url="$(grep -E '^DATABASE_URL=' "$envf" | head -1 | cut -d= -f2-)"
    # postgresql+asyncpg://user:PASS@host:port/db  →  抠 PASS(user 与 PASS 间 : 后、@ 前)
    NOUS_DB_PASSWORD="$(printf '%s' "$url" | sed -E 's#^[a-z+]+://[^:]+:([^@]+)@.*$#\1#')"
  fi
  [[ -n "${NOUS_DB_PASSWORD:-}" ]] || die "拿不到密码:给 NOUS_DB_PASSWORD 或在 backend/.env 配 DATABASE_URL"
}

need_tools() {
  for t in pg_dump pg_restore psql pg_isready; do
    command -v "$t" >/dev/null 2>&1 || die "缺工具 $t(装 postgresql-client-17)"
  done
}

cmd_precheck() {
  need_tools
  c_grn "[precheck] 工具齐:pg_dump/pg_restore/psql/pg_isready"
  pg_isready -h "$SRC_HOST" -p "$SRC_PORT" >/dev/null 2>&1 \
    && c_grn "[precheck] 源 docker pg 可达 $SRC_HOST:$SRC_PORT" \
    || die "源 pg 不可达 $SRC_HOST:$SRC_PORT(docker 容器没起?)"
  pg_isready -h "$DST_HOST" -p "$DST_PORT" >/dev/null 2>&1 \
    && c_grn "[precheck] 目标原生 pg 可达 $DST_HOST:$DST_PORT" \
    || die "目标 pg 不可达 $DST_HOST:$DST_PORT(原生 pg 没装/没起?见 infra/db/README.md)"
}

# 建 role + 库 —— 需 superuser。postgres 超级用户没设密码,只能走 **Unix socket 的 peer
# 认证**(故不能加 -h,加了就走 TCP scram → 认证失败)。用 -p 选中目标 cluster 的 socket。幂等。
cmd_init_target() {
  resolve_password
  c_ylw "[init-target] 在原生 pg(socket, port=$DST_PORT)建 role=$ROLE + 库=$DB(幂等,需 sudo)"
  sudo -u postgres psql -p "$DST_PORT" -v ON_ERROR_STOP=1 <<-SQL
    DO \$\$ BEGIN
      IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${ROLE}') THEN
        CREATE ROLE ${ROLE} LOGIN PASSWORD '${NOUS_DB_PASSWORD}';
      ELSE
        ALTER ROLE ${ROLE} LOGIN PASSWORD '${NOUS_DB_PASSWORD}';
      END IF;
    END \$\$;
    SELECT 'CREATE DATABASE ${DB} OWNER ${ROLE}'
      WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${DB}')\gexec
SQL
  c_grn "[init-target] role + 库 就绪"
}

cmd_migrate() {
  resolve_password; need_tools
  c_ylw "[migrate] dump 源库 → $DUMP"
  PGPASSWORD="$NOUS_DB_PASSWORD" pg_dump -Fc \
    -h "$SRC_HOST" -p "$SRC_PORT" -U "$ROLE" -d "$DB" -f "$DUMP"
  local sz; sz="$(du -h "$DUMP" | cut -f1)"
  c_grn "[migrate] dump 完成($sz)"
  c_ylw "[migrate] restore → 目标库(--clean --if-exists,幂等覆盖)"
  PGPASSWORD="$NOUS_DB_PASSWORD" pg_restore --clean --if-exists --no-owner --role="$ROLE" \
    -h "$DST_HOST" -p "$DST_PORT" -U "$ROLE" -d "$DB" "$DUMP"
  c_grn "[migrate] restore 完成"
}

# 逐表 count 比对源/目标。任一不一致 → 退出码 1(中止切换)。
cmd_verify() {
  resolve_password; need_tools
  c_ylw "[verify] 枚举 public schema 基表,逐表 count 比对"
  local tables
  tables="$(PGPASSWORD="$NOUS_DB_PASSWORD" psql -tAq \
    -h "$SRC_HOST" -p "$SRC_PORT" -U "$ROLE" -d "$DB" \
    -c "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename")"
  [[ -n "$tables" ]] || die "源库无任何表?中止"
  local mismatch=0 total=0
  printf '  %-34s %12s %12s  %s\n' "table" "src" "dst" "ok?"
  printf '  %-34s %12s %12s  %s\n' "----------------------------------" "------------" "------------" "---"
  while IFS= read -r t; do
    [[ -z "$t" ]] && continue
    local s d
    s="$(PGPASSWORD="$NOUS_DB_PASSWORD" psql -tAq -h "$SRC_HOST" -p "$SRC_PORT" -U "$ROLE" -d "$DB" -c "SELECT count(*) FROM \"$t\"")"
    d="$(PGPASSWORD="$NOUS_DB_PASSWORD" psql -tAq -h "$DST_HOST" -p "$DST_PORT" -U "$ROLE" -d "$DB" -c "SELECT count(*) FROM \"$t\"" 2>/dev/null || echo "MISSING")"
    total=$((total+1))
    if [[ "$s" == "$d" ]]; then
      printf '  %-34s %12s %12s  %s\n' "$t" "$s" "$d" "OK"
    else
      printf '  %-34s %12s %12s  %s\n' "$t" "$s" "$d" "MISMATCH"
      mismatch=$((mismatch+1))
    fi
  done <<< "$tables"
  echo
  if [[ "$mismatch" -ne 0 ]]; then
    c_red "[verify] $mismatch/$total 表不一致 —— 中止。源 docker pg 未被改动,可重跑 migrate 或排查。"
    exit 1
  fi
  c_grn "[verify] 全部 $total 表 count 一致。可按 README 切端口到 5432。"
}

case "${1:-all}" in
  precheck)    cmd_precheck ;;
  init-target) cmd_init_target ;;
  migrate)     cmd_precheck; cmd_migrate ;;
  verify)      cmd_verify ;;
  all)         cmd_precheck; cmd_init_target; cmd_migrate; cmd_verify ;;
  *) die "未知子命令 '$1'(precheck|init-target|migrate|verify|all)" ;;
esac
