#!/usr/bin/env bash
# ============================================================================
# nous_center DB 每日自动备份(spec native-pg-systemd-stack PR-4)
#
# pg_dump 自定义格式 → 备份目录(默认 NAS,异机,本地不留副本),滚动保留 N 天。
# **落盘后校验** dump 可 `pg_restore --list` 读 → 才原子改名(半截/坏 dump 不冒充好备份)。
# 由 nous-dbbackup.timer 每日触发;也可手动跑做即时备份。
#
# env(均有默认):
#   NOUS_DB_BACKUP_DIR        备份目录          默认 /mnt/heytime/backup/nous-db-dumps(NAS)
#   NOUS_DB_BACKUP_KEEP_DAYS  保留天数          默认 14
#   PGHOST/PGPORT/DB/ROLE     连接              默认 127.0.0.1/5432/nous_center/nous_heygo
#   NOUS_DB_PASSWORD          密码;未给则从 backend/.env 的 DATABASE_URL 抠
# ============================================================================
set -euo pipefail

BACKUP_DIR="${NOUS_DB_BACKUP_DIR:-/mnt/heytime/backup/nous-db-dumps}"
KEEP_DAYS="${NOUS_DB_BACKUP_KEEP_DAYS:-14}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

log() { printf '%s nous-dbbackup: %s\n' "$(date '+%F %T')" "$*"; }
die() { printf '%s nous-dbbackup ERROR: %s\n' "$(date '+%F %T')" "$*" >&2; exit 1; }

# 连接参数:统一从 backend/.env 的 DATABASE_URL 解析(user/pass/host/port/db)。
# 早先只抠 password、user/db/host/port 用硬编码默认(nous_heygo/nous_center)—— .env 里
# 角色名不同(如 nous)就 `password authentication failed for user "nous_heygo"`。既然信
# DATABASE_URL 取密码,就该信它取全部连接参数。各字段仍可用同名 env 覆盖。
envf="$REPO_ROOT/backend/.env"
if [ -f "$envf" ]; then
  url="$(grep -E '^DATABASE_URL=' "$envf" | head -1 | cut -d= -f2-)"
  u="${url#*://}"                          # user:pass@host:port/db
  cred="${u%%@*}"; hpd="${u#*@}"
  URL_ROLE="${cred%%:*}"; URL_PASS="${cred#*:}"
  hp="${hpd%%/*}"; URL_DB="${hpd##*/}"
  URL_HOST="${hp%%:*}"; URL_PORT="${hp#*:}"
fi
ROLE="${ROLE:-${URL_ROLE:-nous_heygo}}"
DB="${DB:-${URL_DB:-nous_center}}"
PGHOST="${PGHOST:-${URL_HOST:-127.0.0.1}}"
PGPORT="${PGPORT:-${URL_PORT:-5432}}"
NOUS_DB_PASSWORD="${NOUS_DB_PASSWORD:-${URL_PASS:-}}"
[ -n "$NOUS_DB_PASSWORD" ] || die "取不到 DB 密码(设 NOUS_DB_PASSWORD,或在 backend/.env 配 DATABASE_URL)"

command -v pg_dump >/dev/null 2>&1 || die "缺 pg_dump(装 postgresql-client-17)"
mkdir -p "$BACKUP_DIR" || die "建不了备份目录 $BACKUP_DIR(盘没挂?)"

ts="$(date '+%Y%m%d-%H%M')"
out="$BACKUP_DIR/${DB}-${ts}.dump"
tmp="$out.partial"

log "dump $DB@$PGHOST:$PGPORT → $out"
PGPASSWORD="$NOUS_DB_PASSWORD" pg_dump -Fc -h "$PGHOST" -p "$PGPORT" -U "$ROLE" -d "$DB" -f "$tmp" \
  || { rm -f "$tmp"; die "pg_dump 失败"; }

# 校验:坏 dump 绝不冒充好备份。读得了归档目录才原子改名。
if ! PGPASSWORD="$NOUS_DB_PASSWORD" pg_restore --list "$tmp" >/dev/null 2>&1; then
  rm -f "$tmp"
  die "dump 校验失败(pg_restore --list 读不了),已删 — 本次备份作废"
fi
mv "$tmp" "$out"
log "OK ($(du -h "$out" | cut -f1))"

# 滚动:删 KEEP_DAYS 天前的旧 dump。
while IFS= read -r f; do
  [ -n "$f" ] && log "rotate 删旧备份 $f"
done < <(find "$BACKUP_DIR" -maxdepth 1 -name "${DB}-*.dump" -type f -mtime +"$KEEP_DAYS" -print -delete)

n="$(find "$BACKUP_DIR" -maxdepth 1 -name "${DB}-*.dump" -type f | wc -l)"
log "保留期 ${KEEP_DAYS}d,当前 $n 份备份于 $BACKUP_DIR"
