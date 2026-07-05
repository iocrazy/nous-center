#!/usr/bin/env bash
# 把 nous-center 全栈装成 systemd 服务(开机自启 + 崩溃自重启 + journald),取代 nohup。
# 幂等 —— 可反复跑。
#
# Usage:
#   sudo ./infra/systemd/install.sh            # install + enable + start + 自检
#   sudo ./infra/systemd/install.sh uninstall  # stop + disable + remove
#
# 装完自检 + 访问地址会打印在下面 banner 里。日志:
#   journalctl -u nous-backend -f    ·    nousctl status    ·    nousctl logs [backend|status|...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET=/etc/systemd/system
NOUSCTL_DST=/usr/local/bin/nousctl

# 长驻服务(cloudflared 单独处理 —— 缺二进制/凭证时优雅跳过,不中止安装)。
SERVICES=(nous-backend.service nous-status.service nous-aligner.service)
TIMERS=(nous-healthprobe.timer nous-dbbackup.timer)
TARGETS=(nous.target)
SUDOERS=(nous-healthprobe nous-deploy)
# 全部拷进 /etc/systemd/system(含 cloudflared、oneshot probe/dbbackup、target)。
UNIT_FILES=(nous-backend.service nous-cloudflared.service nous-status.service nous-aligner.service \
            nous-healthprobe.service nous-healthprobe.timer nous-dbbackup.service nous-dbbackup.timer nous.target)

LOCAL_URL="${NOUS_LOCAL_URL:-http://127.0.0.1:8000}"
ZT_URL="${NOUS_ZT_URL:-http://10.0.0.10:8000}"
PUBLIC_URL="${NOUS_PUBLIC_URL:-https://api.iocrazy.com}"
STATUS_URL="${NOUS_STATUS_URL:-http://127.0.0.1:8001}"

# ── 样式 ──────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  B=$'\033[1m'; DIM=$'\033[2m'; GRN=$'\033[1;32m'; RED=$'\033[1;31m'; YEL=$'\033[1;33m'; CYN=$'\033[1;36m'; RST=$'\033[0m'
else B=""; DIM=""; GRN=""; RED=""; YEL=""; CYN=""; RST=""; fi
say()  { printf '%s\n' "$*"; }
step() { printf '\n%s▸ %s%s\n' "$CYN" "$*" "$RST"; }
ok()   { printf '  %s✔%s %s\n' "$GRN" "$RST" "$*"; }
warn() { printf '  %s!%s %s\n' "$YEL" "$RST" "$*"; }
bad()  { printf '  %s✗%s %s\n' "$RED" "$RST" "$*"; }
rule() { printf '%s────────────────────────────────────────────────────────────%s\n' "$DIM" "$RST"; }

if [[ "${EUID}" -ne 0 ]]; then echo "ERROR: 需 root(sudo)" >&2; exit 1; fi

# 探一个 HTTP 端点,回显 code(不因 set -e 中止)。
probe() { curl -s --noproxy '*' -m "${2:-5}" -o /dev/null -w '%{http_code}' "$1" 2>/dev/null || echo 000; }
svc_active() { systemctl is-active --quiet "$1" 2>/dev/null; }

case "${1:-install}" in
  install)
    printf '\n%s╔══════════════════════════════════════════════════════════╗%s\n' "$B" "$RST"
    printf   '%s║   nous-center · systemd 全栈安装                          ║%s\n' "$B" "$RST"
    printf   '%s╚══════════════════════════════════════════════════════════╝%s\n' "$B" "$RST"
    say "${DIM}检出: $(cd "$SCRIPT_DIR/../.." && pwd)${RST}"

    # ── 1. 单元文件 ──────────────────────────────────────────────────────
    step "安装 systemd 单元 → $TARGET"
    for u in "${UNIT_FILES[@]}"; do install -m 0644 "$SCRIPT_DIR/$u" "$TARGET/$u"; done
    systemctl daemon-reload
    ok "${#UNIT_FILES[@]} 个单元已安装 + daemon-reload"

    # ── 2. 启用 + 启动长驻服务 ───────────────────────────────────────────
    step "启用 + 启动服务(开机自启)"
    for svc in "${SERVICES[@]}"; do
      if systemctl enable --now "$svc" >/dev/null 2>&1; then ok "$svc"
      else bad "$svc 启动失败 — 查 journalctl -u $svc -n 50"; fi
    done

    # cloudflared:二进制 + 凭证齐才启;否则装单元但不启(公网隧道暂缓,不中止安装)。
    CF_HOME="$(getent passwd "${SUDO_USER:-root}" | cut -d: -f6)"
    if command -v cloudflared >/dev/null 2>&1 && [[ -f "$CF_HOME/.cloudflared/cert.pem" ]]; then
      if systemctl enable --now nous-cloudflared.service >/dev/null 2>&1; then ok "nous-cloudflared(公网隧道)"
      else bad "nous-cloudflared 启动失败 — 查 journalctl -u nous-cloudflared"; fi
    else
      systemctl disable nous-cloudflared.service >/dev/null 2>&1 || true
      warn "cloudflared 二进制/凭证未就位 → 跳过公网隧道(本机 + ZeroTier 不受影响)"
      warn "  以后配好后: sudo systemctl enable --now nous-cloudflared"
    fi

    # ── 3. 定时器 + 总闸 ────────────────────────────────────────────────
    step "启用定时器 + 总闸"
    for tmr in "${TIMERS[@]}"; do systemctl enable --now "$tmr" >/dev/null 2>&1 && ok "$tmr"; done
    for tgt in "${TARGETS[@]}"; do systemctl enable "$tgt" >/dev/null 2>&1 && ok "$tgt (开机总闸)"; done

    # ── 4. nousctl + sudoers ────────────────────────────────────────────
    step "安装 nousctl + sudoers drop-ins"
    install -m 0755 "$SCRIPT_DIR/nousctl" "$NOUSCTL_DST"; ok "nousctl → $NOUSCTL_DST"
    for sd in "${SUDOERS[@]}"; do
      src="$SCRIPT_DIR/../security/$sd.sudoers"; dst="/etc/sudoers.d/$sd"
      install -m 0440 "$src" "$dst"
      if visudo -cf "$dst" >/dev/null 2>&1; then ok "sudoers: $sd"; else bad "sudoers $sd 校验失败,已撤掉"; rm -f "$dst"; fi
    done

    # ── 5. 自检(等后端就绪最多 ~30s)────────────────────────────────────
    step "自检"
    code=000
    for _ in $(seq 1 15); do code="$(probe "$LOCAL_URL/healthz")"; [[ "$code" == 200 ]] && break; sleep 2; done

    for svc in postgresql "${SERVICES[@]}"; do
      if svc_active "$svc"; then ok "$(printf '%-24s active' "$svc")"; else bad "$(printf '%-24s %s' "$svc" "$(systemctl is-active "$svc" 2>/dev/null || echo inactive)")"; fi
    done

    if [[ "$code" == 200 ]]; then
      ok "本机 /healthz  → 200"
      # 组件详情(database / gpus / 常驻模型)
      health="$(curl -s --noproxy '*' -m 5 "$LOCAL_URL/health" 2>/dev/null || echo '{}')"
      db=$(printf '%s' "$health"  | grep -o '"database":"[^"]*"' | cut -d'"' -f4)
      gpus=$(printf '%s' "$health"| grep -o '"gpus":[0-9]*'      | cut -d: -f2)
      [[ -n "$db"   ]] && { [[ "$db" == ok ]] && ok "database    → ok" || bad "database    → $db"; }
      [[ -n "$gpus" ]] && ok "GPU 识别    → $gpus 张"
    else
      bad "本机 /healthz  → $code(后端可能还在预加载常驻模型,稍等再 nousctl status)"
    fi

    zt="$(probe "$ZT_URL/healthz")";     [[ "$zt" == 200 ]] && ok "ZeroTier /healthz → 200" || warn "ZeroTier /healthz → $zt(10.0.0.10 未分配?)"
    if svc_active nous-cloudflared; then
      pub="$(probe "$PUBLIC_URL/healthz" 10)"; [[ "$pub" == 200 ]] && ok "公网隧道 /healthz → 200" || warn "公网隧道 /healthz → $pub(隧道重连中?)"
    fi

    # ── 访问地址 + 收尾 ─────────────────────────────────────────────────
    printf '\n%s╭─ 访问地址 ────────────────────────────────────────────────%s\n' "$B" "$RST"
    printf   '%s│%s  本机管理台   %s%s%s\n'   "$B" "$RST" "$CYN" "$LOCAL_URL"  "$RST"
    printf   '%s│%s  ZeroTier 内网 %s%s%s\n'  "$B" "$RST" "$CYN" "$ZT_URL"     "$RST"
    printf   '%s│%s  公网(隧道)   %s%s%s %s\n' "$B" "$RST" "$CYN" "$PUBLIC_URL" "$RST" "$(svc_active nous-cloudflared && echo '' || echo "${DIM}(cloudflared 暂未启用)${RST}")"
    printf   '%s│%s  独立状态页   %s%s%s\n'   "$B" "$RST" "$CYN" "$STATUS_URL" "$RST"
    printf   '%s╰──────────────────────────────────────────────────────────%s\n' "$B" "$RST"

    printf '\n%s✔ 安装完成%s — 服务已开机自启 + 崩溃自重启。\n' "$GRN" "$RST"
    say "  管控:   ${B}nousctl${RST} status | up | down | restart | logs"
    say "  日志:   ${B}journalctl -u nous-backend -f${RST}"
    say "  健康巡检: 每 2 分钟(journalctl -u nous-healthprobe -f)"
    ;;

  uninstall)
    printf '\n%s▸ 卸载 nous-center systemd 栈%s\n' "$CYN" "$RST"
    for tgt in "${TARGETS[@]}"; do systemctl disable "$tgt" 2>/dev/null || true; rm -f "$TARGET/$tgt"; ok "移除 $tgt"; done
    for tmr in "${TIMERS[@]}"; do systemctl disable --now "$tmr" 2>/dev/null || true; rm -f "$TARGET/$tmr"; ok "移除 $tmr"; done
    for svc in "${SERVICES[@]}" nous-cloudflared.service; do systemctl disable --now "$svc" 2>/dev/null || true; rm -f "$TARGET/$svc"; ok "移除 $svc"; done
    rm -f "$TARGET/nous-healthprobe.service" "$TARGET/nous-dbbackup.service"
    rm -f "$NOUSCTL_DST"
    for sd in "${SUDOERS[@]}"; do rm -f "/etc/sudoers.d/$sd"; done
    systemctl daemon-reload
    printf '%s✔ 已卸载%s(postgresql 保留)。\n' "$GRN" "$RST"
    ;;

  *)
    echo "Usage: $0 [install|uninstall]" >&2; exit 1 ;;
esac
