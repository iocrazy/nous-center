#!/usr/bin/env bash
# Install nous-backend + nous-cloudflared + nous-healthprobe(timer) as systemd
# units so they survive reboots without nohup babysitting. Idempotent — safe to
# re-run.
#
# Usage:
#   sudo ./infra/systemd/install.sh           # install + enable + start
#   sudo ./infra/systemd/install.sh uninstall # stop + disable + remove
#
# After install:
#   journalctl -u nous-backend -f             # tail backend logs
#   journalctl -u nous-cloudflared -f         # tail tunnel logs
#   journalctl -u nous-healthprobe -f         # tail health probe results
#   systemctl status nous-backend nous-cloudflared nous-healthprobe.timer

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET=/etc/systemd/system
# 隧道自愈授权:让探针(heygo)能无密码 restart cloudflared。
SUDOERS_SRC="$SCRIPT_DIR/../security/nous-healthprobe.sudoers"
SUDOERS_DST=/etc/sudoers.d/nous-healthprobe
# Long-running services + the probe timer all get enabled. The probe .service is
# oneshot (no [Install]) — triggered only by the timer, never enabled directly.
SERVICES=(nous-backend.service nous-cloudflared.service nous-status.service)
TIMERS=(nous-healthprobe.timer)
# nous.target:全栈总闸,enable 后开机 + `nousctl up/down` 一键拉起整组(PR-2)。
TARGETS=(nous.target)
# Every unit file copied into /etc/systemd/system (incl. the oneshot probe service + target).
UNIT_FILES=(nous-backend.service nous-cloudflared.service nous-status.service nous-healthprobe.service nous-healthprobe.timer nous.target)
# nousctl 便捷 CLI 装到 PATH。
NOUSCTL_DST=/usr/local/bin/nousctl

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: must run as root (use sudo)" >&2
  exit 1
fi

case "${1:-install}" in
  install)
    for u in "${UNIT_FILES[@]}"; do
      echo ">> installing $u"
      install -m 0644 "$SCRIPT_DIR/$u" "$TARGET/$u"
    done
    systemctl daemon-reload
    for svc in "${SERVICES[@]}"; do
      systemctl enable --now "$svc"
    done
    for tmr in "${TIMERS[@]}"; do
      systemctl enable --now "$tmr"
    done
    # nous.target:enable(开机)。不 --now —— 各服务已 enable --now 起好,target 只是
    # 分组闸,start 它仅确保组员在跑(已在跑则无操作)。
    for tgt in "${TARGETS[@]}"; do
      systemctl enable "$tgt"
    done
    # nousctl 便捷 CLI → /usr/local/bin(可执行)。
    echo ">> installing nousctl ($NOUSCTL_DST)"
    install -m 0755 "$SCRIPT_DIR/nousctl" "$NOUSCTL_DST"
    # 隧道自愈 sudoers:装 0440 + visudo 校验(校验失败立即撤掉,绝不留坏 sudoers)。
    echo ">> installing sudoers drop-in ($SUDOERS_DST)"
    install -m 0440 "$SUDOERS_SRC" "$SUDOERS_DST"
    if ! visudo -cf "$SUDOERS_DST" >/dev/null 2>&1; then
      echo "ERROR: sudoers 校验失败,已撤掉 $SUDOERS_DST" >&2
      rm -f "$SUDOERS_DST"
      exit 1
    fi
    echo
    systemctl --no-pager status "${SERVICES[@]}" "${TIMERS[@]}" | head -50
    echo
    echo "Done. Services auto-restart on failure and survive reboots."
    echo "全栈一键管控: nousctl up | down | restart | status | logs"
    echo "Health probe runs every 2 min: journalctl -u nous-healthprobe -f"
    ;;
  uninstall)
    for tgt in "${TARGETS[@]}"; do
      echo ">> removing $tgt"
      systemctl disable "$tgt" 2>/dev/null || true
      rm -f "$TARGET/$tgt"
    done
    for tmr in "${TIMERS[@]}"; do
      echo ">> removing $tmr"
      systemctl disable --now "$tmr" 2>/dev/null || true
      rm -f "$TARGET/$tmr"
    done
    for svc in "${SERVICES[@]}"; do
      echo ">> removing $svc"
      systemctl disable --now "$svc" 2>/dev/null || true
      rm -f "$TARGET/$svc"
    done
    rm -f "$TARGET/nous-healthprobe.service"
    rm -f "$NOUSCTL_DST"
    rm -f "$SUDOERS_DST"
    systemctl daemon-reload
    echo "Done."
    ;;
  *)
    echo "Usage: $0 [install|uninstall]" >&2
    exit 1
    ;;
esac
