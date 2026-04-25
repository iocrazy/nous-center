#!/usr/bin/env bash
# Install nous-backend + nous-cloudflared as systemd services so they survive
# reboots without nohup babysitting. Idempotent — safe to re-run.
#
# Usage:
#   sudo ./infra/systemd/install.sh           # install + enable + start
#   sudo ./infra/systemd/install.sh uninstall # stop + disable + remove
#
# After install:
#   journalctl -u nous-backend -f             # tail backend logs
#   journalctl -u nous-cloudflared -f         # tail tunnel logs
#   systemctl status nous-backend nous-cloudflared

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET=/etc/systemd/system
SERVICES=(nous-backend.service nous-cloudflared.service)

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: must run as root (use sudo)" >&2
  exit 1
fi

case "${1:-install}" in
  install)
    for svc in "${SERVICES[@]}"; do
      echo ">> installing $svc"
      install -m 0644 "$SCRIPT_DIR/$svc" "$TARGET/$svc"
    done
    systemctl daemon-reload
    for svc in "${SERVICES[@]}"; do
      systemctl enable --now "$svc"
    done
    echo
    systemctl --no-pager status "${SERVICES[@]}" | head -40
    echo
    echo "Done. Services will auto-restart on failure and survive reboots."
    echo "Tail logs: journalctl -u nous-backend -f"
    ;;
  uninstall)
    for svc in "${SERVICES[@]}"; do
      echo ">> removing $svc"
      systemctl disable --now "$svc" 2>/dev/null || true
      rm -f "$TARGET/$svc"
    done
    systemctl daemon-reload
    echo "Done."
    ;;
  *)
    echo "Usage: $0 [install|uninstall]" >&2
    exit 1
    ;;
esac
