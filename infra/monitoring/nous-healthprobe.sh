#!/usr/bin/env bash
# nous-engine 本地健康巡检(2026-06-16 稳定性加固,本地巡检+日志阶段)。
#
# 由 nous-engine-healthprobe.timer 每 2 分钟触发一次,探 vLLM 看门狗管不到的事:
#   1. 后端本机存活      (GET 127.0.0.1:8000/healthz → 200)
#   2. 后端自报健康      (GET /health → status/database/load_failures)
#   3. 公网隧道存活      (GET <public>/health → 非 530/000)—— 默认关闭,见下
#
# 输出结构化行到 stdout(systemd timer → journald;`journalctl -u nous-engine-healthprobe`)。
# 退出码:有「硬故障」(后端连不上 / DB 挂 / 隧道 down)→ 非 0(systemd 标 failed,
# 将来挂 OnFailure= 告警 hook 即可直接接告警通道);仅「软降级」(degraded /
# load_failures)→ 0 但日志 WARN。
#
# 隧道自愈(2026-06-16):cloudflared 在烂网络上会进「半开僵尸」态 —— 进程 active、
# 但 edge 连接全死且它自己不重连(2026-06-17 卡死 2.5h、systemd 全程显示 active)。
# 本脚本探到「**后端本机健康 但 公网持续 530**」连续 NOUS_AUTOHEAL_THRESHOLD 次 →
# `sudo systemctl restart nous-engine-cloudflared` 自愈(类比 vLLM 看门狗,但针对隧道)。
# 只在「后端活、唯独隧道死」时动手 —— 后端本身挂了重启隧道没用,不碰。需 sudoers
# drop-in 授权 heygo 无密码重启该服务(infra/security/nous-engine-healthprobe.sudoers)。
# NOUS_TUNNEL_AUTOHEAL=0 可关自愈,只巡检+日志。PUBLIC_BASE 为空时自愈同样不触发
# (隧道退役场景:否则会每 ~4 分钟 restart 一个已 mask、注定失败的服务)。
#
# 手动单跑:infra/monitoring/nous-healthprobe.sh
set -uo pipefail

LOCAL_BASE="${NOUS_LOCAL_URL:-http://127.0.0.1:8000}"
# 公网隧道探针开关(2026-07-23):api.iocrazy.com 隧道已退役 —— 凭证 cert.pem 丢失,
# cloudflared 起不来(`error parsing tunnel ID: Error locating origin cert`),
# nous-engine-cloudflared 已 mask。留空 = 跳过第 3 项探针**与隧道自愈**。
# 若不留空,自愈逻辑会每 ~4 分钟 restart 一个注定失败的服务,和 mask 对着干刷日志。
# 将来恢复隧道:重新 `cloudflared tunnel login` 拿 cert.pem,再设
# NOUS_PUBLIC_URL=https://<host> 即可原样启用探针 + 自愈,本脚本无需再改。
PUBLIC_BASE="${NOUS_PUBLIC_URL-}"
TIMEOUT="${NOUS_PROBE_TIMEOUT:-10}"
STATE_FILE="${XDG_RUNTIME_DIR:-/tmp}/nous-healthprobe.state"
AUTOHEAL="${NOUS_TUNNEL_AUTOHEAL:-1}"       # 1=探到隧道僵尸自动 restart cloudflared
AUTOHEAL_THRESHOLD="${NOUS_AUTOHEAL_THRESHOLD:-2}"  # 连续 N 次(×2min)才动手,防单次抖动误重启

CURL=(curl -s --noproxy '*' --max-time "$TIMEOUT")
alerts=()   # 硬故障
warns=()    # 软降级

ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }

# --- 1. 后端本机存活 ---
live_code="$("${CURL[@]}" -o /dev/null -w '%{http_code}' "$LOCAL_BASE/healthz" 2>/dev/null || echo 000)"
if [[ "$live_code" != "200" ]]; then
  alerts+=("backend-local-down(/healthz=$live_code)")
fi

# --- 2. 后端自报健康(只在本机活着时才解析)---
if [[ "$live_code" == "200" ]]; then
  health_json="$("${CURL[@]}" "$LOCAL_BASE/health" 2>/dev/null || echo '')"
  parsed="$(printf '%s' "$health_json" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("PARSE_FAIL"); sys.exit(0)
db = d.get("database")
lf = d.get("load_failures") or {}
out = []
# 只报「可执行」信号。不报裸 status==degraded:本部署 Lane-K llm runner supervisor
# 常驻 running:false → status 恒 degraded,但 vLLM 由 model_manager 独立 spawn、LLM
# 服务照常(qwen35 真机验过)。报它=每 2 分钟一次噪声。DB 挂(硬)/ 模型加载失败(软)
# 才是真要看的;后端死/隧道死由本脚本另两项探针覆盖。
if db != "ok":
    out.append("ALERT db=%s" % db)
if lf:
    out.append("WARN load_failures=%s" % ",".join(lf.keys()))
print("\n".join(out))
' 2>/dev/null || echo 'PARSE_FAIL')"
  if [[ "$parsed" == "PARSE_FAIL" ]]; then
    warns+=("health-json-unparseable")
  else
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      case "$line" in
        ALERT\ *) alerts+=("${line#ALERT }") ;;
        WARN\ *)  warns+=("${line#WARN }") ;;
      esac
    done <<< "$parsed"
  fi
fi

# --- 3. 公网隧道存活(仅当 PUBLIC_BASE 非空)---
# skip = 未配置公网地址,该项不参与 alerts / 自愈 / 退出码。
pub_code="skip"
if [[ -n "$PUBLIC_BASE" ]]; then
  pub_code="$("${CURL[@]}" -o /dev/null -w '%{http_code}' "$PUBLIC_BASE/health" 2>/dev/null || echo 000)"
  # 530 = cloudflare 隧道 down(origin 不可达);000 = 连不上。2xx/3xx/4xx 都算隧道通。
  if [[ "$pub_code" == "000" || "$pub_code" == "530" || "$pub_code" == "502" ]]; then
    alerts+=("public-tunnel-down(<public>/health=$pub_code)")
  fi
fi

# --- 连续计数(状态文件:<硬故障streak> <隧道僵尸streak> <ts>)---
prev_hard=0; prev_tunnel=0
# shellcheck disable=SC2034  # _ts 仅占位
[[ -f "$STATE_FILE" ]] && read -r prev_hard prev_tunnel _ts < "$STATE_FILE" 2>/dev/null || true
[[ "$prev_hard" =~ ^[0-9]+$ ]] || prev_hard=0
[[ "$prev_tunnel" =~ ^[0-9]+$ ]] || prev_tunnel=0

# 「隧道僵尸」专项 streak:后端本机健康 但 公网 530/502/000 = restart cloudflared 才有意义
# (后端也挂了 → 是别的问题,重启隧道无用,不碰)。
# pub_code="skip"(未配置公网地址)时恒为 0 —— 不触发自愈。
tunnel_zombie=0
if [[ "$live_code" == "200" && ( "$pub_code" == "000" || "$pub_code" == "502" || "$pub_code" == "530" ) ]]; then
  tunnel_zombie=1
fi
cur_tunnel=0
(( tunnel_zombie )) && cur_tunnel=$((prev_tunnel + 1))
cur_hard=0
(( ${#alerts[@]} > 0 )) && cur_hard=$((prev_hard + 1))

# --- 隧道自愈:连续 N 次隧道僵尸 → restart cloudflared(它自己不重连半开连接)---
healed=""
if [[ "$AUTOHEAL" == "1" ]] && (( cur_tunnel >= AUTOHEAL_THRESHOLD )); then
  if sudo -n /usr/bin/systemctl restart nous-engine-cloudflared 2>/dev/null; then
    healed="[HEAL] restart nous-engine-cloudflared(隧道连续 ${cur_tunnel} 次僵尸)"
    cur_tunnel=0  # 重置 → 给重连留时间;没修好下轮重新累计(自带 ~${AUTOHEAL_THRESHOLD}×2min 冷却)
  else
    healed="[HEAL-FAIL] 想 restart nous-engine-cloudflared 但 sudo 无权限 —— 装 infra/security/nous-engine-healthprobe.sudoers"
  fi
fi

echo "$cur_hard $cur_tunnel $(ts)" > "$STATE_FILE" 2>/dev/null || true

# --- 日志 + 退出码 ---
if (( ${#alerts[@]} > 0 )); then
  echo "[ALERT] $(ts) 连续第 ${cur_hard} 次 — $(IFS='; '; echo "${alerts[*]}")${warns:+; 降级: $(IFS='; '; echo "${warns[*]}")}${healed:+; $healed}"
  exit 1
fi
if (( ${#warns[@]} > 0 )); then
  echo "[WARN]  $(ts) 后端活但有降级 — $(IFS='; '; echo "${warns[*]}")"
  exit 0
fi
if [[ "$pub_code" == "skip" ]]; then
  echo "[OK]    $(ts) backend healthy (local=$live_code; 隧道探针未启用)"
else
  echo "[OK]    $(ts) backend+tunnel healthy (local=$live_code public=$pub_code)"
fi
exit 0
