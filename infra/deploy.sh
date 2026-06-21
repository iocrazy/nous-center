#!/usr/bin/env bash
# nous-center 一键上线(2026-06-19)。
#
#   ./infra/deploy.sh
#
# 流程:拉 master → 前端 build → 重启后端 → 自检。每步 fail-loud,**绝不静默**——
# 杜绝「build 没成 / dist 没更新,却以为上线了」的乌龙(2026-06-19 矩阵那次踩到:
# 后端重启了但 dist 停在旧版,页面没变,来回好几趟才发现)。
#
# 为什么需要它:这台机器**没有自动部署** —— 后端 serve 编译好的 frontend/dist(不是源码),
# 且作为常驻进程跑;改了得 ① 前端 build ② 重启后端 才生效。本脚本把这两步 + 拉代码 + 自检
# 收成一条命令。
#
# 以 heygo 身份跑(git/npm 不能用 root,否则在仓库里造 root 属主文件);只有重启那步用 sudo
# (会提示输密码,除非配了 NOPASSWD)。
#
# ⚠️ 重启后端会**卸掉所有已加载模型**,vLLM ~30s 重载 —— 生产高峰期慎发版。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL="${NOUS_LOCAL_URL:-http://127.0.0.1:8000}"
PUBLIC="${NOUS_PUBLIC_URL:-https://api.iocrazy.com}"

if [[ "${EUID}" -eq 0 ]]; then
  echo "ERROR: 别用 root/sudo 跑整个脚本(会在仓库造 root 属主文件)。直接 ./infra/deploy.sh,只有重启那步会自己提权。" >&2
  exit 1
fi

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m✅ %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m❌ %s\033[0m\n' "$*" >&2; exit 1; }

cd "$REPO"

# ---------- 1. 同步 master(只在专用生产检出)----------
# 生产与 dev 检出分离(2026-06-21):systemd 指向专用 nous-prod 检出,dev session 各用
# 自己的 worktree,谁都不碰生产。生产检出 detached、只读、deploy 专用 → fetch + reset
# --hard 同步到 origin/master(gitignore 的 .env/dist/node_modules 不受影响)。
# .nous-production 标记防呆:dev 检出误跑 deploy.sh 会被 reset --hard 卷走未提交改动 → 直接拒绝。
step "同步到 origin/master"
[[ -f "$REPO/.nous-production" ]] || die "不在专用生产检出(缺 .nous-production 标记)。生产部署只在 nous-prod 跑;dev 改动走 PR→CI→master 后,在 nous-prod 里 ./infra/deploy.sh。"
git fetch origin master -q || die "git fetch 失败。"
git reset --hard origin/master || die "git reset --hard origin/master 失败。"
ok "已同步到 $(git rev-parse --short HEAD)"

# ---------- 2. 前端 build(fail-loud + dist 时间戳校验)----------
step "前端 build"
build_start=$(date +%s)
cd "$REPO/frontend"
if npm run build; then
  ok "npm run build 完成"
elif [[ -d src/wasm/pkg ]]; then
  # prebuild(wasm-pack)常因缺 rust 工具链挂;wasm pkg 已存在时回退直接 tsc+vite。
  echo ">> npm run build 失败,回退 tsc -b && vite build(复用现有 wasm pkg)"
  ./node_modules/.bin/tsc -b && ./node_modules/.bin/vite build || die "前端 build 失败(tsc/vite)。"
  ok "回退 build 完成"
else
  die "前端 build 失败,且无 src/wasm/pkg 可回退 —— 先 npm run wasm:build。"
fi
# 关键防呆:dist 必须真被这次 build 写新(否则就是『以为上了、其实没动』)。
dist_mtime=$(stat -c %Y "$REPO/frontend/dist/index.html" 2>/dev/null || echo 0)
(( dist_mtime >= build_start )) || die "frontend/dist 没更新(mtime 早于 build 开始)—— build 没真正产出,中止,不重启。"
ok "dist 已更新($(date -d @"$dist_mtime" '+%H:%M:%S'))"
cd "$REPO"

# ---------- 3. 重启后端(sudo;cloudflared 经 PartOf 跟随)----------
step "重启 nous-backend(卸全模型,vLLM ~30s 重载)"
sudo systemctl restart nous-backend || die "systemctl restart 失败。"
ok "已发出重启"

# ---------- 4. 自检:等本机 /healthz 200 ----------
step "自检"
for i in $(seq 1 45); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' --max-time 5 "$LOCAL/healthz" 2>/dev/null || echo 000)
  [[ "$code" == "200" ]] && { ok "后端本机健康(/healthz 200,${i}×2s)"; break; }
  [[ $i -eq 45 ]] && die "等 90s 后端仍未起来(/healthz=$code)。查 journalctl -u nous-backend -n 50。"
  sleep 2
done
pub=$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' --max-time 12 "$PUBLIC/healthz" 2>/dev/null || echo 000)
if [[ "$pub" == "200" ]]; then ok "公网隧道通(<public>/healthz 200)"
else echo "⚠️  公网 <public>/healthz=$pub(隧道可能还在重连;巡检/隧道自愈会兜,过会儿再看)"; fi

printf '\n\033[1;32m✅ 上线完成 — %s\033[0m\n' "$(git rev-parse --short HEAD)"
