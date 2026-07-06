#!/usr/bin/env bash
# GPU 稳定性守护(开机由 nous-gpu-guard.service 执行,root)。
#
# 缓解 RTX PRO 6000 Blackwell GSP 固件 bug:开机把 persistence mode 强制打开,
# 做成**开机强制**而非一次性手动脚本 —— 排查发现旧的 setup-gpu-mitigations.sh
# 「enable nvidia-persistenced」其实无效(发行版 unit 带 --no-persistence-mode →
# persistence mode 始终 Disabled),且没人保证它跑过。本 unit 每次开机把状态强制到位。
#
# persistence mode = ON:vLLM 进程退出后驱动不反初始化 GPU,避免频繁加载/卸载
# 反复经历上电-掉电功率状态迁移(稳定性 + 首次加载更快)。
set -euo pipefail

SMI="$(command -v nvidia-smi || true)"
[ -n "$SMI" ] || { echo "nvidia-smi 不存在,跳过 GPU guard" >&2; exit 0; }

# persistence mode 全局开(-pm 1)。失败不致命(驱动/权限问题)。
if "$SMI" -pm 1 >/dev/null 2>&1; then
    echo "✅ persistence mode 已开(-pm 1)"
else
    echo "⚠️ 无法设 persistence mode(驱动/权限?)" >&2
fi
