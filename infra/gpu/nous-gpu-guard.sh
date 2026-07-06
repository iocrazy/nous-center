#!/usr/bin/env bash
# GPU 稳定性守护(开机由 nous-gpu-guard.service 执行,root)。
#
# 缓解 RTX PRO 6000 Blackwell GSP 固件 bug 的两条硬件侧措施,做成**开机强制**
# 而非一次性手动脚本 —— 排查发现旧的 setup-gpu-mitigations.sh「enable
# nvidia-persistenced」其实无效(发行版 unit 带 --no-persistence-mode → persistence
# mode 始终 Disabled),且没人保证它跑过。本 unit 每次开机把状态强制到位。
#
#   1) persistence mode = ON:vLLM 进程退出后驱动不反初始化 GPU,避免频繁
#      加载/卸载反复经历上电-掉电功率状态迁移(稳定性 + 首次加载更快)。
#   2) 3090 功率封顶:3090 的瞬时功率尖峰臭名昭著,2×3090 + PRO6000 共用 PSU/
#      供电轨,封顶降低"重启后多卡并发加载"的相关联电流冲击(掉总线诱因)。
#      PRO 6000 的崩溃是 GSP 固件(非功率),故只封 3090,留其性能余量。
#
# 功率上限可用环境变量覆盖(在 service 里 Environment= 设):
#   NOUS_GPU_3090_WATTS(默认 320)—— 3090 stable 值,几乎无性能损失。
set -euo pipefail

SMI="$(command -v nvidia-smi || true)"
[ -n "$SMI" ] || { echo "nvidia-smi 不存在,跳过 GPU guard" >&2; exit 0; }

CAP_3090="${NOUS_GPU_3090_WATTS:-320}"

# persistence mode 全局开(-pm 1)。失败不致命(驱动/权限问题)。
"$SMI" -pm 1 >/dev/null 2>&1 || echo "⚠️ 无法设 persistence mode" >&2

# 逐卡按型号封功率。只动 3090;其它卡(PRO 6000 等)保持默认。
while IFS=, read -r idx name; do
    idx="${idx// /}"; name="${name## }"
    case "$name" in
        *3090*)
            if "$SMI" -i "$idx" -pl "$CAP_3090" >/dev/null 2>&1; then
                echo "GPU $idx ($name): power limit → ${CAP_3090}W"
            else
                echo "⚠️ GPU $idx ($name): 设功率上限 ${CAP_3090}W 失败(可能低于最小允许值)" >&2
            fi
            ;;
        *)
            echo "GPU $idx ($name): 保持默认功率(不封顶)"
            ;;
    esac
done < <("$SMI" --query-gpu=index,name --format=csv,noheader)

echo "✅ nous-gpu-guard 应用完成"
