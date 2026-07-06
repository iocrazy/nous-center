#!/usr/bin/env bash
# 一次性给本机 GPU 做 Blackwell GSP 固件 bug 缓解(非根治)。
#
#   sudo ./infra/gpu/setup-gpu-mitigations.sh
#
# 根治不存在(595/580 open 驱动上 PRO 6000 Blackwell 的 GSP 固件 bug,负载触发,换驱动无效)。
# 真正的「修」是运维层:① 开机别预加载模型压 PRO 6000(已把所有模型取消常驻,boot 全空载,
# 靠按需懒加载)② 别长时间满载压 PRO 6000(大模型考虑双 3090 张量并行)③ 崩了冷断电恢复。
# 本脚本做的是降低复发概率的两条系统级缓解。详见 infra/gpu/README.md + Notion FULLCHIP_RESET 篇。
set -euo pipefail
[ "${EUID}" -eq 0 ] || { echo "需 root:sudo $0" >&2; exit 1; }
SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">> 装 modprobe drop-in:关 nvidia 动态省电(GC6)"
install -m 0644 "$SD/nvidia-power.conf" /etc/modprobe.d/nvidia-power.conf

echo ">> update-initramfs(让 modprobe 改动进 initramfs,下次启动生效)"
update-initramfs -u

echo ">> enable nvidia-persistenced(unit 存在则起,注意发行版默认带 --no-persistence-mode,"
echo "   真正把 persistence mode 打开的是下面的 nous-gpu-guard(-pm 1))"
systemctl enable --now nvidia-persistenced.service || \
  echo "   ⚠️ nvidia-persistenced enable 失败(驱动未装该 unit?跳过)"

echo ">> 装 + enable nous-gpu-guard(开机强制 persistence mode + 3090 功率封顶)"
# 排查发现:旧脚本只 enable nvidia-persistenced,但发行版 unit 带 --no-persistence-mode
# → persistence mode 始终 Disabled;且脚本是一次性,没人保证跑过。改成开机 systemd
# oneshot 强制到位(-pm 1 + 3090 -pl),每次启动生效,不再"缓解没上机"。
chmod +x "$SD/nous-gpu-guard.sh"
install -m 0644 "$SD/nous-gpu-guard.service" /etc/systemd/system/nous-gpu-guard.service
systemctl daemon-reload
systemctl enable --now nous-gpu-guard.service && \
  echo "   ✅ nous-gpu-guard 已 enable + 立即执行" || \
  echo "   ⚠️ nous-gpu-guard 启动失败,看 journalctl -u nous-gpu-guard"

echo
echo "✅ 完成。modprobe 改动需**重启**生效;persistence/功率封顶已即时应用(并随开机强制)。"
echo "   建议配合冷断电(断 AC 2-3min)重启,既应用本缓解,又复位掉 PRO 6000 的"
echo "   fullchip-reset 僵死。核验:nvidia-smi --query-gpu=index,persistence_mode,"
echo "   power.limit --format=csv"
