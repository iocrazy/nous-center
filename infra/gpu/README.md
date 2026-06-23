# GPU 缓解(RTX PRO 6000 Blackwell GSP 固件 bug)

## 症状
跑模型时 **RTX PRO 6000 Blackwell** GSP 固件崩溃 → 卡进 `NV_ERR_GPU_IN_FULLCHIP_RESET`
僵死(`status 0x0f = NV_ERR_GPU_NOT_FULL_POWER`),把同机显示用的 3090 一起拖黑 → 开机黑屏 /
整机硬死,SSH 可能还活但 `nvidia-smi` 掉句柄(`Unable to determine the device handle ... Unknown Error`)。

## 定性
595.71.05 / 580 open 驱动上 Blackwell 的**已知 GSP 固件 bug**,**负载触发**(开机即满载预加载、
或长时间满载推理最易触发)。换驱动 / 装 Windows 均无效(GSP 固件与 OS 无关)。佐证:
open-gpu-kernel-modules issues **#1111**(同款 PRO6000+2×3090 WRX90)/ #1151 / #1134。
完整排查见 Notion「WRX90 工作站 Ubuntu 启动黑屏排查:GPU0 FULLCHIP_RESET 僵死」。

## 处置(分层)

**1. 恢复(崩了之后,唯一可靠)**:**冷断电** —— 关机 → 断 AC(拔电源线 / 关 PSU 硬开关)→
长按机箱电源键 30s 放电 → 等 2-3 分钟 → 通电开机。热重启不一定够(GSP 要重新上电复位)。

**2. 开机不再崩(运维,已做)**:
- **所有模型取消常驻** → nous-backend 开机加载 0 模型,PRO 6000 boot 时空载,不再被 18 秒砸崩。
  模型靠**按需懒加载**(首调自动 load),见 `ensure_vllm_base_url`。
- **nous-aligner 已 `systemctl disable`**(开机不在 3090 上加载对齐模型)。要 ASR 词级时间戳时
  再 `sudo systemctl enable --now nous-aligner`。

**3. 系统级缓解(本目录,降低复发)**:
```bash
sudo ./infra/gpu/setup-gpu-mitigations.sh   # 关 GC6 动态省电 + 开 persistence;重启生效
```

**4. 长期根治方向(未做,需硬件)**:大模型(35B fp8 ~40GB)别再压 PRO 6000 ——
把拔掉的 3090 装回,改 **双 3090 张量并行**(`--tensor-parallel-size 2`,48GB 装得下 40GB),
PRO 6000 只留给短时 / 超大任务,禁长时间满载。#1111 作者验证双 3090 TP 稳跑 20+ 小时。

## BIOS(已做)
Primary Graphics → `Onboard`(ASPEED VGA)+ grub `nvidia_drm.modeset=0` → 控制台走板载 BMC VGA,
显卡僵死不再连累黑屏、BMC KVM 全程可见。
