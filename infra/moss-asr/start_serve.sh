#!/bin/sh
# nous-moss-asr 生产启动脚本(systemd ExecStart 指它)。
# 脚本文件启动(非 inline `sh -c`),故进程 cmdline = `python .venv/bin/sgl-omni serve ...`;
# systemd KillMode=control-group 收整个 cgroup,不靠 pgrep -f 匹配(SPIKE.md:sglang 有
# worker 子进程,cmdline 不含 config 名,pgrep 清不干净会留孤儿吃显存)。
cd "$(dirname "$0")"

# torch 默认 FASTEST_FIRST 枚举 → Pro 6000 会被排到 cuda:0。PCI_BUS_ID 让枚举跟随总线号,
# 配合下面 UUID 钉死后进程内只见这一张卡。
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# UUID 钉到 index0 的 3090,**绝不落 Pro 6000**:Pro 6000 Blackwell GSP 固件有负载触发崩溃
# bug(被压一把模型就 fullchip-reset 僵死、拖黑整机,见 infra/gpu/README.md 与 nous-aligner
# .service 同段注释)。裸 cuda:0 在 PCI_BUS_ID 排序下随槽位变,可能正好命中 Pro 6000 → 不可靠;
# UUID 绑定不随槽位/index 变,永远是这张 3090。进程内只可见它,故 config 里 device=cuda:0 即指它。
# 单机 infra,硬编码 UUID(同本仓库其余绝对路径/UUID 硬编码);换卡需同步改这里。
# 注意:这张 3090 与 nous-aligner 钉的那张(GPU-78dcdbeb…)是两张不同的 3090。
export CUDA_VISIBLE_DEVICES=GPU-2fd7c91c-af39-7b02-66b9-988331ce3bd7

# sgl-omni 首次冷启会现场 nvcc 编译 sgl-kernel(sm_86 无预编译,如 fused_rope);本机无
# /usr/local/cuda,指向 venv 内自带的 cu13 工具链(setup.sh 已装好并建 lib64/libcudart 软链)。
# 三样缺一不可(PR-0 主循环 serve 调通实证):CUDA_HOME 定工具链根;PATH 里 venv/bin 找 ninja、
# cu13/bin 找 nvcc;LD_LIBRARY_PATH 让链接/运行期找到 libcudart 等。warm cache 时不重编(秒级起),
# 但首次部署冷启会真编,缺这些直接编不过。
VENV="$(pwd)/.venv"
export CUDA_HOME="$VENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

# 模型是本地全量检出 + trust_remote_code,不需联网;强制 HF 离线,免得经 mihomo 代理去
# 探 huggingface.co 拖慢/挂起冷启。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 本机 mihomo 代理会拦 127.0.0.1 回环(backend→微服务同机),排除掉。
export NO_PROXY=127.0.0.1,localhost

# 端口默认 8003(拓扑约定;backend 经 NOUS_MOSS_ASR_URL 指过来)。
PORT="${NOUS_MOSS_ASR_PORT:-8003}"

# 日志走 stdout,systemd journal 接管(journalctl -u nous-moss-asr -f);不再 >> 文件。
exec "$VENV/bin/python" "$VENV/bin/sgl-omni" serve \
  --config moss_config.yaml \
  --host 127.0.0.1 --port "$PORT"
