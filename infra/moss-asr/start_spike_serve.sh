#!/bin/sh
# PR-0 spike 启动脚本(避免 inline sh -c 命令行含 "sgl-omni serve" 被 pkill/pgrep -f 自匹配)
cd "$(dirname "$0")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-2fd7c91c-af39-7b02-66b9-988331ce3bd7
export NO_PROXY=127.0.0.1,localhost
export CUDA_HOME=/media/heygo/program/projects-code/repos/nous-engine/infra/moss-asr/.venv/lib/python3.12/site-packages/nvidia/cu13
export PATH="$(pwd)/.venv/bin:$PATH"
exec ./.venv/bin/python .venv/bin/sgl-omni serve \
  --config moss_spike_config.yaml \
  --host 127.0.0.1 --port 8003 \
  >> logs/serve5.log 2>&1
