#!/usr/bin/env bash
# nous-aligner 微服务一次性搭建:独立 venv(qwen-asr,与 backend 隔离)。
# 以 heygo 身份跑(不用 root)。重建 prod 检出 / 升级 qwen-asr 时重跑。
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$DIR/.venv"

echo "==> 建 aligner venv: $VENV (py3.12)"
uv venv --python 3.12 "$VENV"

echo "==> 装 qwen-asr(钉 transformers 4.57,~5GB:torch/transformers/qwen-asr)"
uv pip install --python "$VENV/bin/python" -r "$DIR/requirements.txt"

echo "==> 自检:导入 + CUDA 可见"
"$VENV/bin/python" - <<'PY'
import torch
from qwen_asr import Qwen3ForcedAligner  # noqa: F401
print("qwen-asr import OK, cuda_available=", torch.cuda.is_available(), "n_gpu=", torch.cuda.device_count())
PY

echo "==> 完成。模型需先下:"
echo "    modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir \$MODELS_ROOT/nous/speech/Qwen3-ForcedAligner-0.6B"
echo "    启动: $VENV/bin/python $DIR/aligner_service.py  (或 systemd: nous-aligner.service)"
