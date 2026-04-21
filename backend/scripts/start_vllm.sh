#!/bin/bash
set -e

# Preflight: ensure vllm CLI is available in the current venv.
# vllm now lives in [project.optional-dependencies].inference (pyproject.toml)
# because the API server code does NOT import it (talks over HTTP). If you see
# "vllm: command not found", run:
#     uv sync --extra inference
# in this backend/ directory to install it.
if ! command -v vllm >/dev/null 2>&1; then
    echo "error: vllm CLI not found in PATH" >&2
    echo "hint: run 'uv sync --extra inference' to install it" >&2
    exit 1
fi

echo "Starting vLLM on GPU1..."
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen2.5-VL-7B-Instruct \
    --port 8100 \
    --max-model-len 4096 \
    --trust-remote-code
