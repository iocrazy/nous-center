#!/bin/bash
set -e
echo "Starting vLLM on GPU1..."
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen2.5-VL-7B-Instruct \
    --port 8100 \
    --max-model-len 4096 \
    --trust-remote-code
