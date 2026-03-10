#!/bin/bash
set -e

echo "Starting Celery workers..."

# Image worker on GPU0
CUDA_VISIBLE_DEVICES=0 celery -A src.workers.celery_app worker \
    --queue=image \
    --concurrency=1 \
    --hostname=image@%h \
    -l info &

# TTS worker on GPU1
CUDA_VISIBLE_DEVICES=1 celery -A src.workers.celery_app worker \
    --queue=tts \
    --concurrency=1 \
    --hostname=tts@%h \
    -l info &

# Video worker (will use both GPUs when active)
celery -A src.workers.celery_app worker \
    --queue=video \
    --concurrency=1 \
    --hostname=video@%h \
    -l info &

wait
