#!/bin/bash
set -e
echo "Starting FastAPI..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
