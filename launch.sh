#!/usr/bin/env bash
set -e

MODEL_ID=${1:-Qwen/Qwen2.5-VL-3B-Instruct}
FAMILY=${2:-qwen2_5}
VERSION=${3:-v4}

IMAGE=${IMAGE:?set IMAGE=/path/to/image.jpg}
QUESTION=${QUESTION:?set QUESTION="your question"}
SAVE_DIR=${SAVE_DIR:-outputs}

python main.py \
    --model-id "$MODEL_ID" \
    --family "$FAMILY" \
    --version "$VERSION" \
    --image "$IMAGE" \
    --question "$QUESTION" \
    --save-dir "$SAVE_DIR"
