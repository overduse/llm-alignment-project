#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Change directory to the project root
cd "$(dirname "$0")/.."
PROJECT_ROOT="${PWD}"

# Verify model path exists
if [ ! -d "${PROJECT_ROOT}/models/Qwen3-8B-Base" ]; then
    echo "Error: Base model directory not found at ${PROJECT_ROOT}/models/Qwen3-8B-Base"
    echo "Please download the model first."
    exit 1
fi

# Verify dataset exists
if [ ! -f "${PROJECT_ROOT}/data/sft_pairs_20k.jsonl" ]; then
    echo "Error: Dataset not found at ${PROJECT_ROOT}/data/sft_pairs_20k.jsonl"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "Starting 8xH100 LLaMA-Factory SFT..."
echo "Model: ./models/Qwen3-8B-Base"
echo "Dataset: data/sft_pairs_20k.jsonl"

# You can use DeepSpeed ZeRO-3 by configuring it in LLaMA-Factory:
deepspeed: config/llamafactory/ds_z3_config.json
llamafactory-cli train config/llamafactory/sft_qwen3_full.yaml
