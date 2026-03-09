#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Change directory to the project root
cd "$(dirname "$0")/.."
PROJECT_ROOT="${PWD}"

echo "========================================================"
echo " Starting Offline DPO Data Generation Pipeline"
echo " (Using vLLM for ultra-fast candidate generation)"
echo "========================================================"

# Step 1: Generate candidates offline using vLLM (this will use all 4 H100 GPUs)
echo -e "\n[1/3] Generating candidates using Qwen3-8B-SFT..."
python src/generate/generate_dpo_candidates_offline.py \
    --model checkpoints/qwen3-8b-sft \
    --input data/processed/dpo_seed_prompts.jsonl \
    --output data/generated/candidates_unscored.jsonl \
    --tp 4 \
    --num-candidates 4

# Crucial: vLLM script will automatically exit and free the GPU memory here.

# Step 2: Score candidates offline using the local Judge model
# (We load the judge model onto the freed GPUs)
echo -e "\n[2/3] Scoring candidates using Judge Model (Qwen3-8B-Base)..."
python src/generate/score_candidates_offline.py \
    --input data/generated/candidates_unscored.jsonl \
    --output data/generated/candidates_scored.jsonl \
    --judge-model models/Qwen3-8B-Base

# Step 3: Format the scored candidates into DPO pairs
echo -e "\n[3/3] Formatting scores into DPO pairs (chosen/rejected)..."
python src/train/build_dpo_data.py \
    --input data/generated/candidates_scored.jsonl \
    --output data/generated/dpo_pairs.jsonl \
    --format scores

echo -e "\n========================================================"
echo "✅ Pipeline Complete!"
echo "Your DPO dataset is ready at: data/generated/dpo_pairs.jsonl"
echo "You can now run: bash scripts/run_dpo_llamafactory.sh"
echo "========================================================"
