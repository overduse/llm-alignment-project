#!/usr/bin/env bash
# Full-parameter DPO on 4x or 8x H100. Data: data/generated/dpo_pairs.jsonl.
# Requires: SFT checkpoint (checkpoints/qwen3-8b-sft) and dpo_pairs.jsonl.

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="${PWD}"

# GPUs: set in cluster or default 4 cards
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export FORCE_TORCHRUN=1

SFT_MODEL="${SFT_MODEL:-checkpoints/qwen3-8b-sft}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/qwen3-8b-dpo}"
DPO_DATA="${PROJECT_ROOT}/data/generated/dpo_pairs.jsonl"

if [ ! -d "${PROJECT_ROOT}/${SFT_MODEL}" ]; then
  echo "Error: SFT checkpoint not found: ${PROJECT_ROOT}/${SFT_MODEL}"
  exit 1
fi
if [ ! -f "${DPO_DATA}" ]; then
  echo "Error: DPO data not found: ${DPO_DATA}"
  echo "Run: python src/train/build_dpo_data.py --input data/generated/candidates_scored.jsonl --output data/generated/dpo_pairs.jsonl --format scores"
  exit 1
fi

echo "Starting DPO training..."
echo "  SFT model: ${SFT_MODEL}"
echo "  Output:    ${OUTPUT_DIR}"
echo "  Data:      ${DPO_DATA} ($(wc -l < "${DPO_DATA}") pairs)"
echo "  GPUs:      ${CUDA_VISIBLE_DEVICES}"

llamafactory-cli train config/llamafactory/dpo_qwen3_full.yaml \
  model_name_or_path="${PROJECT_ROOT}/${SFT_MODEL}" \
  ref_model="${PROJECT_ROOT}/${SFT_MODEL}" \
  output_dir="${PROJECT_ROOT}/${OUTPUT_DIR}" \
  dataset_dir="${PROJECT_ROOT}/data"

echo "Done. Checkpoints: ${OUTPUT_DIR}"
