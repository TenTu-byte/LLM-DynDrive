#!/usr/bin/env bash
set -euo pipefail

# npu-smi info
# pip list


PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
cd ${PROJECT_DIR}

datasets=(
    Math_AIME2024
    Math_Math500
    Math_AIME2025
    Math_AMC23
    Math_GSM8K
    Math_Olympiad
)

for ds in "${datasets[@]}"; do
    echo "=== Evaluating dataset: ${ds} ==="
    python baseline_pangu.py \
      --model_name_or_path "$DATASET_DIR/models_yulin/openPangu-Embedded-7B-V1.1" \
      --dataset_dir "./Data" \
      --dataset "$ds" \
      --output_path "$DATASET_DIR/outputs_yulin/outputs_baseline" \
      --max_generated_tokens 16000 \
      --num_npus 8 \
      --trust_remote_code
    echo "=== Finished ${ds} ==="
done