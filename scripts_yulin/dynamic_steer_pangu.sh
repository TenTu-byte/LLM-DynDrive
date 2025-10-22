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
    echo "=== Running dataset: ${ds} ==="
    python -u transformer_inference_steer_dp_pangu.py \
        --model_name_or_path "$DATASET_DIR/models_yulin/openPangu-Embedded-7B-V1.1" \
        --dataset_dir "./Data/" \
        --output_path "$DATASET_DIR/outputs_yulin/outputs_steer_dynamic" \
        --dataset "$ds" \
        --max_generated_tokens 16000 \
        --num_npus 8 \
        --steer_vector_path $DATASET_DIR/outputs_yulin/openPangu-Embedded-7B-V1.1/Math_Math/steer_vector_layer27_conf_mixed.pt \
        --steer_layer 27 \
        --steer_coef -1
    echo "=== Finished ${ds} ==="
done
