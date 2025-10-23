#!/usr/bin/env bash
set -euo pipefail

# npu-smi info
# pip list


PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
cd ${PROJECT_DIR}


data_names=(
    Math_AIME2024
    # Math_Math500
    # Math_AIME2025
    # Math_AMC23
    # Math_GSM8K
    # Math_Olympiad
)

# Dynamic steering
# for dn in "${data_names[@]}"; do
#     echo "=== Evaluating dataset: ${dn} ==="
#     python eval_pangu.py \
#         --model_name_or_path "$DATASET_DIR/models_yulin/openPangu-Embedded-7B-V1.1" \
#         --data_name "$dn" \
#         --generation_path "$DATASET_DIR/outputs_yulin/outputs_steer_dynamic/openPangu-Embedded-7B-V1.1/${dn}/steer_temp0.7_maxlen16000.merged.jsonl"
#     echo "=== Finished ${dn} ==="
# done

# Baseline
for dn in "${data_names[@]}"; do
    echo "=== Evaluating dataset: ${dn} ==="
    python eval_pangu.py \
        --model_name_or_path "$DATASET_DIR/models_yulin/openPangu-Embedded-7B-V1.1" \
        --data_name "$dn" \
        --generation_path "$DATASET_DIR/outputs_yulin/outputs_baseline/openPangu-Embedded-7B-V1.1/${dn}/origin_temp0.7_maxlen16000.merged.jsonl"
    echo "=== Finished ${dn} ==="
done