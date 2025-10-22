#!/usr/bin/env bash
# One-click merge for all dataset eval result shards.
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
    # Math_GSM8K
    # Math_Olympiad
)

for ds in "${datasets[@]}"; do
    echo "=== Merging dataset: ${ds} ==="
    python merge_shards.py \
        --dir "$DATASET_DIR/outputs_yulin/outputs_steer_dynamic/openPangu-Embedded-7B-V1.1/${ds}"\
        --base 'steer_temp0.7_maxlen16000'
    echo "=== Finished ${ds} ==="
done
