#!/usr/bin/env bash

# Change working directory
PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
cd ${PROJECT_DIR}

# Set parameters
model_name=DeepSeek-R1-Distill-Qwen-1.5B
models=$DATASET_DIR/models_yulin
outputs=$DATASET_DIR/outputs_yulin/outputs_temp_dynamic

# merge shards
datasets=(
    Math_AIME2024
    Math_AIME2025
    Math_AMC23
    Math_Math500
    Math_GSM8K
    Math_Olympiad
)

for ds in "${datasets[@]}"; do
    echo "=== Checking dataset: ${ds} ==="
    python check.py \
        --model_name_or_path $models/$model_name \
        --data_name "$ds" \
        --generation_path "$outputs/$model_name/$ds/steer_temp1.0_maxlen16000.merged.jsonl"
    echo "=== Finished ${ds} ==="
done
