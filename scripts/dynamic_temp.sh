#!/usr/bin/env bash
set -ex
nvidia-smi

# Setup environment

# Change working directory
PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
cd ${PROJECT_DIR}

env

# Dataset switches (1=run, 0=skip)
run_aime2024=${run_aime2024:-1}
run_aime2025=${run_aime2025:-1}
run_amc23=${run_amc23:-1}
run_math500=${run_math500:-1}
run_gsm8k=${run_gsm8k:-1}
run_olympiad=${run_olympiad:-1}

# Build datasets array based on switches
datasets=()
[[ $run_aime2024 -eq 1 ]] && datasets+=(Math_AIME2024)
[[ $run_aime2025 -eq 1 ]] && datasets+=(Math_AIME2025)
[[ $run_amc23 -eq 1 ]] && datasets+=(Math_AMC23)
[[ $run_math500 -eq 1 ]] && datasets+=(Math_Math500)
[[ $run_gsm8k -eq 1 ]] && datasets+=(Math_GSM8K)
[[ $run_olympiad -eq 1 ]] && datasets+=(Math_Olympiad)

echo "=== Datasets to run: ${datasets[@]} ==="

# Multi-GPU inference
for ds in "${datasets[@]}"; do
    echo "=== Running dataset: ${ds} ==="
    python -u transformer_inference_temp_dp.py \
        --model_name_or_path "$models/$model_name" \
        --dataset_dir "./Data/" \
        --output_path "$outputs/outputs_temp_dynamic/$model_name" \
        --dataset "$ds" \
        --max_generated_tokens $max_tokens \
        --num_gpus 8 \
        --steer_vector_path "$outputs/vectors/$model_name/steer_vector.pt" \
        --steer_layer 58 \
        --steer_coef -1 \
        --temperatuyre 1
    echo "=== Finished ${ds} ==="
done
