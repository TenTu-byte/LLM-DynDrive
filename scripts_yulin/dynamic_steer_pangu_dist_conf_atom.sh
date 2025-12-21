#!/usr/bin/env bash
printenv

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

# Multi-node inference
for ds in "${datasets[@]}"; do
    echo "=== Running dataset: ${ds} ==="
    torchrun \
        --nnodes=$MA_NUM_HOSTS \
        --node_rank=$VC_TASK_INDEX \
        --nproc_per_node=$MA_NUM_GPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=29500 \
        transformer_inference_steer_dp_pangu_dist_conf.py \
        --model_name_or_path "$models/openPangu-Embedded-7B-V1.1" \
        --dataset_dir "./Data/" \
        --dataset "$ds" \
        --output_path "$outputs/beta/outputs_steer_dynamic_conf" \
        --steer_vector_path "$outputs/beta/openPangu-Embedded-7B-V1.1/Math_Math/steer_vector_layer${steer_layer}_conf_mixed.pt" \
        --steer_layer $steer_layer \
        --steer_coef -1 \
        --run_id $run_id \
        --max_generated_tokens $max_tokens \
        --seed $seed \
        --q25 $q25 \
        --q75 $q75 \
        --low_val $low_val \
        --tau $tau
    echo "=== Finished ${ds} ==="
done
