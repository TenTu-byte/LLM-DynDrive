#!/usr/bin/env bash
set -ex
npu-smi info

# Setup environment
pip install transformers==4.53.2
pip uninstall -y omegaconf
pip install latex2sympy2==1.9.1
pip install word2number==1.1

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH="/usr/local/Ascend/ascend-toolkit/latest"
export HCCL_CONNECT_TIMEOUT=7200 # 2h
export HCCL_EXEC_TIMEOUT=7200    # 2h
export HCCL_IF_BASE_PORT=64000

# Change working directory
PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
cd ${PROJECT_DIR}

# set npu plog env
ma_vj_name=`echo ${MA_VJ_NAME} | sed 's:ma-job:modelarts-job:g'`
task_name="worker-${VC_TASK_INDEX}"
task_plog_path=${MA_LOG_DIR}/${ma_vj_name}/${task_name}
mkdir -p ${task_plog_path}
# export ASCEND_PROCESS_LOG_PATH=${task_plog_path}
export ASCEND_PROCESS_LOG_PATH=${ASCEND_PROCESS_LOG_PATH}/${VC_TASK_INDEX}
echo "plog path: ${ASCEND_PROCESS_LOG_PATH}"

# Resolve master IP
if [[ -z "$MASTER_ADDR" ]]; then
  MASTER_ADDR="${MA_VJ_NAME}-${MA_TASK_NAME}-0.${MA_VJ_NAME}"
  MASTER_ADDR=$(ping "$MASTER_ADDR" -c 1 | sed '1{s/[^(]*(//;s/).*//;q}')
fi

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
        transformer_inference_steer_dp_pangu_dist.py \
        --model_name_or_path "$models/openPangu-Embedded-7B-V1.1" \
        --dataset_dir "./Data/" \
        --dataset "$ds" \
        --output_path "$outputs/test3/outputs_steer_dynamic" \
        --steer_vector_path "$outputs/test3/openPangu-Embedded-7B-V1.1/Math_Math/steer_vector_layer${steer_layer}_conf_mixed.pt" \
        --steer_layer $steer_layer \
        --steer_coef -1 \
        --run_id $run_id \
        --max_generated_tokens $max_tokens \
        --seed $seed \
        --low_val_2 $low_val_2 \
        --high_val_2 $high_val_2
    echo "=== Finished ${ds} ==="
done

# Default values
# v0:
# steer_layer=27
# run_id=v0
# low_val_2=None
# high_val_2=None
#
# v1:
# steer_layer=33
# run_id=v1
# low_val_2=None
# high_val_2=None
#
# v2:
# steer_layer=26
# run_id=v2
# low_val_2=None
# high_val_2=None
#
# all:
# max_tokensS=16000
# seed=42


# --q25c $q25c \
# --q75c $q75c \
# --low_val_1 $low_val_1 \
# --high_val_1 $high_val_1 \
# --q25v $q25v \
# --q75v $q75v \
