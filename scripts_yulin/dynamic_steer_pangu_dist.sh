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

# Multi-node inference
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
        --output_path "$outputs/outputs_steer_dynamic" \
        --steer_vector_path "$outputs/openPangu-Embedded-7B-V1.1/Math_Math/steer_vector_layer27_conf_mixed.pt" \
        --steer_layer $STEER_LAYER \
        --steer_coef -1 \
        --run_id $RUN_ID \
        --max_generated_tokens $MAX_TOKENS \
        --seed $SEED \
        --low_val_2 $LOW_VAL_2 \
        --high_val_2 $HIGH_VAL_2
    echo "=== Finished ${ds} ==="
done

# Default values
# v0:
# STEER_LAYER=27
# RUN_ID=v0
# LOW_VAL_2=None
# HIGH_VAL_2=None
# v1:
# STEER_LAYER=33
# RUN_ID=v1
# LOW_VAL_2=None
# HIGH_VAL_2=None
# all:
# MAX_TOKENS=16000
# SEED=42
