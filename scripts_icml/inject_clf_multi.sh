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

python -u inject_clf_multi.py \
  --model_name_or_path "$models/openPangu-Embedded-7B-V1.1" \
  --dataset_dir "./Data/" \
  --dataset "Math_AIME2025" \
  --output_path "$outputs/beta/outputs_steer_dynamic_clf_multicheck" \
  --num_gpus 8 \
  --trust_remote_code \
  --hs_device auto \
  --insert_text "\n[unused17]\n\n" \
  --clf "$outputs/icml/openPangu-Embedded-7B-V1.1/Math_Math/classifer/remain_clf_allk/remain_clf.joblib" \
  --meta "$outputs/icml/openPangu-Embedded-7B-V1.1/Math_Math/classifer/remain_clf_allk/remain_clf_meta.json"