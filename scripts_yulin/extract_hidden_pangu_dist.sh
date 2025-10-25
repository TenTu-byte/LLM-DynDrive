set -euo pipefail
npu-smi info

# Setup environment
pip install transformers==4.53.2
pip uninstall -y omegaconf
pip install latex2sympy2==1.9.1
pip install word2number==1.1

# Change working directory
PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
cd ${PROJECT_DIR}

# Multi-node inference
torchrun \
	--nnodes=$MA_NUM_HOSTS \
	--node_rank=$VC_TASK_INDEX \
	--nproc_per_node=$MA_NUM_GPUS \
	--master_addr=$MA_CURRENT_HOST_IP \
	--master_port=29500 \
	transformer_inference_dp_pangu_dist.py \
	--model_name_or_path "$models/openPangu-Embedded-7B-V1.1" \
	--dataset_dir "./Data" \
	--dataset Math_Math \
	--output_path "$outputs" \
  	--max_generated_tokens 16000 \
	--trust_remote_code
