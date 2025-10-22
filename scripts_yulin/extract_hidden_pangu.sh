#!/usr/bin/env bash
set -euo pipefail

# npu-smi info
# pip list


PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
cd ${PROJECT_DIR}

python transformer_inference_dp_pangu.py \
  --model_name_or_path "$DATASET_DIR/models_yulin/openPangu-Embedded-7B-V1.1" \
  --dataset_dir "./Data" \
  --dataset "Math_Math" \
  --output_path "$DATASET_DIR/outputs_yulin" \
  --max_generated_tokens 16000 \
  --num_npus 8 \
  --trust_remote_code