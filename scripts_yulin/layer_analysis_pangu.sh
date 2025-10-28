#!/usr/bin/env bash
set -ex

# npu-smi info
# pip list


PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
cd ${PROJECT_DIR}


python layer_analysis.py \
  --jsonl_path $DATASET_DIR/outputs_yulin_gy/openPangu-Embedded-7B-V1.1/Math_Math/origin_temp0.7_maxlen16000.jsonl \
  --hidden_dir $DATASET_DIR/outputs_yulin_gy/openPangu-Embedded-7B-V1.1/Math_Math/ \
  --layers 1-34 \
  --max_files 500 \
  --expected_offset 0 \
  --alpha 1.0 \
  --pca_components 64 \
  --test_size 0.2 \
  --random_state 42 \
  --file_pattern hidden_{idx:03d}.pt