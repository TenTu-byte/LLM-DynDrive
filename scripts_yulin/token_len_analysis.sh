#!/usr/bin/env bash
set -ex

PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
cd ${PROJECT_DIR}

python token_len_analysis.py \
  --input_path $DATASET_DIR/outputs_yulin_gy/outputs_baseline/openPangu-Embedded-7B-V1.1/Math_AIME2025/maxlen32000_seed42.jsonl \
  --model_name_or_path $DATASET_DIR/models_yulin_gy/openPangu-Embedded-7B-V1.1 \
  --num_samples 3 \