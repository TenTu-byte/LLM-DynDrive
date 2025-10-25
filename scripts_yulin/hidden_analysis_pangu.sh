#!/usr/bin/env bash
set -euo pipefail

# npu-smi info
# pip list


PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
cd ${PROJECT_DIR}

LAYER_ID=33  # v0: 27
THRESHOLD=0.74  # v0: 0.75

python hidden_analysis_mixed_auto_pangu.py \
  --layer_id $LAYER_ID \
  --jsonl_path "$DATASET_DIR/outputs_yulin_gy/openPangu-Embedded-7B-V1.1/Math_Math/origin_temp0.7_maxlen16000.jsonl" \
  --hidden_dir "$DATASET_DIR/outputs_yulin_gy/openPangu-Embedded-7B-V1.1/Math_Math/" \
  --save_path  "$DATASET_DIR/outputs_yulin_gy/openPangu-Embedded-7B-V1.1/Math_Math/steer_vector_layer${LAYER_ID}_conf_mixed.pt" \
  --threshold $THRESHOLD \
  --max_files 500 \
  --expected_offset 1 \
  --device npu \
  --report_path  "$DATASET_DIR/outputs_yulin_gy/openPangu-Embedded-7B-V1.1/Math_Math/analysis_layer${LAYER_ID}.json" \
  --verbose