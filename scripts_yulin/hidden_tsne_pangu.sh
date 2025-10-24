#!/usr/bin/env bash
set -euo pipefail

# npu-smi info
# pip list


PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
cd ${PROJECT_DIR}

LAYER_ID=27
THRESHOLD=0.75

python hidden_analysis_tsne_pangu.py \
  --layer_id $LAYER_ID \
  --jsonl_path "$DATASET_DIR/outputs_yulin/openPangu-Embedded-7B-V1.1/Math_Math/origin_temp0.7_maxlen16000.merged.jsonl" \
  --hidden_dir "$DATASET_DIR/outputs_yulin/openPangu-Embedded-7B-V1.1/Math_Math/" \
  --output_png "$DATASET_DIR/outputs_yulin/openPangu-Embedded-7B-V1.1/Math_Math/tsne_layer27_conf_7b.png" \
  --output_csv "$DATASET_DIR/outputs_yulin/openPangu-Embedded-7B-V1.1/Math_Math/tsne_layer27_conf_7b.csv" \
  --max_files 500 \
  --expected_offset 1 \
  --perplexity 30 \
  --n_iter 1000 \
  --pca_dim 50
