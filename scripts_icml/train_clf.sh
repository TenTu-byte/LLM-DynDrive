#!/usr/bin/env bash
set -ex

# Change working directory
PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
cd ${PROJECT_DIR}

Q_MAX_LIST=(0.02 0.03 0.05 0.07 0.08 0.1 0.12)

for q_max in "${Q_MAX_LIST[@]}"; do
    python train_clf.py \
      --folder "$DATASET_DIR/outputs_yulin_gy/icml/openPangu-Embedded-7B-V1.1/Math_Math" \
      --k -1 \
      --layer -1 \
      --max_points 50000 \
      --out_dir "$DATASET_DIR/outputs_yulin_gy/icml/openPangu-Embedded-7B-V1.1/Math_Math/classifer/remain_clf_allk" \
      --pos_is_short \
      --q_max ${q_max}
done