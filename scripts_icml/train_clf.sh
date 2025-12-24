#!/usr/bin/env bash
set -ex

# Change working directory
PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
cd ${PROJECT_DIR}

python train_clf.py \
  --folder "$outputs/icml/openPangu-Embedded-7B-V1.1/Math_Math" \
  --k -1 \
  --layer -1 \
  --max_points 50000 \
  --out_dir "$outputs/icml/openPangu-Embedded-7B-V1.1/Math_Math/classifer/remain_clf_allk" \
  --pos_is_short