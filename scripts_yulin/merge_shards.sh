PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
cd ${PROJECT_DIR}

python merge_shards.py \
  --dir "$DATASET_DIR/outputs_yulin/openPangu-Embedded-7B-V1.1/Math_Math"\
  --base 'origin_temp0.7_maxlen16000'
