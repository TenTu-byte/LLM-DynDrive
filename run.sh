# npu-smi info
pip install transformers==4.53.2
# pip list


PROJECT_DIR=$(dirname "$(realpath "$0")")
cd ${PROJECT_DIR}

python transformer_inference_dp_pangu.py \
  --model_name_or_path "$DATASET_DIR/models_yulin/openPangu-Embedded-7B-V1.1" \
  --dataset_dir "./Data" \
  --dataset "Math_Math" \
  --output_path "$DATASET_DIR/outputs_yulin" \
  --max_generated_tokens 10000 \
  --num_npus 8 \
  --trust_remote_code