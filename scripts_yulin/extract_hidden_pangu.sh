npu-smi info
pip install --upgrade pip
pip install transformers==4.53.2
pip list
printenv

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH="/usr/local/Ascend/ascend-toolkit/latest"
export ASCEND_RT_VISIBLE_DEVICES=0

PROJECT_DIR=$(dirname "$(realpath "$0")")
cd ${PROJECT_DIR}

python transformer_inference_dp.py \
  --model_name_or_path '/models/openPangu-Embedded-7B-V1.1' \
  --dataset_dir "./Data" \
  --dataset "Math_Olympiad" \
  --output_path "./outputs" \
  --max_generated_tokens 10000 \
  --num_npus 8 \
  --trust_remote_code
