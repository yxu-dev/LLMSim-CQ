#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/zz359/workspace-CQ-zzy/LLMSim-CQ-zzy"
PY_SCRIPT="${PROJECT_ROOT}/visualization/fisher_loss_visualization.py"
OUTPUT_DIR="${PROJECT_ROOT}/visualization/fisher-loss"

# 你可以在命令前覆盖这些变量：
#   CUDA_VISIBLE_DEVICES=0 bash visualization/run_fisher_loss_visualization.sh
#   LAYER_IDX=8 MAX_ITER=80 bash visualization/run_fisher_loss_visualization.sh
LAYER_IDX="${LAYER_IDX:-16}"
MAX_ITER="${MAX_ITER:-60}"
MAX_GROUPS="${MAX_GROUPS:-64}"
ROWS_PER_SAMPLE="${ROWS_PER_SAMPLE:-512}"
MAX_SAMPLES="${MAX_SAMPLES:-16}"
NUM_BITS="${NUM_BITS:-8}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
# 默认使用 GPU 7；可在命令前通过 CUDA_VISIBLE_DEVICES 覆盖
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export CUDA_VISIBLE_DEVICES
# 输出文件名前缀（按你的命名规范）
# 示例: llama-3.1-8b_2c8b / llama-3.1-8b_4c8b / llama-3.1-8b_8c8b
# 默认占位: llama-3.1-8b_ncnb
FILE_BASENAME="${FILE_BASENAME:-llama-3.1-8b_ncnb}"

mkdir -p "${OUTPUT_DIR}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${OUTPUT_DIR}/${FILE_BASENAME}_fisher_loss_run_layer${LAYER_IDX}_${TIMESTAMP}.log"
GPU_INFO_FILE="${OUTPUT_DIR}/${FILE_BASENAME}_gpu_info_${TIMESTAMP}.txt"

{
  echo "=========================================================="
  echo "Fisher Loss Visualization Runner"
  echo "Project Root : ${PROJECT_ROOT}"
  echo "Python Script: ${PY_SCRIPT}"
  echo "Output Dir   : ${OUTPUT_DIR}"
  echo "Run Time     : $(date '+%F %T')"
  echo "Layer Index  : ${LAYER_IDX}"
  echo "Max Iter     : ${MAX_ITER}"
  echo "Max Groups   : ${MAX_GROUPS}"
  echo "Rows/Sample  : ${ROWS_PER_SAMPLE}"
  echo "Max Samples  : ${MAX_SAMPLES}"
  echo "Num Bits     : ${NUM_BITS}"
  echo "Seed         : ${SEED}"
  echo "Device       : ${DEVICE}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "=========================================================="
} | tee "${LOG_FILE}"

if command -v nvidia-smi >/dev/null 2>&1; then
  {
    echo "========== GPU INFO =========="
    nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader
    echo "=============================="
  } | tee "${GPU_INFO_FILE}" | tee -a "${LOG_FILE}"
else
  echo "[Warn] nvidia-smi not found, cannot query GPU info." | tee -a "${LOG_FILE}"
fi

echo "[Run] Start plotting Fisher loss curves..." | tee -a "${LOG_FILE}"
python -u "${PY_SCRIPT}" \
  --layer-idx "${LAYER_IDX}" \
  --max-iter "${MAX_ITER}" \
  --max-groups "${MAX_GROUPS}" \
  --rows-per-sample "${ROWS_PER_SAMPLE}" \
  --max-samples "${MAX_SAMPLES}" \
  --num-bits "${NUM_BITS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --output-dir "${OUTPUT_DIR}" \
  --no-show 2>&1 | tee -a "${LOG_FILE}"

# 将 python 默认产物重命名为指定前缀格式
RAW_PNG="${OUTPUT_DIR}/fisher_weighted_loss_curve_layer${LAYER_IDX}.png"
RAW_JSON="${OUTPUT_DIR}/fisher_weighted_loss_curve_layer${LAYER_IDX}.json"
FINAL_PNG="${OUTPUT_DIR}/${FILE_BASENAME}_fisher_weighted_loss_curve_layer${LAYER_IDX}.png"
FINAL_JSON="${OUTPUT_DIR}/${FILE_BASENAME}_fisher_weighted_loss_curve_layer${LAYER_IDX}.json"

if [[ -f "${RAW_PNG}" ]]; then
  mv -f "${RAW_PNG}" "${FINAL_PNG}"
  echo "[Done] Renamed plot to: ${FINAL_PNG}" | tee -a "${LOG_FILE}"
fi
if [[ -f "${RAW_JSON}" ]]; then
  mv -f "${RAW_JSON}" "${FINAL_JSON}"
  echo "[Done] Renamed json to: ${FINAL_JSON}" | tee -a "${LOG_FILE}"
fi

echo "[Done] $(date '+%F %T')" | tee -a "${LOG_FILE}"
echo "[Done] Log saved to: ${LOG_FILE}"
echo "[Done] GPU info saved to: ${GPU_INFO_FILE}"
echo "[Done] Plots/json saved under: ${OUTPUT_DIR}"
