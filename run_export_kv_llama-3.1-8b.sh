#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=6
set -euo pipefail


# 说明：
# - 修改 MODEL_PATH 为本地模型路径或 HuggingFace 仓库名
# - 修改 CALIB_TXT 为校准文本文件（每行一条样本）
# - OUTPUT_DIR 为 KV 导出保存位置

MODEL_PATH="meta-llama/Meta-Llama-3.1-8B"
OUTPUT_DIR="/home/zz359/workspace-vq/LLMSim-CQ/output/kv/llama-3.1-8b"

python /home/zz359/workspace-vq/LLMSim-CQ/export_KV.py \
  --model "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_seq_len 4096 \
  --batch_size 1 \
  --dtype bfloat16 \
  --long_seq_threshold 2048 \
  --verbose

