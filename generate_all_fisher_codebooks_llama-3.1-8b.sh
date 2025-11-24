#!/bin/bash

# --- 配置 ---
# 1. 立即停止：如果任何命令失败，脚本将立即退出
set -e

# 2. 指定要使用的 GPU
export CUDA_VISIBLE_DEVICES=7

# 3. 配置 Python 脚本和路径
# 确保这个脚本与您的 .py 脚本在同一目录，或者相应地修改路径
PYTHON_SCRIPT="run_weighted_kmeans.py" 

FISHER_PATH="/home/yx277/workspace-vq/fisher_runs_output/run_llama-3.1-8b-4c8b/fisher_diag.pt"
DATA_PATH="/home/yx277/workspace-vq/LLMSim-CQ/output/kv-simi-llama-3.1-8b-wikitext"
OUTPUT_DIR="/home/yx277/workspace-vq/LLMSim-CQ/fisher_weighted_codebook/llama-3.1-8b/4c8b"

# --- 执行 ---

# Llama 3.1 8B 有 32 层 (索引 0-31)
TOTAL_LAYERS=32

echo "--- 开始生成 Fisher 加权 K-Means 码本 ---"
echo "模型: Llama 3.1 8B"
echo "总层数: $TOTAL_LAYERS (索引 0-31)"
echo "Fisher 路径: $FISHER_PATH"
echo "数据路径: $DATA_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"
echo "------------------------------------------------------------"

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 循环遍历所有层，从 0 到 31
for (( i=0; i<$TOTAL_LAYERS; i++ ))
do
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] ==> 开始处理第 $i 层..."
  
  python "$PYTHON_SCRIPT" \
    --fisher_path "$FISHER_PATH" \
    --data_path "$DATA_PATH" \
    --layer_idx $i \
    --output_dir "$OUTPUT_DIR"
    
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] ==> 第 $i 层处理完毕。码本已保存。"
  echo "------------------------------------------------------------"
done

echo "[$(date +'%Y-%m-%d %H:%M:%S')] 所有层 (0-31) 均已处理完毕。"
echo "码本生成完成。请检查: $OUTPUT_DIR"