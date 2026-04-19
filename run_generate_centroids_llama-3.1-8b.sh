#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=7

set -euo pipefail

# fisher_diag.pt 和 kv_cache/
DATA_ROOT="/home/zz359/workspace-CQ-zzy/LLMSim-CQ-zzy/output/llama-3.1-8b-2c4b"

# 质心保存目录
OUTPUT_DIR="${DATA_ROOT}/centroids"

# 模型总层数 (Llama-3.1-8B 为 32 层)
NUM_LAYERS=32

echo "=========================================================="
echo "Starting Weighted K-Means Clustering for ${NUM_LAYERS} layers..."
echo "Data Source: ${DATA_ROOT}"
echo "Output Dir:  ${OUTPUT_DIR}"
echo "=========================================================="

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 循环从 0 到 31
for (( i=0; i<NUM_LAYERS; i++ )); do
    echo ""
    echo ">>> [Layer $i / $((NUM_LAYERS-1))] Processing..."
    
    python "generate_centroids.py" \
      --data_path "${DATA_ROOT}" \
      --layer_idx "$i" \
      --output_dir "${OUTPUT_DIR}"
      
    echo ">>> Layer $i Done."
done

echo ""
echo "=========================================================="
echo "All ${NUM_LAYERS} layers finished!"
echo "Centroids saved to: ${OUTPUT_DIR}"
echo "=========================================================="