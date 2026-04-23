#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=5

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

# fisher_diag.pt 和 kv_cache/
DATA_ROOT="${DATA_ROOT:-output/llama-3.1-8b-4c8b}"

# 码本位宽 b，质心数量为 2^b
NUM_BITS="${NUM_BITS:-8}"

# 质心保存目录
OUTPUT_DIR="${DATA_ROOT}/centroids"

# 模型总层数 (Llama-3.1-8B 为 32 层)
NUM_LAYERS=32

echo "=========================================================="
echo "Starting Weighted K-Means Clustering for ${NUM_LAYERS} layers..."
echo "Data Source: ${DATA_ROOT}"
echo "Output Dir:  ${OUTPUT_DIR}"
echo "Num Bits:    ${NUM_BITS}"
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
      --output_dir "${OUTPUT_DIR}" \
      --num_bits "${NUM_BITS}"
      
    echo ">>> Layer $i Done."
done

echo ""
echo "=========================================================="
echo "All ${NUM_LAYERS} layers finished!"
echo "Centroids saved to: ${OUTPUT_DIR}"
echo "=========================================================="