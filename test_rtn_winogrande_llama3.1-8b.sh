#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

# RTN (per-tensor) Winogrande evaluation entry.
# NOTE:
# - Dataset loading is handled internally by lm_eval when --tasks winogrande is set.
# - This script standardizes output/log paths under result/llama-3.1-8b.
# - If RTN integration into lm_eval/hf model path is not wired yet, this will behave like baseline.

echo "================================================"
echo "开始测试 RTN per-tensor (Winogrande)"
echo "================================================"

START_TIME=$(date +%s)
RESULT_DIR="${RESULT_DIR:-result/llama-3.1-8b}"
mkdir -p "${RESULT_DIR}"

python -m lm_eval.run_models --model hf \
  --model_args pretrained=meta-llama/Llama-3.1-8B,attn_implementation=eager,rtn_pertensor_bits=4 \
  --tasks winogrande \
  --batch_size auto \
  --device cuda:3 \
  --verbosity INFO \
  --output_path "${RESULT_DIR}/rtn_pertensor_winogrande.json" 2>&1 | tee "${RESULT_DIR}/rtn_pertensor_winogrande.log"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "================================================"
echo "RTN per-tensor 测试完成"
echo "总耗时: ${ELAPSED} 秒 ($(($ELAPSED / 60)) 分钟)"
echo "结果文件: ${RESULT_DIR}/rtn_pertensor_winogrande.json"
echo "日志文件: ${RESULT_DIR}/rtn_pertensor_winogrande.log"
echo "================================================"
