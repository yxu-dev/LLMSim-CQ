#!/bin/bash

# FP 基线测试 - 优化版本（更大的 batch size）

echo "================================================"
echo "开始测试 FP 基线版本（优化配置）"
echo "================================================"
START_TIME=$(date +%s)

python -m lm_eval.run_models --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B,attn_implementation=eager \
    --tasks winogrande \
    --batch_size auto \
    --device cuda:5 \
    --verbosity INFO \
    --output_path results/llama-3.1-8b/fp_winogrande_optimized.json 2>&1 | tee fp_test_optimized_log.txt

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "================================================"
echo "FP 测试完成（优化版）！"
echo "总耗时: ${ELAPSED} 秒 ($(($ELAPSED / 60)) 分钟)"
echo "================================================"




