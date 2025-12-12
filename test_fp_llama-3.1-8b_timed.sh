#!/bin/bash

# FP 基线测试 - 带时间测量
# 不使用任何量化

echo "================================================"
echo "开始测试 FP 基线版本（无量化）"
echo "================================================"
START_TIME=$(date +%s)

python -m lm_eval.run_models --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B,attn_implementation=eager \
    --tasks winogrande \
    --batch_size 1 \
    --device cuda:1 \
    --verbosity INFO \
    --output_path results/llama-3.1-8b/fp_winogrande_timed.json 2>&1 | tee fp_test_log.txt

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "================================================"
echo "FP 测试完成！"
echo "总耗时: ${ELAPSED} 秒 ($(($ELAPSED / 60)) 分钟)"
echo "================================================"




