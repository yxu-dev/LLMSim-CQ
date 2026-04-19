#!/bin/bash

# CQ 量化测试 - 优化版本（更大的 batch size）
# 目标：提升 GPU 利用率，减少开销占比

echo "================================================"
echo "开始测试 CQ 量化版本（优化配置）"
echo "================================================"
START_TIME=$(date +%s)

python -m lm_eval.run_models --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B,cq_codebook_dir=/home/zz359/workspace-CQ-zzy/LLMSim-CQ-zzy/output/llama-3.1-8b-2c4b/centroids,attn_implementation=eager \
    --tasks winogrande \
    --batch_size auto \
    --device cuda:7 \
    --verbosity INFO \
    --output_path results/llama-3.1-8b/cq_4c4b_winogrande_optimized.json 2>&1 | tee cq_test_optimized_log.txt

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "================================================"
echo "CQ 测试完成（优化版）！"
echo "总耗时: ${ELAPSED} 秒 ($(($ELAPSED / 60)) 分钟)"
echo "================================================"
echo "请检查日志中是否有 'Enabled CQ KV-cache quantization' 信息"
echo "================================================"




