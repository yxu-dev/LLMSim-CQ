#!/bin/bash

# CQ 量化测试 - 带时间测量和详细日志
# 关键：必须确保 use_cache=True，否则 CQ 不会生效！

echo "================================================"
echo "开始测试 CQ 量化版本"
echo "================================================"
START_TIME=$(date +%s)

python -m lm_eval.run_models --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B,cq_codebook_dir=/home/zz359/workspace-vq/fisher_weighted_codebook/llama-3.1-8b/4c8b,attn_implementation=eager \
    --tasks winogrande \
    --batch_size 1 \
    --device cuda:1 \
    --verbosity INFO \
    --output_path results/llama-3.1-8b/cq_4c8b_winogrande_timed.json 2>&1 | tee cq_test_log.txt

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "================================================"
echo "CQ 测试完成！"
echo "总耗时: ${ELAPSED} 秒 ($(($ELAPSED / 60)) 分钟)"
echo "================================================"
echo "请检查日志中是否有 'Enabled CQ KV-cache quantization' 信息"
echo "如果没有，说明 CQ 未启用！"
echo "================================================"




