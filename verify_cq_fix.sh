#!/bin/bash
# 快速验证 CQ 量化修复是否生效

set -e

echo "=========================================="
echo "CQ 量化修复验证脚本"
echo "=========================================="
echo ""

# 配置
MODEL="meta-llama/Llama-3.1-8B-Instruct"
CODEBOOK_DIR="/home/zz359/workspace-vq/fisher_weighted_codebook/llama-3.1-8b/4c8b"
TASK="winogrande"
DEVICE="cuda:5"
LIMIT=20  # 使用少量样本快速测试

echo "模型: $MODEL"
echo "任务: $TASK"
echo "样本数: $LIMIT"
echo "设备: $DEVICE"
echo ""

# 检查 codebook 是否存在
if [ ! -d "$CODEBOOK_DIR" ]; then
    echo "❌ 错误: Codebook 目录不存在: $CODEBOOK_DIR"
    exit 1
fi

echo "✓ Codebook 目录存在"
echo ""

# Baseline 测试（无量化）
echo "=========================================="
echo "步骤 1/2: 运行 Baseline（无量化）"
echo "=========================================="
python -m lm_eval.run_models --model hf \
    --model_args pretrained=$MODEL \
    --tasks $TASK \
    --batch_size 1 \
    --device $DEVICE \
    --limit $LIMIT \
    --output_path results/baseline_temp.json

BASELINE_ACC=$(python -c "import json; data=json.load(open('results/baseline_temp.json')); print(data['results']['$TASK']['acc,none'])")
echo ""
echo "Baseline 精度: $BASELINE_ACC"
echo ""

# CQ 测试（有量化）
echo "=========================================="
echo "步骤 2/2: 运行 CQ 量化版本"
echo "=========================================="
python -m lm_eval.run_models --model hf \
    --model_args pretrained=$MODEL,cq_codebook_dir=$CODEBOOK_DIR \
    --tasks $TASK \
    --batch_size 1 \
    --device $DEVICE \
    --limit $LIMIT \
    --output_path results/cq_temp.json

CQ_ACC=$(python -c "import json; data=json.load(open('results/cq_temp.json')); print(data['results']['$TASK']['acc,none'])")
echo ""
echo "CQ 量化精度: $CQ_ACC"
echo ""

# 对比结果
echo "=========================================="
echo "结果对比"
echo "=========================================="
echo "Baseline:  $BASELINE_ACC"
echo "CQ 量化:   $CQ_ACC"

DIFF=$(python -c "print(abs($BASELINE_ACC - $CQ_ACC))")
echo "差异:      $DIFF"
echo ""

# 判断修复是否生效
if (( $(echo "$DIFF > 0.001" | bc -l) )); then
    echo "✅ 修复成功！CQ 量化正在生效（精度有差异）"
    echo ""
    echo "说明："
    echo "- Baseline 和 CQ 版本的精度有显著差异"
    echo "- 这表明量化确实在影响模型行为"
    echo "- 通常 CQ 量化会导致 1-3% 的精度下降"
else
    echo "❌ 修复可能未生效！"
    echo ""
    echo "精度差异小于 0.1%，这不太正常。可能的原因："
    echo "1. 修改的代码没有生效（检查是否重新加载了模块）"
    echo "2. Codebook 文件有问题"
    echo "3. 样本数太少导致差异不明显（尝试增加 LIMIT）"
fi

echo ""
echo "临时文件已保存至："
echo "  - results/baseline_temp.json"
echo "  - results/cq_temp.json"

