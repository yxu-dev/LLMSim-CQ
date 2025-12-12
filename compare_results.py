#!/usr/bin/env python3
"""比较 CQ 和 FP 基线的评测结果"""

import json
import sys
from pathlib import Path

def load_results(filepath):
    """加载评测结果 JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)

def compare_results(cq_file, fp_file):
    """比较两个结果文件"""
    try:
        cq_results = load_results(cq_file)
        fp_results = load_results(fp_file)
    except FileNotFoundError as e:
        print(f"❌ 错误：找不到结果文件 - {e}")
        return

    print("=" * 60)
    print("📊 CQ vs FP 基线对比结果")
    print("=" * 60)
    
    # 提取 winogrande 结果
    cq_acc = cq_results.get('results', {}).get('winogrande', {}).get('acc,none', 'N/A')
    fp_acc = fp_results.get('results', {}).get('winogrande', {}).get('acc,none', 'N/A')
    
    print(f"\n📈 准确率 (Accuracy):")
    print(f"  CQ 量化:  {cq_acc:.4f}" if isinstance(cq_acc, float) else f"  CQ 量化:  {cq_acc}")
    print(f"  FP 基线:  {fp_acc:.4f}" if isinstance(fp_acc, float) else f"  FP 基线:  {fp_acc}")
    
    if isinstance(cq_acc, float) and isinstance(fp_acc, float):
        diff = cq_acc - fp_acc
        diff_percent = (diff / fp_acc) * 100
        print(f"\n📉 精度损失:")
        print(f"  绝对差异:  {diff:+.4f}")
        print(f"  相对差异:  {diff_percent:+.2f}%")
        
        if abs(diff) < 0.0001:
            print("\n⚠️  警告：精度完全相同，可能 CQ 量化未生效！")
            print("   请检查日志中是否有 'Enabled CQ KV-cache quantization' 信息")
    
    # 提取配置信息
    print(f"\n⚙️  配置信息:")
    cq_config = cq_results.get('config', {})
    print(f"  CQ 模型: {cq_config.get('model_args', 'N/A')}")
    
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        cq_file = sys.argv[1]
        fp_file = sys.argv[2]
    else:
        # 默认路径
        cq_file = "results/llama-3.1-8b/cq_4c8b_winogrande_timed.json"
        fp_file = "results/llama-3.1-8b/fp_winogrande_timed.json"
    
    print(f"📁 CQ 结果文件: {cq_file}")
    print(f"📁 FP 结果文件: {fp_file}\n")
    
    compare_results(cq_file, fp_file)




