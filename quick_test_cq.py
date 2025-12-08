#!/usr/bin/env python
"""快速测试 CQ 量化是否生效（无需加载完整模型）"""

import torch
import sys

print("=" * 60)
print("快速测试：检查 CQ 量化修复是否生效")
print("=" * 60)
print()

# 测试 1: 检查修改是否存在
print("✓ 测试 1: 检查代码修改...")
with open('lm_eval/models/huggingface.py', 'r') as f:
    content = f.read()
    if 'use_cache=True' in content and '# Enable cache for CQ quantization' in content:
        print("  ✅ 代码修改已存在")
    else:
        print("  ❌ 代码修改未找到！")
        sys.exit(1)

# 测试 2: 检查 CQ 模块是否可导入
print("\n✓ 测试 2: 检查 CQ 量化模块...")
try:
    from lm_eval.quantization.cq_cache import CQQuantizationConfig, enable_cq_kv_cache
    print("  ✅ CQ 量化模块导入成功")
except ImportError as e:
    print(f"  ❌ CQ 量化模块导入失败: {e}")
    sys.exit(1)

# 测试 3: 检查 codebook 文件
print("\n✓ 测试 3: 检查 codebook 文件...")
import os
codebook_dir = "/home/zz359/workspace-vq/fisher_weighted_codebook/llama-3.1-8b/4c8b"
if os.path.exists(codebook_dir):
    files = [f for f in os.listdir(codebook_dir) if f.endswith('.npy')]
    k_files = [f for f in files if f.startswith('k_centroids')]
    v_files = [f for f in files if f.startswith('v_centroids')]
    print(f"  ✅ Codebook 目录存在")
    print(f"  ✅ 找到 {len(k_files)} 个 K codebook 文件")
    print(f"  ✅ 找到 {len(v_files)} 个 V codebook 文件")
else:
    print(f"  ❌ Codebook 目录不存在: {codebook_dir}")
    sys.exit(1)

# 测试 4: 测试 CodebookManager
print("\n✓ 测试 4: 测试 CodebookManager...")
try:
    from lm_eval.quantization.cq_cache import CodebookManager
    manager = CodebookManager(codebook_dir, layer_prefix="layer", n_layers=2)  # 只加载2层测试
    print(f"  ✅ CodebookManager 初始化成功")
    print(f"  ✅ 加载了 {manager.num_layers} 层 codebook")
    
    # 测试获取 codebook
    k_cb, v_cb = manager.get_codebooks(0)
    print(f"  ✅ Layer 0 K codebook shape: {k_cb.shape}")
    print(f"  ✅ Layer 0 V codebook shape: {v_cb.shape}")
except Exception as e:
    print(f"  ❌ CodebookManager 测试失败: {e}")
    sys.exit(1)

# 测试 5: 测试量化/反量化函数
print("\n✓ 测试 5: 测试量化/反量化函数...")
try:
    from lm_eval.quantization.cq_cache import quantize_cq, dequantize_cq
    
    # 创建测试数据
    num_tokens = 10
    num_groups, num_centroids, group_channels = k_cb.shape
    total_channels = num_groups * group_channels
    
    test_tensor = torch.randn(num_tokens, total_channels)
    
    # 量化
    indices = quantize_cq(test_tensor, k_cb)
    print(f"  ✅ 量化成功: {test_tensor.shape} -> {indices.shape}")
    
    # 反量化
    reconstructed = dequantize_cq(indices, k_cb)
    print(f"  ✅ 反量化成功: {indices.shape} -> {reconstructed.shape}")
    
    # 检查维度
    assert test_tensor.shape == reconstructed.shape
    print(f"  ✅ 维度匹配")
    
    # 计算重建误差
    mse = torch.mean((test_tensor - reconstructed) ** 2).item()
    print(f"  ✅ 重建误差 (MSE): {mse:.6f}")
    
except Exception as e:
    print(f"  ❌ 量化/反量化测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 所有测试通过！CQ 量化基础功能正常")
print("=" * 60)
print("\n下一步: 运行完整的模型测试来验证实际效果")
print("  运行: python test_cq_fix.py")
print("  或: ./verify_cq_fix.sh")






