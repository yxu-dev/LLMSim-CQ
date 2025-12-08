#!/usr/bin/env python
"""测试优化前后的量化实现是否等价"""

import torch
import numpy as np

# 加载实际的 codebook
codebook_path = "/home/zz359/workspace-vq/fisher_weighted_codebook/llama-3.1-8b/4c8b/k_centroids_fisher_layer0.npy"
centroids = torch.from_numpy(np.load(codebook_path)).to(torch.float32)

print(f"Codebook shape: {centroids.shape}")
num_groups, num_centroids, group_channels = centroids.shape

# 创建测试数据
num_tokens = 10
total_channels = num_groups * group_channels
test_tensor = torch.randn(num_tokens, total_channels)

print(f"Test tensor shape: {test_tensor.shape}")

# 原始实现（带 for 循环）
def quantize_cq_original(input_tensor, centroids):
    num_tokens, total_channels = input_tensor.shape
    num_groups, num_centroids, group_channels = centroids.shape
    
    reshaped = input_tensor.view(num_tokens, num_groups, group_channels)
    quantized = torch.empty((num_tokens, num_groups), dtype=torch.uint8, device=input_tensor.device)
    
    for group_idx in range(num_groups):
        group_data = reshaped[:, group_idx, :].to(torch.float32)
        group_centroids = centroids[group_idx].to(input_tensor.device)
        distances = torch.cdist(group_data, group_centroids, p=2.0)
        quantized[:, group_idx] = torch.argmin(distances, dim=1).to(torch.uint8)
    
    return quantized

# 优化实现（向量化）
def quantize_cq_optimized(input_tensor, centroids):
    num_tokens, total_channels = input_tensor.shape
    num_groups, num_centroids, group_channels = centroids.shape
    
    reshaped = input_tensor.view(num_tokens, num_groups, group_channels).float()
    centroids = centroids.to(input_tensor.device)
    
    reshaped_expanded = reshaped.unsqueeze(2)
    centroids_expanded = centroids.unsqueeze(0)
    
    distances = torch.sum((reshaped_expanded - centroids_expanded) ** 2, dim=-1)
    quantized = torch.argmin(distances, dim=-1).to(torch.uint8)
    
    return quantized

print("\n测试原始实现...")
import time
start = time.time()
result_original = quantize_cq_original(test_tensor, centroids)
time_original = time.time() - start
print(f"原始实现时间: {time_original:.4f}秒")
print(f"结果 shape: {result_original.shape}")
print(f"前5个 tokens 的前10个 groups 的索引:\n{result_original[:5, :10]}")

print("\n测试优化实现...")
start = time.time()
result_optimized = quantize_cq_optimized(test_tensor, centroids)
time_optimized = time.time() - start
print(f"优化实现时间: {time_optimized:.4f}秒")
print(f"结果 shape: {result_optimized.shape}")
print(f"前5个 tokens 的前10个 groups 的索引:\n{result_optimized[:5, :10]}")

print(f"\n速度提升: {time_original/time_optimized:.2f}x")

# 检查结果是否相同
if torch.allclose(result_original.float(), result_optimized.float()):
    print("✅ 两种实现的结果完全相同！")
else:
    diff = (result_original != result_optimized).sum().item()
    total = result_original.numel()
    print(f"❌ 结果不同！差异: {diff}/{total} ({diff/total*100:.2f}%)")
    
    # 找出第一个不同的位置
    first_diff = torch.where(result_original != result_optimized)
    if len(first_diff[0]) > 0:
        token_idx = first_diff[0][0].item()
        group_idx = first_diff[1][0].item()
        print(f"\n第一个差异位置: token {token_idx}, group {group_idx}")
        print(f"  原始: {result_original[token_idx, group_idx]}")
        print(f"  优化: {result_optimized[token_idx, group_idx]}")
        
        # 检查该 group 的距离计算
        group_data = test_tensor[token_idx, group_idx*group_channels:(group_idx+1)*group_channels]
        group_centroids = centroids[group_idx]
        
        # 原始方法的距离
        dist_original = torch.cdist(group_data.unsqueeze(0).float(), group_centroids, p=2.0)
        # 优化方法的距离
        dist_optimized = torch.sum((group_data.float() - group_centroids) ** 2, dim=-1).sqrt()
        
        print(f"\n距离对比 (前10个 centroids):")
        print(f"  原始: {dist_original[0, :10]}")
        print(f"  优化: {dist_optimized[:10]}")
        print(f"  差异: {(dist_original[0, :10] - dist_optimized[:10]).abs().max():.6f}")





