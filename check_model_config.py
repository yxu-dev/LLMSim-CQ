#!/usr/bin/env python
"""检查 Llama 3.1 8B 的实际配置"""

from transformers import AutoConfig

model_id = "meta-llama/Llama-3.1-8B-Instruct"
config = AutoConfig.from_pretrained(model_id)

print(f"模型: {model_id}")
print(f"\n基本配置:")
print(f"  hidden_size: {config.hidden_size}")
print(f"  num_attention_heads: {config.num_attention_heads}")
print(f"  num_key_value_heads: {getattr(config, 'num_key_value_heads', 'N/A')}")
print(f"  num_hidden_layers: {config.num_hidden_layers}")

if hasattr(config, 'num_key_value_heads'):
    kv_heads = config.num_key_value_heads
    q_heads = config.num_attention_heads
    head_dim = config.hidden_size // q_heads
    kv_dim = kv_heads * head_dim
    print(f"\nKV Cache 维度:")
    print(f"  Q heads: {q_heads}")
    print(f"  KV heads: {kv_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  KV projection size: {kv_dim}")
    print(f"\n这是 GQA (Grouped Query Attention)")
else:
    print(f"\n标准 Multi-Head Attention")
    print(f"  KV projection size: {config.hidden_size}")

# 检查 codebook
import numpy as np
codebook_path = "/home/zz359/workspace-vq/fisher_weighted_codebook/llama-3.1-8b/4c8b/k_centroids_fisher_layer0.npy"
k_cb = np.load(codebook_path)
print(f"\nCodebook shape: {k_cb.shape}")
print(f"  num_groups: {k_cb.shape[0]}")
print(f"  num_centroids: {k_cb.shape[1]}")
print(f"  group_channels: {k_cb.shape[2]}")
print(f"  total_channels: {k_cb.shape[0] * k_cb.shape[2]}")

if hasattr(config, 'num_key_value_heads'):
    expected_dim = config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)
    actual_dim = k_cb.shape[0] * k_cb.shape[2]
    print(f"\n维度匹配检查:")
    print(f"  期望的 KV 维度: {expected_dim}")
    print(f"  Codebook 维度: {actual_dim}")
    if expected_dim == actual_dim:
        print(f"  ✅ 维度匹配！")
    else:
        print(f"  ❌ 维度不匹配！")







