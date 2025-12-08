#!/usr/bin/env python
"""Quick test to verify CQ quantization is actually being applied."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.quantization.cq_cache import CQQuantizationConfig, enable_cq_kv_cache

# Load model
model_id = "meta-llama/Llama-3.1-8B-Instruct"
print(f"Loading model: {model_id}")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.eval()

# Enable CQ
codebook_dir = "/home/zz359/workspace-vq/fisher_weighted_codebook/llama-3.1-8b/4c8b"
print(f"Enabling CQ with codebook: {codebook_dir}")
cq_config = CQQuantizationConfig(
    codebook_dir=codebook_dir,
    layer_prefix="layer",
    num_layers=model.config.num_hidden_layers,
)
enable_cq_kv_cache(model, cq_config)

# Test input
text = "The quick brown fox jumps over the lazy dog. " * 10
inputs = tokenizer(text, return_tensors="pt").to(model.device)

print("\n=== Test 1: Without use_cache (old behavior) ===")
with torch.no_grad():
    outputs1 = model(**inputs)
    logits1 = outputs1.logits
    print(f"Logits shape: {logits1.shape}")
    print(f"First token top-5 logits: {logits1[0, 0, :5]}")

print("\n=== Test 2: With use_cache=True (new behavior) ===")
with torch.no_grad():
    outputs2 = model(**inputs, use_cache=True)
    logits2 = outputs2.logits
    print(f"Logits shape: {logits2.shape}")
    print(f"First token top-5 logits: {logits2[0, 0, :5]}")
    print(f"Has past_key_values: {outputs2.past_key_values is not None}")
    if outputs2.past_key_values is not None:
        print(f"Cache type: {type(outputs2.past_key_values)}")

print("\n=== Comparing outputs ===")
max_diff = torch.max(torch.abs(logits1 - logits2)).item()
mean_diff = torch.mean(torch.abs(logits1 - logits2)).item()
print(f"Max difference: {max_diff}")
print(f"Mean difference: {mean_diff}")

if max_diff > 1e-3:
    print("✓ GOOD: Significant difference detected - CQ quantization is working!")
else:
    print("✗ BAD: No significant difference - CQ might not be active")






