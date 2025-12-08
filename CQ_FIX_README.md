# CQ 量化精度问题修复说明

## 问题诊断

### 根本原因
量化后精度没有变化的原因是：**在 loglikelihood 评估时，模型调用没有启用 KV cache，导致 CQ 量化完全没有生效**。

### 详细分析

1. **CQ 量化的工作机制**（`lm_eval/quantization/cq_cache.py`）：
   - `enable_cq_kv_cache()` 会 monkey-patch 模型的 `forward` 方法
   - Patched forward 只在满足以下条件时创建量化 cache：
     ```python
     if use_cache and past_key_values is None:
         kwargs["past_key_values"] = self._cq_cache_factory()  # 创建量化 cache
     ```

2. **问题所在**（`lm_eval/models/huggingface.py` 第 993 行，修复前）：
   ```python
   def _model_call(self, inps, attn_mask=None, labels=None):
       # ... 省略 ...
       return self.model(inps).logits  # ❌ 没有传递 use_cache=True
   ```
   
   - `_model_call` 是 `_loglikelihood_tokens` 调用模型的核心方法
   - 它没有传递 `use_cache=True` 参数
   - 导致 patched forward 中的条件不满足，不会创建量化 cache
   - 结果：**模型使用的是原始 FP16/BF16 KV cache，量化完全没有生效**

3. **为什么 `_model_generate` 没有这个问题**：
   - 第 1029 行明确传递了 `use_cache=True`
   - 所以在 generation 任务中，CQ 量化是正常工作的
   - 但 loglikelihood 评估（最常用的评测方式）完全不受影响

## 修复方案

### 修改文件：`lm_eval/models/huggingface.py`

**修改位置**：第 993 行

**修改前**：
```python
return self.model(inps).logits
```

**修改后**：
```python
# Enable cache for CQ quantization to take effect
return self.model(inps, use_cache=True).logits
```

### 修复效果

修复后，在 loglikelihood 评估中：
1. 每次调用 `_model_call` 时会传递 `use_cache=True`
2. Patched forward 检测到 `use_cache=True` 和 `past_key_values=None`
3. 创建新的 `QuantizedDynamicCache` 实例
4. KV cache 被量化存储和反量化使用
5. **精度会相应下降**，反映真实的量化效果

### 为什么每个 batch 创建新 cache 是正确的

在 `_loglikelihood_tokens` 中：
- 每个 chunk/batch 包含不同的输入序列
- 序列长度各不相同
- 它们之间不应该共享 KV cache
- 每个 batch 独立评估是正确的行为

## 验证方法

### 方法 1：运行测试脚本
```bash
cd /home/zz359/workspace-vq/LLMSim-CQ
python test_cq_fix.py
```

预期输出：
- 应该显示 "✓ GOOD: Significant difference detected - CQ quantization is working!"
- Max difference 应该 > 1e-3

### 方法 2：对比 baseline 和 CQ 的评测结果

**Baseline（无量化）**：
```bash
python -m lm_eval.run_models --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks winogrande \
    --batch_size 1 \
    --device cuda:0 \
    --limit 10
```

**CQ 量化**：
```bash
python -m lm_eval.run_models --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,cq_codebook_dir=/home/zz359/workspace-vq/fisher_weighted_codebook/llama-3.1-8b/4c8b \
    --tasks winogrande \
    --batch_size 1 \
    --device cuda:0 \
    --limit 10
```

**预期结果**：
- CQ 版本的精度应该略低于 baseline（取决于量化配置，通常下降 1-3%）
- 如果两者精度完全相同，说明量化没有生效

### 方法 3：检查 cache 类型

在代码中添加调试输出：
```python
# 在 lm_eval/quantization/cq_cache.py 的 patched_forward 中
def patched_forward(self, *args, **kwargs):
    use_cache = kwargs.get("use_cache", None)
    past_key_values = kwargs.get("past_key_values", None)
    if getattr(self, "_cq_enabled", False):
        should_use_cache = use_cache if use_cache is not None else True
        if should_use_cache and past_key_values is None:
            kwargs["past_key_values"] = self._cq_cache_factory()
            print(f"[DEBUG] Created QuantizedDynamicCache")  # 添加这行
            kwargs.setdefault("use_cache", True)
    return original_forward(*args, **kwargs)
```

运行后应该看到多次 `[DEBUG] Created QuantizedDynamicCache` 输出。

## 其他相关文件

以下文件在本次修复中被检查但不需要修改：

1. **`lm_eval/quantization/cq_cache.py`**：
   - CQ 量化核心逻辑正确
   - `patched_forward` 逻辑正确
   - 不需要修改

2. **`scripts/run_cq_eval.py`**：
   - 用于 perplexity 评估
   - 第 77 行已经正确使用 `use_cache=True`
   - 不需要修改

3. **测试脚本**（`test_cq_llama-3.1-8b.sh` 等）：
   - 配置正确
   - 只需要确保 codebook 路径存在

## 注意事项

1. **Seq2Seq 模型**：本次修复仅针对 causal LM（如 Llama）。Seq2Seq 模型的 `_model_call` 分支（第 985-987 行）可能需要类似修复。

2. **性能影响**：启用 cache 后，内存使用会增加（存储量化索引），但相比 FP16 cache 仍然节省大量内存。

3. **兼容性**：此修复不影响未启用 CQ 的模型，因为 `use_cache=True` 只是启用标准 Transformers cache。

## 总结

这是一个**单行修改**解决的关键 bug：
- **修改前**：量化 codebook 被正确加载，但从未实际使用
- **修改后**：量化在 loglikelihood 评估中正常工作
- **影响**：所有使用 `lm_eval` 进行的 CQ 量化评测结果

修复后，用户应该能观察到量化对精度的真实影响。







