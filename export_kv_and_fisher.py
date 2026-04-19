#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from datasets import load_dataset

"""
功能：
1. 准备数据：自动下载 WikiText-2, Tokenize 并重组成 [num_samples, seq_len] 的稠密矩阵。
2. 导出 KV Cache:在 Forward 过程中导出 k_proj/v_proj 的激活值 -> 保存为 .pt
3. 计算 Fisher 信息：在 Backward 过程中计算 k_proj/v_proj 的梯度平方期望 -> 保存为 fisher_diag.pt
"""

# 1. 数据集处理 (BlockTokenDataset)
class BlockTokenDataset(Dataset):
    """
    将数据集的所有文本 Tokenize 并拼接，然后切分为固定长度的块。
    严格保证输出形状为 [num_samples, seq_len]，无 Padding。
    """
    def __init__(self, tokenizer, dataset_name, dataset_config, split="train", 
                 num_samples=16, seq_len=2048):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.num_samples = num_samples
        
        # 1. 加载 HuggingFace 数据集
        print(f"[Data] 正在加载数据集 {dataset_name} ({split})...")
        # 如果 config 为空字符串则不传
        if dataset_config:
            ds = load_dataset(dataset_name, dataset_config, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)

        # 2. Tokenize 并拼接所有文本
        # 我们需要收集足够的 token 来填满 num_samples * seq_len
        required_tokens = num_samples * seq_len
        
        all_tokens = []
        print(f"[Data] 正在 Tokenize 并拼接文本，目标 Token 数: {required_tokens}...")
        
        # 遍历数据集，直到收集够
        for row in ds:
            # WikiText 的字段名通常是 'text'，如果是其他数据集可能需要修改
            text = row.get("text", "") 
            if not text.strip(): continue
            
            # 编码文本，不截断，不padding，不加特殊符号（视模型而定，通常 Llama 需要自己管理 BOS/EOS）
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            all_tokens.extend(ids)
            
            if len(all_tokens) >= required_tokens:
                break
        
        if len(all_tokens) < required_tokens:
            raise ValueError(f"错误: 数据集太小，只有 {len(all_tokens)} tokens，不足以构建 {num_samples}x{seq_len} 的数据。")

        # 3. 截取并重塑
        # 只取前 N*L 个 token
        self.data = torch.tensor(all_tokens[:required_tokens], dtype=torch.long)
        # 重塑为 [16, 2048]
        self.data = self.data.reshape(num_samples, seq_len)
        
        print(f"[Data] 数据准备完成! 最终数据形状: {self.data.shape}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        # 构造 input_ids 和 attention_mask
        input_ids = self.data[i]
        # 因为是稠密数据，mask 全为 1
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

def shift_labels_for_ce(input_ids, ignore_index=-100):
    """
    因果语言模型计算 Loss 时，Labels 需要相对于 Input 左移一位。
    Input: [A, B, C] -> Label: [B, C, Ignore]
    """
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = ignore_index
    return labels

# 2. Forward Hook (KV Cache 导出器)
class KVExporterHook:
    def __init__(self, output_dir, key_export_domain="post_rope", verbose=False):
        self.output_dir = output_dir
        self.verbose = verbose
        self.key_export_domain = key_export_domain
        self.handles = []
        self.rotary_emb = None
        self.current_position_ids = None
        # 用于标记当前的 sample id
        self.current_sample_idx = 0 
        os.makedirs(os.path.join(output_dir, "kv_cache"), exist_ok=True)

    def _apply_rope_to_key(self, key_proj_out, num_kv_heads, head_dim):
        """
        将 k_proj 输出 [B, S, D] 转换为 RoPE 后 key，再还原回 [B, S, D]。
        """
        if self.rotary_emb is None:
            raise RuntimeError("rotary_emb is not initialized, cannot export post-RoPE keys")

        bsz, seq_len, hidden_dim = key_proj_out.shape
        if hidden_dim != num_kv_heads * head_dim:
            raise ValueError(
                f"hidden_dim mismatch for key RoPE: {hidden_dim} vs {num_kv_heads}*{head_dim}"
            )

        if self.current_position_ids is None:
            position_ids = torch.arange(seq_len, device=key_proj_out.device).unsqueeze(0).expand(bsz, -1)
        else:
            position_ids = self.current_position_ids.to(device=key_proj_out.device)

        key_states = key_proj_out.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(key_states, position_ids)
        _, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin)
        return key_states.transpose(1, 2).reshape(bsz, seq_len, hidden_dim)

    def _make_hook(self, layer_idx, kind, num_kv_heads=None, head_dim=None):
        def fwd_hook(mod, inp, out):
            # out 是 k_proj 或 v_proj 的输出 [Batch, Seq, Dim]
            # 关键：必须 detach()，否则会把整个计算图带下来导致 OOM
            # 转为 float16 或 bfloat16 以节省磁盘空间
            export_tensor = out
            if kind == "key" and self.key_export_domain == "post_rope":
                local_head_dim = head_dim
                if local_head_dim is None and self.rotary_emb is not None and hasattr(self.rotary_emb, "inv_freq"):
                    local_head_dim = int(self.rotary_emb.inv_freq.numel() * 2)
                if local_head_dim is None:
                    raise RuntimeError("Cannot infer head_dim for post-RoPE key export")

                local_num_kv_heads = num_kv_heads
                if local_num_kv_heads is None:
                    if out.shape[-1] % local_head_dim != 0:
                        raise RuntimeError(
                            f"Cannot infer num_kv_heads: out_dim={out.shape[-1]} is not divisible by head_dim={local_head_dim}"
                        )
                    local_num_kv_heads = out.shape[-1] // local_head_dim

                export_tensor = self._apply_rope_to_key(out, local_num_kv_heads, local_head_dim)
            tensor_cpu = export_tensor.detach().cpu().to(torch.float16) 
            
            # 保存文件: sample0_layer15_key.pt
            fname = f"sample{self.current_sample_idx}_layer{layer_idx}_{kind}.pt"
            save_path = os.path.join(self.output_dir, "kv_cache", fname)
            torch.save(tensor_cpu, save_path)
            
            if self.verbose and kind == 'key' and layer_idx == 0:
                print(f"  [Export] 已保存 {kind} layer {layer_idx} shape={tuple(tensor_cpu.shape)}")
        return fwd_hook

    def add(self, model):
        # 兼容 model.model (Llama) 或 model (GPT)
        core = model.model if hasattr(model, "model") else model
        self.rotary_emb = getattr(core, "rotary_emb", None)
        if self.key_export_domain == "post_rope" and self.rotary_emb is None:
            raise RuntimeError(
                "key_export_domain=post_rope requires a Llama-like core model with `rotary_emb`."
            )

        for i, lyr in enumerate(core.layers):
            attn = getattr(lyr, "self_attn", None)
            n_kv = getattr(attn, "num_key_value_heads", None)
            d_hd = getattr(attn, "head_dim", None)
            if hasattr(attn, "k_proj"):
                self.handles.append(
                    attn.k_proj.register_forward_hook(self._make_hook(i, "key", n_kv, d_hd))
                )
            if hasattr(attn, "v_proj"):
                self.handles.append(attn.v_proj.register_forward_hook(self._make_hook(i, "value")))

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []

    def step_counter(self):
        self.current_sample_idx += 1

    def set_position_ids(self, position_ids):
        self.current_position_ids = position_ids

# 3. Backward Hook (Fisher 信息累加器)
class FisherAccumulatorHook:
    def __init__(self):
        self.sum_g2 = {}       # (layer, kind) -> [Hid] (梯度平方和)
        self.token_counts = {} # (layer, kind) -> int   (计数)
        self.handles = []
        self.meta_heads = {}   # layer -> (n_kv, d_hd) (用于后续 reshape)

    def _key(self, layer, kind): return (int(layer), str(kind))

    def _make_hook(self, layer, kind):
        def bwd_hook(mod, grad_input, grad_output):
            # grad_output[0] 是输出的梯度 [Batch, Seq, Dim]
            g = grad_output[0]
            key = self._key(layer, kind)
            
            # 计算 g^2 并对 batch/seq 取平均，得到 [Dim]
            with torch.no_grad():
                g2 = (g ** 2).mean(dim=(0, 1)) 
                if key not in self.sum_g2:
                    self.sum_g2[key] = g2.clone()
                    self.token_counts[key] = 1
                else:
                    self.sum_g2[key] += g2
                    self.token_counts[key] += 1
        return bwd_hook

    def add(self, model):
        core = model.model if hasattr(model, "model") else model
        for i, lyr in enumerate(core.layers):
            attn = getattr(lyr, "self_attn", None)
            # 记录 head 信息，用于后续 reshape
            n_kv = getattr(attn, "num_key_value_heads", None)
            d_hd = getattr(attn, "head_dim", None)
            if n_kv and d_hd: self.meta_heads[i] = (n_kv, d_hd)

            if hasattr(attn, "k_proj"):
                self.handles.append(attn.k_proj.register_full_backward_hook(self._make_hook(i, "k")))
            if hasattr(attn, "v_proj"):
                self.handles.append(attn.v_proj.register_full_backward_hook(self._make_hook(i, "v")))

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []

    def finalize(self, num_coupled_channels=None):
        out = {}
        for key, s in self.sum_g2.items():
            cnt = max(1, self.token_counts[key])
            mean_g2 = s / cnt # E[g^2]
            
            # Reshape logic (适配 Coupled Quantization)
            layer, kind = key
            n_kv, d_hd = self.meta_heads.get(layer, (None, None))
            H = mean_g2.numel()

            if num_coupled_channels is not None:
                # 强制 reshape 为 [Groups, Coupled_Channels]
                # 例如 Hidden=1024, C=4 -> [256, 4]
                d_final = num_coupled_channels
                n_final = H // d_final
            else:
                n_final, d_final = n_kv, d_hd
            
            out[key] = mean_g2.view(n_final, d_final).cpu().float()
        return out

# 4. 主程序
def main():
    parser = argparse.ArgumentParser(description="导出 KV Cache 激活并计算 Fisher 信息 (Coupled Quantization)")
    
    # 模型与输出参数
    parser.add_argument("--model", type=str, required=True, help="HuggingFace 模型名称或路径")
    parser.add_argument("--output_dir", type=str, required=True, help="结果输出根目录")
    
    # 数据参数 (可修改)
    parser.add_argument("--num_samples", type=int, default=16, help="校准样本数量 (默认为 16)")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="每个样本的序列长度 (默认为 2048)")
    parser.add_argument("--dataset", type=str, default="EleutherAI/wikitext_document_level", help="数据集名称")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="数据集配置")
    
    # Fisher/CQ 参数
    parser.add_argument("--num_coupled_channels", type=int, default=4, help="耦合通道数 (C), 例如 4")
    parser.add_argument("--num_bits", type=int, default=8, help="量化比特数 (B), 例如 8")
    parser.add_argument(
        "--key_export_domain",
        type=str,
        default="post_rope",
        choices=["pre_rope", "post_rope"],
        help="导出 key 的域：pre_rope(k_proj原始输出) 或 post_rope(应用RoPE后，推荐)",
    )
    
    args = parser.parse_args()
    
    # 准备环境
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}...")
    
    # 准备 Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    
    # 准备 BlockTokenDataset (关键修改)
    # 这将自动下载数据，拼接并切分为 [16, 2048]
    ds = BlockTokenDataset(
        tokenizer=tok,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split="train", 
        num_samples=args.num_samples,
        seq_len=args.max_seq_len
    )
    
    # DataLoader: batch_size=1 是为了防止显存溢出 (Backprop 需要保存计算图)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    
    # 加载模型
    print(f"Loading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16, 
        device_map=device
    )
    # 注意：必须开启梯度计算，不能用 torch.no_grad()
    # model.eval() 仅用于关闭 Dropout 等训练行为
    model.eval()
    
    # 注册 Hooks
    exporter = KVExporterHook(
        args.output_dir,
        key_export_domain=args.key_export_domain,
        verbose=True,
    )
    accumulator = FisherAccumulatorHook()
    
    exporter.add(model)
    accumulator.add(model)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    print(f">>> 开始处理: 共 {len(ds)} 个样本 (Seq Len: {args.max_seq_len})...")
    print(">>> 同时执行: Forward (KV导出) & Backward (Fisher计算)")
    
    # 循环处理
    with torch.enable_grad(): # 确保梯度开启
        for i, batch in enumerate(dl):
            print(f"Processing sample {i+1}/{len(ds)}...")
            
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            # 与模型默认行为对齐的 position_ids（当前数据集无 padding，即 [0..S-1]）
            position_ids = attn.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attn == 0, 1)
            exporter.set_position_ids(position_ids)
            
            # Forward Pass
            # 1. 模型前向传播
            # 2. Exporter Hook 触发 -> 保存 .pt 文件到磁盘
            out = model(input_ids=input_ids, attention_mask=attn)
            
            # Loss Calculation
            # 计算标准的 Next Token Prediction Loss
            logits = out.logits
            labels = shift_labels_for_ce(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Backward Pass
            model.zero_grad()
            # 1. 反向传播
            # 2. Accumulator Hook 触发 -> 累加梯度平方
            loss.backward()
            
            # Cleanup
            exporter.step_counter()
            del out, logits, loss
            torch.cuda.empty_cache()

    # 保存 Fisher 结果
    print(">>> 正在生成 Fisher 信息矩阵...")
    fisher_data = accumulator.finalize(num_coupled_channels=args.num_coupled_channels)
    
    fisher_save_path = os.path.join(args.output_dir, "fisher_diag.pt")
    meta = {
        "model": args.model,
        "num_samples": args.num_samples,
        "seq_len": args.max_seq_len,
        "num_coupled_channels": args.num_coupled_channels,
        "num_bits": args.num_bits,
        "key_export_domain": args.key_export_domain,
    }
    torch.save({"meta": meta, "fisher": fisher_data}, fisher_save_path)
    
    # 清理 Hook
    exporter.remove()
    accumulator.remove()
    
    print(f"\n[DONE] 全部任务完成！")
    print(f"1. KV Cache 激活文件 (Raw Tensors) -> {os.path.join(args.output_dir, 'kv_cache')}")
    print(f"2. Fisher 权重文件 -> {fisher_save_path}")

if __name__ == "__main__":
    main()