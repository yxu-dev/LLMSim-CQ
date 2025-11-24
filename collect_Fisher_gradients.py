# lm_eval/collect_Fisher_gradients.py
import os, json, math, argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
导出 LLaMA3.1-8B 每层 k_proj / v_proj 的激活梯度（grad_output），
并计算 Fisher 对角近似（E[g^2]）。支持 DDP 合并。
默认抓取的是 k_proj/v_proj 的“输出”（RoPE 之前）。后续量化请保持一致。
"""

# ------------------- Dataset -------------------
class LineTextDataset(Dataset):
    def __init__(self, tok, path, max_len, num_samples=None):
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if num_samples is not None:
            lines = lines[:num_samples]
        enc = tok(lines, padding=False, truncation=True, max_length=max_len, return_tensors=None)
        self.ids = enc["input_ids"]
        self.attn = enc["attention_mask"]

    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.ids[i], dtype=torch.long),
            "attention_mask": torch.tensor(self.attn[i], dtype=torch.long)
        }

def collate_pad(batch, pad_id):
    maxlen = max(x["input_ids"].shape[0] for x in batch)
    B = len(batch)
    ids = torch.full((B,maxlen), pad_id, dtype=torch.long)
    att = torch.zeros((B,maxlen), dtype=torch.long)
    for i,b in enumerate(batch):
        L = b["input_ids"].shape[0]
        ids[i,:L] = b["input_ids"]
        att[i,:L] = b["attention_mask"]
    return {"input_ids": ids, "attention_mask": att}

def shift_labels_for_ce(input_ids, ignore_index=-100):
    labels = input_ids.clone()
    labels[:,:-1] = input_ids[:,1:]
    labels[:,-1] = ignore_index
    return labels

# ------------------- Hooker -------------------
class KVGradHooker:
    """
    记录每层 k_proj/v_proj 输出的梯度，累积 g^2 的和与样本数。
    同时根据 self_attn.num_key_value_heads / head_dim 还原到 [n_kv_heads, d_kv]。
    """
    def __init__(self):
        self.grad = {}         # (layer, kind) -> last grad [B,T,Hid]
        self.sum_g2 = {}       # (layer, kind) -> [Hid] 累积
        self.token_counts = {} # (layer, kind) -> 累积次数（批次计数）
        self.handles = []
        self.meta_heads = {}   # layer -> (n_kv, d_kv)

    def _key(self, layer, kind): return (int(layer), str(kind))

    def add(self, model):
        core = model.model if hasattr(model, "model") else model
        for i, lyr in enumerate(core.layers):
            attn = getattr(lyr, "self_attn", None)
            if attn is None: continue
            k_proj = getattr(attn, "k_proj", None)
            v_proj = getattr(attn, "v_proj", None)
            n_kv = getattr(attn, "num_key_value_heads", None)
            d_hd = getattr(attn, "head_dim", None)
            if n_kv is not None and d_hd is not None:
                self.meta_heads[i] = (int(n_kv), int(d_hd))

            if k_proj is not None:
                self._reg_linear(i, "k", k_proj)
            if v_proj is not None:
                self._reg_linear(i, "v", v_proj)

    def _reg_linear(self, layer, kind, linear: nn.Linear):
        def bwd_hook(mod, grad_input, grad_output):
            g = grad_output[0]  # [B,T,Hid]
            key = self._key(layer, kind)
            self.grad[key] = g.detach()
            with torch.no_grad():
                g2 = (g ** 2).mean(dim=(0,1))  # [Hid] 
                if key not in self.sum_g2:
                    self.sum_g2[key] = g2.clone()
                    self.token_counts[key] = 1
                else:
                    self.sum_g2[key] += g2
                    self.token_counts[key] += 1

        # 只需要 backward hook（不强制存激活，省显存）
        self.handles.append(linear.register_full_backward_hook(bwd_hook))

    def remove(self):
        for h in self.handles: h.remove()
        self.handles.clear()

    def finalize_fisher(self, num_coupled_channels=None, num_bits=None):
        """返回 Fisher 字典：(layer, kind) -> [n_kv, d_kv]
        
        Args:
            num_coupled_channels: 耦合通道数 (C)，用于量化分组
            num_bits: 量化比特数 (B)
        """
        out = {}
        for key, s in self.sum_g2.items():
            cnt = max(1, self.token_counts.get(key, 1))
            mean_g2 = s / cnt  # [Hid]
            layer, kind = key
            n_kv, d_kv = self.meta_heads.get(layer, (None, None))
            H = mean_g2.numel()
            
            # 如果指定了 num_coupled_channels，使用它作为 d_kv
            if num_coupled_channels is not None:
                d_kv = num_coupled_channels
                if H % d_kv == 0:
                    n_kv = H // d_kv
                else:
                    print(f"Warning: Hidden size {H} not divisible by num_coupled_channels {d_kv}")
                    n_kv = H
                    d_kv = 1
            elif n_kv is None or d_kv is None:
                # 尝试推断
                # 尽量保证能整除
                guess = 1
                for v in (8, 16, 32, 64, 128, 256):
                    if H % v == 0:
                        guess = H // v; d_kv = v; n_kv = guess; break
                if n_kv is None or d_kv is None:
                    n_kv, d_kv = 1, H
            out[key] = mean_g2.view(n_kv, d_kv).cpu().float()
        return out

# ------------------- DDP helpers -------------------
def ddp_is_initialized():
    return dist.is_available() and dist.is_initialized()

def ddp_rank():
    return dist.get_rank() if ddp_is_initialized() else 0

def ddp_world():
    return dist.get_world_size() if ddp_is_initialized() else 1

def all_reduce_inplace(t):
    if ddp_is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

# ------------------- Main -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--calib_txt", type=str, required=True)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--num_samples", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--dump_raw_grads", action="store_true",
                    help="将每个 step 的原始 grad_output 逐层落盘（体积巨大，慎用）")
    ap.add_argument("--num_coupled_channels", type=int, default=None,
                    help="耦合通道数 (C)，用于量化分组，例如 4 表示 4 个通道一组")
    ap.add_argument("--num_bits", type=int, default=None,
                    help="量化比特数 (B)，例如 8 表示 8 比特量化")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # DDP init（如果用 torchrun 启动会自动带 env）
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")

    dtype = {"float16":torch.float16, "bfloat16":torch.bfloat16, "float32":torch.float32}[args.dtype]

    # Load tokenizer/model
    if ddp_rank()==0: print(f"[Load] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=None
    ).to(ddp_rank())
    model = model.to(f"cuda:{ddp_rank()}")
    model.eval()

    # Dataset / Loader
    ds = LineTextDataset(tok, args.calib_txt, args.max_seq_len, num_samples=args.num_samples)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=lambda b: collate_pad(b, tok.pad_token_id))

    # Hook
    hook = KVGradHooker()
    hook.add(model)

    ce = nn.CrossEntropyLoss(ignore_index=-100)

    # Iterate
    for step, batch in enumerate(dl):
        input_ids = batch["input_ids"].to(f"cuda:{ddp_rank()}")
        attn = batch["attention_mask"].to(f"cuda:{ddp_rank()}")

        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits
        labels = shift_labels_for_ce(input_ids)

        loss = ce(logits.view(-1, logits.size(-1)), labels.view(-1))
        model.zero_grad(set_to_none=True)
        loss.backward()

        if args.dump_raw_grads:
            # 将当前 step 的原始梯度保存（按层）
            raw_dir = os.path.join(args.save_dir, f"raw_rank{ddp_rank()}")
            os.makedirs(raw_dir, exist_ok=True)
            for (layer, kind), g in hook.grad.items():
                # g: [B,T,Hid] 存 float16/32
                torch.save(g.cpu().to(torch.float16),
                           os.path.join(raw_dir, f"step{step:05d}_layer{layer:02d}_{kind}.pt"))

        if ddp_rank()==0 and step % 10 == 0:
            print(f"[rank {ddp_rank()}] step {step} loss={loss.item():.4f}")

    # 合并 Fisher 统计：sum_g2 & counts 做 all_reduce
    if ddp_is_initialized():
        for k, v in hook.sum_g2.items():
            v = v.to(f"cuda:{ddp_rank()}")
            all_reduce_inplace(v)
            hook.sum_g2[k] = v.cpu()
        for k, c in hook.token_counts.items():
            tc = torch.tensor([c], dtype=torch.int64, device=f"cuda:{ddp_rank()}")
            all_reduce_inplace(tc)
            hook.token_counts[k] = int(tc.cpu().item())

    fisher = hook.finalize_fisher(
        num_coupled_channels=args.num_coupled_channels,
        num_bits=args.num_bits
    )  # (layer,kind)->[n_kv,d_kv]
    hook.remove()

    # 仅 rank0 落盘
    if ddp_rank()==0:
        meta = {
            "version": "fisher_from_grad_kvproj_prerope",
            "model": args.model,
            "dtype": args.dtype,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "world_size": ddp_world(),
            "num_coupled_channels": args.num_coupled_channels,
            "num_bits": args.num_bits,
            "note": "E[g^2] over batches/tokens at k_proj/v_proj outputs.",
        }
        torch.save({"meta": meta, "fisher": fisher}, os.path.join(args.save_dir, "fisher_diag.pt"))
        with open(os.path.join(args.save_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[DONE] Saved fisher to {os.path.join(args.save_dir, 'fisher_diag.pt')}")
    # 结束 DDP
    if ddp_is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    main()