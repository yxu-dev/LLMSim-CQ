#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出 Llama 模型的 k/v 前向激活（与 `llama_simi_attention_forward`
相同的 sample / phase / step 命名逻辑），并把超过阈值的序列长度
记录到 txt 文件。
"""

import argparse
import datetime
import os
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------- Dataset -------------------
class LineTextDataset(Dataset):
    def __init__(self, tok, path: str, max_len: int, num_samples: Optional[int] = None):
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if num_samples is not None:
            lines = lines[:num_samples]
        enc = tok(lines, padding=False, truncation=True, max_length=max_len, return_tensors=None)
        self.ids = enc["input_ids"]
        self.attn = enc["attention_mask"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.ids[i], dtype=torch.long),
            "attention_mask": torch.tensor(self.attn[i], dtype=torch.long),
        }


def collate_pad(batch, pad_id: int):
    maxlen = max(x["input_ids"].shape[0] for x in batch)
    B = len(batch)
    ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    att = torch.zeros((B, maxlen), dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        ids[i, :L] = b["input_ids"]
        att[i, :L] = b["attention_mask"]
    return {"input_ids": ids, "attention_mask": att}


# ------------------- KV exporter -------------------
class SimiKVHook:
    """
    使用与 llama_simi_attention_forward 相同的命名与计数逻辑，
    将各层 k_proj / v_proj 的前向输出落盘，并记录超长序列。
    """

    def __init__(self, output_dir: str, long_seq_threshold: int = 2048, verbose: bool = False):
        self.output_dir = output_dir
        self.long_seq_threshold = long_seq_threshold
        self.verbose = verbose
        self.long_seq_log = os.path.join(output_dir, "prefill_long_sequences.txt")

        self.handles = []
        self.cache: Dict[Tuple[int, str], torch.Tensor] = {}
        self.sample_counter = 0
        self.current_context: Optional[Dict[str, int]] = None

    def _make_hook(self, layer_idx: int, kind: str):
        def fwd_hook(mod, inp, out):
            # 保存到 CPU，避免占用过多显存
            self.cache[(layer_idx, kind)] = out.detach().cpu()

        return fwd_hook

    def add(self, model):
        os.makedirs(self.output_dir, exist_ok=True)
        core = model.model if hasattr(model, "model") else model
        layers = getattr(core, "layers", [])
        for i, lyr in enumerate(layers):
            attn = getattr(lyr, "self_attn", None)
            if attn is None:
                continue
            if hasattr(attn, "k_proj"):
                self.handles.append(attn.k_proj.register_forward_hook(self._make_hook(i, "key")))
            if hasattr(attn, "v_proj"):
                self.handles.append(attn.v_proj.register_forward_hook(self._make_hook(i, "value")))

    def set_context(self, seq_len: int, has_past_kv: bool = False, past_len: int = 0):
        if seq_len > 1:
            phase = "Prefill"
            step = 0
            new_sample = True
        elif has_past_kv:
            phase = "Decode"
            step = int(past_len) + 1
            new_sample = False
        else:
            phase = "Initial"
            step = 0
            new_sample = True

        if new_sample or self.sample_counter == 0:
            self.sample_counter += 1

        self.current_context = {
            "sample": self.sample_counter,
            "phase": phase,
            "step": step,
            "seq_len": seq_len,
        }

    def flush(self):
        if not self.cache:
            self.current_context = None
            return

        ctx = self.current_context or {
            "sample": self.sample_counter or 0,
            "phase": "Unknown",
            "step": 0,
            "seq_len": None,
        }

        shape_cache: Dict[int, Dict[str, Tuple[int, ...]]] = {}
        for (layer, kind), tensor in self.cache.items():
            fname = f"sample{ctx['sample']}_{ctx['phase']}_step{ctx['step']}_layer{layer}_{kind}.pt"
            torch.save(tensor, os.path.join(self.output_dir, fname))
            shape_cache.setdefault(layer, {})[kind] = tuple(tensor.shape)

        if self.verbose:
            if ctx["phase"] == "Prefill":
                step_info = f"processing {ctx['seq_len']} tokens"
            elif ctx["phase"] == "Decode":
                step_info = f"step {ctx['step']}"
            else:
                step_info = "first token"

            for layer, kv in sorted(shape_cache.items()):
                kshape = kv.get("key")
                vshape = kv.get("value")
                print(
                    f"Sample {ctx['sample']} - {ctx['phase']} - Layer {layer} ({step_info}): "
                    f"key_states={kshape}, value_states={vshape}"
                )

        if (
            ctx["phase"] == "Prefill"
            and ctx.get("seq_len") is not None
            and ctx["seq_len"] > self.long_seq_threshold
        ):
            os.makedirs(os.path.dirname(self.long_seq_log), exist_ok=True)
            kshape = shape_cache.get(0, {}).get("key")
            vshape = shape_cache.get(0, {}).get("value")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.long_seq_log, "a", encoding="utf-8") as f:
                f.write(
                    f"[{timestamp}] Sample {ctx['sample']}: Prefill phase with seq_len={ctx['seq_len']} tokens, "
                    f"key_states={kshape}, value_states={vshape}\n"
                )

        self.cache.clear()
        self.current_context = None
        torch.cuda.empty_cache()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.cache.clear()
        self.current_context = None


# ------------------- Calib text helper -------------------
def prepare_calib_txt(
    output_dir: str,
    dataset: str,
    dataset_config: Optional[str],
    split: str,
    text_field: str,
    num_samples: Optional[int],
) -> str:
    """
    从 HuggingFace 数据集自动生成 calib.txt，写入 output_dir。
    返回 calib.txt 路径。
    """
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover - import guard
        raise ImportError("需要安装 datasets 才能自动生成 calib.txt，请执行 pip install datasets") from e

    calib_path = os.path.join(output_dir, "calib.txt")
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集，如果 dataset_config 为 None 或空字符串则不传递
    if dataset_config:
        ds = load_dataset(dataset, dataset_config, split=split)
    else:
        ds = load_dataset(dataset, split=split)

    print(f"[calib] 加载数据集: {dataset}, config={dataset_config}, split={split}, 总样本数={len(ds)}")
    
    # 打印数据集的列名以便调试
    if len(ds) > 0:
        print(f"[calib] 数据集字段: {list(ds[0].keys())}")

    if num_samples is not None:
        ds = ds.select(range(min(num_samples, len(ds))))

    texts = []
    for idx, row in enumerate(ds):
        val = row.get(text_field)
        if val is None:
            if idx == 0:
                print(f"[calib] 警告: 字段 '{text_field}' 不存在，可用字段: {list(row.keys())}")
            continue
        if isinstance(val, list):
            val = " ".join(map(str, val))
        val = str(val).strip().replace("\n", " ")
        if val:
            texts.append(val)

    if len(texts) == 0:
        raise ValueError(
            f"未能从数据集中提取任何文本！数据集={dataset}, config={dataset_config}, "
            f"字段={text_field}。请检查参数是否正确。"
        )

    with open(calib_path, "w", encoding="utf-8") as f:
        for line in texts:
            f.write(line + "\n")

    print(f"[calib] 已生成 {len(texts)} 条样本到 {calib_path}")
    return calib_path


# ------------------- Main -------------------
def main():
    ap = argparse.ArgumentParser(description="导出 Llama k/v 激活（simi 风格命名）并记录超长序列。")
    ap.add_argument("--model", type=str, required=True, help="模型名称或本地路径")
    ap.add_argument("--output_dir", type=str, required=True, help="KV 导出目录（文件名为 sample_phase_step_layer_key/value.pt）")
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument(
        "--long_seq_threshold",
        type=int,
        default=2048,
        help="序列长度大于该值时记录到 prefill_long_sequences.txt",
    )
    ap.add_argument("--verbose", action="store_true", help="打印每层的 KV 形状")
    ap.add_argument("--device", type=str, default=None, help="显卡如 cuda:0；留空自动选择")
    ap.add_argument(
        "--dataset",
        type=str,
        default="EleutherAI/wikitext_document_level",
        help="用于生成 calib.txt 的 HuggingFace 数据集名",
    )
    ap.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="HuggingFace 数据集 config（wikitext 可选: wikitext-2-raw-v1, wikitext-103-raw-v1）",
    )
    ap.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="数据集 split（可使用切片语法，如 train[:1024]）",
    )
    ap.add_argument(
        "--dataset_text_field",
        type=str,
        default="page",
        help="文本字段名称（默认 wikitext_document_level 使用 page）",
    )
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    calib_txt_path = prepare_calib_txt(
        output_dir=args.output_dir,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
        text_field=args.dataset_text_field,
        num_samples=args.num_samples,
    )

    ds = LineTextDataset(tok, calib_txt_path, args.max_seq_len, num_samples=None)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_pad(b, tok.pad_token_id),
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=None
    ).to(device)
    model.eval()

    kv_hook = SimiKVHook(
        output_dir=args.output_dir,
        long_seq_threshold=args.long_seq_threshold,
        verbose=args.verbose,
    )
    kv_hook.add(model)

    with torch.no_grad():
        for step, batch in enumerate(dl):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            kv_hook.set_context(seq_len=int(input_ids.shape[1]), has_past_kv=False, past_len=0)
            model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            kv_hook.flush()

            if step % 10 == 0:
                print(
                    f"[step {step}] seq_len={input_ids.shape[1]} -> sample {kv_hook.sample_counter}, "
                    f"saved to {args.output_dir}"
                )

    kv_hook.remove()
    print(f"[DONE] KV 导出完成，结果保存在：{args.output_dir}")


if __name__ == "__main__":
    main()