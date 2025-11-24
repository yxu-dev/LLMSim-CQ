#!/usr/bin/env python
"""Run perplexity evaluation with CQ-quantized KV cache enabled."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lm_eval.quantization.cq_cache import CQQuantizationConfig, enable_cq_kv_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CQ-quantized Llama models.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model ID or path")
    parser.add_argument("--codebook-dir", required=True, help="Directory containing Fisher CQ codebooks")
    parser.add_argument("--layer-prefix", default="layer", help="Filename prefix used in codebooks")
    parser.add_argument("--dataset", default="wikitext", help="Hugging Face dataset name")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1", help="Dataset config name")
    parser.add_argument("--split", default="test", help="Dataset split (can be sliced)")
    parser.add_argument("--limit", type=int, default=None, help="Optional dataset slice length (e.g., 1 for split[:1])")
    parser.add_argument("--max-eval-tokens", type=int, default=131072, help="Maximum tokens to evaluate")
    parser.add_argument("--stride", type=int, default=512, help="Stride for sliding-window perplexity")
    parser.add_argument("--max-length", type=int, default=2048, help="Context length for evaluation window")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Computation dtype")
    parser.add_argument("--disable-cq", action="store_true", help="Skip CQ patching (baseline run)")
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def load_model_and_tokenizer(model_id: str, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=dtype,
    )
    model.eval()
    return model, tokenizer


def sliding_window_perplexity(
    model,
    input_ids: torch.Tensor,
    stride: int,
    max_length: int,
) -> float:
    nlls = []
    total_tokens = 0
    seq_len = input_ids.size(1)
    for i in range(0, seq_len, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)
        trg_len = end_loc - i
        if trg_len <= 0:
            continue
        input_slice = input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_slice.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_slice, labels=target_ids, use_cache=True)
        nlls.append(outputs.loss.float() * trg_len)
        total_tokens += trg_len
        if end_loc == seq_len:
            break
    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / total_tokens)
    return ppl.item()


def main():
    args = parse_args()
    dtype = get_dtype(args.dtype)
    model, tokenizer = load_model_and_tokenizer(args.model, dtype)

    if not args.disable_cq:
        cq_config = CQQuantizationConfig(
            codebook_dir=args.codebook_dir,
            layer_prefix=args.layer_prefix,
            num_layers=getattr(model.config, "num_hidden_layers", None),
        )
        enable_cq_kv_cache(model, cq_config)

    split = args.split
    if args.limit is not None:
        split = f"{split}[:{args.limit}]"
    dataset = load_dataset(args.dataset, args.dataset_config, split=split)
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    if args.max_eval_tokens is not None:
        encodings["input_ids"] = encodings["input_ids"][..., : args.max_eval_tokens]
    input_ids = encodings["input_ids"].to(model.device)

    ppl = sliding_window_perplexity(model, input_ids, args.stride, args.max_length)
    status = "CQ" if not args.disable_cq else "baseline"
    print(f"[{status}] Perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()
