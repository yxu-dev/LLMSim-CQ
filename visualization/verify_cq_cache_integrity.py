#!/usr/bin/env python
"""Verify CQ KV-cache shape/order/index integrity on exported artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lm_eval.quantization.cq_cache import CQCacheLayer, dequantize_cq, quantize_cq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify CQ cache shape/order/index integrity.")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root containing kv_cache/, centroids/, fisher_diag.pt (optional).",
    )
    parser.add_argument("--layer-idx", type=int, default=0, help="Layer index to verify.")
    parser.add_argument(
        "--sample-idx", type=int, default=0, help="sampleN_layerX_*.pt sample index."
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=8,
        help="num_key_value_heads used to reconstruct runtime [B,H,S,Dh].",
    )
    parser.add_argument(
        "--check-multi-update",
        action="store_true",
        help="Also run synthetic multi-step update index alignment check.",
    )
    return parser.parse_args()


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def verify_real_artifacts(
    data_root: Path, layer_idx: int, sample_idx: int, num_kv_heads: int
) -> dict[str, object]:
    kv_dir = data_root / "kv_cache"
    cent_dir = data_root / "centroids"

    key_path = kv_dir / f"sample{sample_idx}_layer{layer_idx}_key.pt"
    value_path = kv_dir / f"sample{sample_idx}_layer{layer_idx}_value.pt"
    k_cb_path = cent_dir / f"k_centroids_fisher_layer{layer_idx}.npy"
    v_cb_path = cent_dir / f"v_centroids_fisher_layer{layer_idx}.npy"

    key_raw = torch.load(key_path, map_location="cpu").float()  # [B,S,D]
    value_raw = torch.load(value_path, map_location="cpu").float()  # [B,S,D]
    k_cb = torch.from_numpy(np.load(k_cb_path)).float()  # [G,C,Dg]
    v_cb = torch.from_numpy(np.load(v_cb_path)).float()

    if key_raw.ndim != 3 or value_raw.ndim != 3:
        raise ValueError(f"Expected raw K/V shape [B,S,D], got {key_raw.shape} / {value_raw.shape}")

    bsz, seq_len, hidden_dim = key_raw.shape
    if hidden_dim % num_kv_heads != 0:
        raise ValueError(
            f"hidden_dim={hidden_dim} must be divisible by num_kv_heads={num_kv_heads}"
        )
    head_dim = hidden_dim // num_kv_heads

    # Runtime tensors used by CQ cache update: [B,H,S,Dh]
    key_states = key_raw.view(bsz, seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    value_states = value_raw.view(bsz, seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    layer = CQCacheLayer(k_cb, v_cb)
    layer.lazy_initialization(key_states)

    # 1/4) Original runtime shape and dimension order.
    runtime_shape_ok = key_states.shape == (bsz, num_kv_heads, seq_len, head_dim)

    # Flatten -> inverse (without quantization): checks ordering assumptions.
    flat_k = layer._flatten_heads(key_states)
    flat_v = layer._flatten_heads(value_states)
    inv_k = flat_k.view(bsz, seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    inv_v = flat_v.view(bsz, seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    flatten_inverse_exact = bool(torch.equal(inv_k, key_states) and torch.equal(inv_v, value_states))

    # 2) Shape pre/post quantization.
    k_idx = quantize_cq(flat_k, k_cb)
    v_idx = quantize_cq(flat_v, v_cb)
    quant_shape_ok = (
        k_idx.shape == (bsz * seq_len, k_cb.shape[0])
        and v_idx.shape == (bsz * seq_len, v_cb.shape[0])
        and k_idx.dtype == torch.uint8
        and v_idx.dtype == torch.uint8
    )

    # 3) Reconstructed shape equals original runtime shape.
    deq_k = dequantize_cq(k_idx, k_cb)
    deq_v = dequantize_cq(v_idx, v_cb)
    rec_k = deq_k.view(bsz, seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    rec_v = deq_v.view(bsz, seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    recon_shape_ok = rec_k.shape == key_states.shape and rec_v.shape == value_states.shape

    # 5) Token -> centroid index mapping (row order) check.
    mapping_ok = True
    mapping_max_diff = 0.0
    for b in range(bsz):
        for s in (0, min(10, seq_len - 1), seq_len - 1):
            row = b * seq_len + s
            src_k = key_states[b, :, s, :].reshape(-1)
            src_v = value_states[b, :, s, :].reshape(-1)
            d_k = _max_abs_diff(src_k, flat_k[row])
            d_v = _max_abs_diff(src_v, flat_v[row])
            mapping_max_diff = max(mapping_max_diff, d_k, d_v)
            if d_k != 0.0 or d_v != 0.0:
                mapping_ok = False
                break
        if not mapping_ok:
            break

    return {
        "raw_key_shape": tuple(key_raw.shape),
        "raw_value_shape": tuple(value_raw.shape),
        "runtime_key_shape": tuple(key_states.shape),
        "runtime_value_shape": tuple(value_states.shape),
        "k_codebook_shape": tuple(k_cb.shape),
        "v_codebook_shape": tuple(v_cb.shape),
        "flat_k_shape": tuple(flat_k.shape),
        "flat_v_shape": tuple(flat_v.shape),
        "k_indices_shape": tuple(k_idx.shape),
        "v_indices_shape": tuple(v_idx.shape),
        "deq_k_shape": tuple(deq_k.shape),
        "deq_v_shape": tuple(deq_v.shape),
        "recon_key_shape": tuple(rec_k.shape),
        "recon_value_shape": tuple(rec_v.shape),
        "runtime_shape_ok": runtime_shape_ok,
        "flatten_inverse_exact": flatten_inverse_exact,
        "quant_shape_ok": quant_shape_ok,
        "recon_shape_ok": recon_shape_ok,
        "mapping_ok": mapping_ok,
        "mapping_max_diff": mapping_max_diff,
        "flatten_inverse_max_diff_k": _max_abs_diff(inv_k, key_states),
        "flatten_inverse_max_diff_v": _max_abs_diff(inv_v, value_states),
    }


def verify_multi_update_mapping() -> bool:
    torch.manual_seed(0)
    bsz, num_heads, head_dim = 2, 4, 8
    seq1, seq2 = 3, 2
    hidden = num_heads * head_dim
    num_groups = 8
    group_channels = hidden // num_groups
    num_centroids = 16

    k_cb = torch.randn(num_groups, num_centroids, group_channels)
    v_cb = torch.randn(num_groups, num_centroids, group_channels)
    layer = CQCacheLayer(k_cb, v_cb)

    key1 = torch.randn(bsz, num_heads, seq1, head_dim)
    val1 = torch.randn(bsz, num_heads, seq1, head_dim)
    key2 = torch.randn(bsz, num_heads, seq2, head_dim)
    val2 = torch.randn(bsz, num_heads, seq2, head_dim)

    layer.update(key1, val1)
    idx1 = quantize_cq(layer._flatten_heads(key1), k_cb).view(bsz, seq1, num_groups)
    if not torch.equal(layer.k_indices, idx1):
        return False

    layer.update(key2, val2)
    idx2 = quantize_cq(layer._flatten_heads(key2), k_cb).view(bsz, seq2, num_groups)
    expected = torch.cat([idx1, idx2], dim=1)
    return bool(torch.equal(layer.k_indices, expected))


def main() -> None:
    args = parse_args()
    result = verify_real_artifacts(
        data_root=args.data_root,
        layer_idx=args.layer_idx,
        sample_idx=args.sample_idx,
        num_kv_heads=args.num_kv_heads,
    )

    print("=== CQ cache integrity report ===")
    for k in [
        "raw_key_shape",
        "raw_value_shape",
        "runtime_key_shape",
        "runtime_value_shape",
        "k_codebook_shape",
        "v_codebook_shape",
        "flat_k_shape",
        "flat_v_shape",
        "k_indices_shape",
        "v_indices_shape",
        "deq_k_shape",
        "deq_v_shape",
        "recon_key_shape",
        "recon_value_shape",
    ]:
        print(f"{k}: {result[k]}")

    print("\n--- 5 checks ---")
    print(f"1) original runtime K/V shape visible: {result['runtime_shape_ok']}")
    print(f"2) pre/post quantization shape transition expected: {result['quant_shape_ok']}")
    print(f"3) reconstructed K/V shape equals original runtime shape: {result['recon_shape_ok']}")
    print("4) dimension order [B,H,S,Dh] consistency (flatten<->inverse exact): "
          f"{result['flatten_inverse_exact']} "
          f"(maxdiff K={result['flatten_inverse_max_diff_k']}, V={result['flatten_inverse_max_diff_v']})")
    print(f"5) token->centroid row mapping aligned (no position shift): {result['mapping_ok']} "
          f"(maxdiff={result['mapping_max_diff']})")

    if args.check_multi_update:
        multi_ok = verify_multi_update_mapping()
        print(f"extra) multi-step update index alignment: {multi_ok}")


if __name__ == "__main__":
    main()
