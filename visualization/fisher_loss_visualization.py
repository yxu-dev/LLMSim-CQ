#!/usr/bin/env python
"""Visualize Fisher-weighted reconstruction error convergence curves.

This script re-runs weighted K-Means on sampled KV activations and plots
the Fisher-weighted reconstruction error over iterations for:
  - 2c8b
  - 4c8b
  - 8c8b

Outputs are saved to:
  /home/zz359/workspace-CQ-zzy/LLMSim-CQ-zzy/visualization/fisher-loss
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_OUTPUT_DIR = Path(
    "/home/zz359/workspace-CQ-zzy/LLMSim-CQ-zzy/visualization/fisher-loss"
)
DEFAULT_ROOT_DIR = Path("/home/zz359/workspace-CQ-zzy/LLMSim-CQ-zzy/output")
DEFAULT_CONFIGS = ["llama-3.1-8b-2c8b", "llama-3.1-8b-4c8b", "llama-3.1-8b-8c8b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Fisher-weighted reconstruction error curves for 2c8b/4c8b/8c8b."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=DEFAULT_ROOT_DIR,
        help=f"Directory containing config folders (default: {DEFAULT_ROOT_DIR})",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Config folder names under --root-dir.",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=16,
        help="Layer index for convergence analysis (default: 16).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=16,
        help="Max sample*.pt files used per config/layer/kind (default: 16).",
    )
    parser.add_argument(
        "--rows-per-sample",
        type=int,
        default=512,
        help="Rows sampled from each sample file after flattening (default: 512).",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=64,
        help=(
            "Max channel-groups used per kind (uniformly sampled). "
            "Reduce for faster runs, increase for higher fidelity (default: 64)."
        ),
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=60,
        help="Max K-Means iterations for each group (default: 60).",
    )
    parser.add_argument(
        "--num-bits",
        type=int,
        default=8,
        help="Bit width for codebook size K=2^bits (default: 8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device preference (default: cuda).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for curves (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display interactive plot window.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_coupled_channels(config_name: str) -> int:
    match = re.search(r"-(\d+)c\d+b$", config_name)
    if not match:
        raise ValueError(f"Cannot infer coupled channels from config name: {config_name}")
    return int(match.group(1))


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_sampled_activations(
    kv_cache_dir: Path,
    layer_idx: int,
    kind: str,
    max_samples: int,
    rows_per_sample: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    pattern = f"sample*_layer{layer_idx}_{kind}.pt"
    files = sorted(kv_cache_dir.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found for pattern: {kv_cache_dir / pattern}")
    files = files[:max_samples]

    chunks: List[torch.Tensor] = []
    for path in files:
        tensor = torch.load(path, map_location="cpu").float()
        if tensor.ndim == 3:
            tensor = tensor.reshape(-1, tensor.shape[-1])
        elif tensor.ndim != 2:
            raise ValueError(f"Unexpected tensor ndim={tensor.ndim} in {path}")

        n_rows = tensor.shape[0]
        if n_rows > rows_per_sample:
            indices = rng.choice(n_rows, size=rows_per_sample, replace=False)
            tensor = tensor[indices]
        chunks.append(tensor)

    return torch.cat(chunks, dim=0)  # [N, D]


def weighted_kmeans_pp_init(
    data: torch.Tensor, weights: torch.Tensor, k: int, gen: torch.Generator
) -> torch.Tensor:
    n, d = data.shape
    centroids = torch.empty((k, d), device=data.device, dtype=torch.float32)
    first_idx = torch.randint(0, n, (1,), generator=gen, device=data.device)
    centroids[0] = data[first_idx]

    weights_expanded = weights.unsqueeze(0)
    min_dist = ((data - centroids[0]) ** 2 * weights_expanded).sum(dim=1)

    for i in range(1, k):
        prob_sum = min_dist.sum()
        if not torch.isfinite(prob_sum) or prob_sum <= 0:
            next_idx = torch.randint(0, n, (1,), generator=gen, device=data.device)
        else:
            probs = min_dist / prob_sum
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            next_idx = torch.multinomial(probs, num_samples=1, generator=gen)
        centroids[i] = data[next_idx]
        dist_new = ((data - centroids[i]) ** 2 * weights_expanded).sum(dim=1)
        min_dist = torch.minimum(min_dist, dist_new)

    return centroids


def weighted_kmeans_loss_curve(
    data: torch.Tensor,
    weights: torch.Tensor,
    k: int,
    max_iter: int,
    gen: torch.Generator,
) -> List[float]:
    data = data.to(torch.float32)
    weights = weights.to(torch.float32)
    n, d = data.shape
    k = int(min(k, n))
    if k < 1:
        raise ValueError("k must be >= 1")

    centroids = weighted_kmeans_pp_init(data, weights, k, gen)
    weights_expanded = weights.unsqueeze(0)

    loss_curve: List[float] = []
    last_assignments: torch.Tensor | None = None
    for _ in range(max_iter):
        diff = data.unsqueeze(1) - centroids.unsqueeze(0)  # [N, K, D]
        dist = (diff * diff * weights_expanded.unsqueeze(1)).sum(dim=2)  # [N, K]
        min_dist, assignments = torch.min(dist, dim=1)  # [N], [N]
        loss_curve.append(float(min_dist.mean().item()))

        if last_assignments is not None and torch.equal(assignments, last_assignments):
            break
        last_assignments = assignments

        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros((k,), device=data.device, dtype=torch.float32)
        expand_idx = assignments.unsqueeze(1).expand(-1, d)
        new_centroids.scatter_add_(0, expand_idx, data)
        counts.scatter_add_(0, assignments, torch.ones_like(assignments, dtype=torch.float32))

        empty = counts < 1e-6
        counts = counts.clamp_min(1.0)
        new_centroids = new_centroids / counts.unsqueeze(1)
        if empty.any():
            rand_idx = torch.randint(
                0, n, (int(empty.sum().item()),), generator=gen, device=data.device
            )
            new_centroids[empty] = data[rand_idx]
        centroids = new_centroids

    return loss_curve


def aggregate_curves(curves: List[List[float]]) -> np.ndarray:
    if not curves:
        raise ValueError("curves is empty")
    max_len = max(len(c) for c in curves)
    arr = np.zeros((len(curves), max_len), dtype=np.float32)
    for i, curve in enumerate(curves):
        arr[i, : len(curve)] = np.array(curve, dtype=np.float32)
        if len(curve) < max_len:
            arr[i, len(curve) :] = curve[-1]
    return arr.mean(axis=0)


def summarize_config_curve(
    config_dir: Path,
    config_name: str,
    layer_idx: int,
    max_samples: int,
    rows_per_sample: int,
    max_groups: int,
    max_iter: int,
    num_bits: int,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    kv_cache_dir = config_dir / "kv_cache"
    fisher_path = config_dir / "fisher_diag.pt"
    if not kv_cache_dir.exists():
        raise FileNotFoundError(f"Missing kv_cache dir: {kv_cache_dir}")
    if not fisher_path.exists():
        raise FileNotFoundError(f"Missing fisher file: {fisher_path}")

    fisher_loaded = torch.load(fisher_path, map_location="cpu")
    fisher_dict = fisher_loaded["fisher"]
    coupled_channels = infer_coupled_channels(config_name)
    rng = np.random.default_rng(seed)

    all_curves: List[List[float]] = []
    for kind in ("key", "value"):
        fisher_kind = "k" if kind == "key" else "v"
        fisher_weights = fisher_dict.get((layer_idx, fisher_kind))
        if fisher_weights is None:
            raise KeyError(f"Missing fisher key {(layer_idx, fisher_kind)} in {fisher_path}")

        data = load_sampled_activations(
            kv_cache_dir=kv_cache_dir,
            layer_idx=layer_idx,
            kind=kind,
            max_samples=max_samples,
            rows_per_sample=rows_per_sample,
            rng=rng,
        )
        n, d = data.shape
        num_groups = d // coupled_channels
        if d % coupled_channels != 0:
            raise ValueError(
                f"Activation dim {d} not divisible by coupled channels {coupled_channels}"
            )
        if tuple(fisher_weights.shape) != (num_groups, coupled_channels):
            raise ValueError(
                f"Fisher shape mismatch for {config_name}/{kind}: "
                f"{tuple(fisher_weights.shape)} vs {(num_groups, coupled_channels)}"
            )

        if num_groups > max_groups:
            group_indices = np.linspace(0, num_groups - 1, max_groups, dtype=np.int64)
        else:
            group_indices = np.arange(num_groups, dtype=np.int64)

        k = 2**num_bits
        gen = torch.Generator(device=device.type).manual_seed(seed + (17 if kind == "value" else 7))

        data = data.to(device)
        fisher_weights = fisher_weights.to(device=device, dtype=torch.float32)
        for g in group_indices:
            start = int(g) * coupled_channels
            end = start + coupled_channels
            group_data = data[:, start:end]
            group_w = fisher_weights[g]
            curve = weighted_kmeans_loss_curve(
                data=group_data,
                weights=group_w,
                k=k,
                max_iter=max_iter,
                gen=gen,
            )
            all_curves.append(curve)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return aggregate_curves(all_curves)


def plot_curves(curves: Dict[str, np.ndarray], output_png: Path, no_show: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, curve in curves.items():
        x = np.arange(1, len(curve) + 1, dtype=np.int32)
        ax.plot(x, curve, linewidth=2.0, label=name)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fisher-weighted reconstruction error")
    ax.set_title("Fisher-weighted reconstruction error convergence")
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.legend()
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    print(f"[Saved] {output_png}")

    if not no_show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    curves: Dict[str, np.ndarray] = {}
    for config_name in args.configs:
        config_dir = args.root_dir / config_name
        print(f"\n[Running] {config_name} @ layer{args.layer_idx}")
        curve = summarize_config_curve(
            config_dir=config_dir,
            config_name=config_name,
            layer_idx=args.layer_idx,
            max_samples=args.max_samples,
            rows_per_sample=args.rows_per_sample,
            max_groups=args.max_groups,
            max_iter=args.max_iter,
            num_bits=args.num_bits,
            device=device,
            seed=args.seed,
        )
        curves[config_name] = curve
        print(
            f"  iterations={len(curve)}, first={curve[0]:.6f}, "
            f"last={curve[-1]:.6f}, rel_drop={(curve[0]-curve[-1])/(curve[0]+1e-12):.2%}"
        )

    plot_path = output_dir / f"fisher_weighted_loss_curve_layer{args.layer_idx}.png"
    plot_curves(curves, plot_path, args.no_show)

    curve_json = output_dir / f"fisher_weighted_loss_curve_layer{args.layer_idx}.json"
    serializable = {k: [float(vv) for vv in v.tolist()] for k, v in curves.items()}
    curve_json.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"[Saved] {curve_json}")


if __name__ == "__main__":
    main()
