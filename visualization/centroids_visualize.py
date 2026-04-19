#!/usr/bin/env python
"""Centroid visualization toolkit for CQ k-means outputs.

Centroids are stored as a 3D tensor:
    [num_groups, K, group_dim]

This script can visualize centroids from multiple perspectives, and supports
PCA-3D only mode for faster batch analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize centroid .npy files.")
    parser.add_argument(
        "--centroid-path",
        type=Path,
        required=True,
        help="Path to centroid .npy file, e.g. k_centroids_fisher_layer1.npy",
    )
    parser.add_argument(
        "--group-id",
        type=int,
        default=0,
        help="Group id for single-group heatmap (default: 0)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=120,
        help="Histogram bins (default: 120)",
    )
    parser.add_argument(
        "--pca-max-points",
        type=int,
        default=10000,
        help="Max points shown in global 3D PCA plot (default: 10000)",
    )
    parser.add_argument(
        "--plot-mode",
        type=str,
        choices=["all", "pca3d"],
        default="pca3d",
        help="Visualization mode: all plots or only 3D PCA (default: pca3d)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set, save figures to this directory",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show() (useful on servers)",
    )
    return parser.parse_args()


def load_centroids(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Centroid file not found: {path}")
    centroids = np.load(path)
    if centroids.ndim != 3:
        raise ValueError(
            f"Expected centroid shape [num_groups, K, group_dim], got {centroids.shape}"
        )
    return centroids.astype(np.float32, copy=False)


def save_or_show(fig: plt.Figure, output_path: Path | None, no_show: bool) -> None:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        print(f"[Saved] {output_path}")
    if not no_show:
        plt.show()
    plt.close(fig)


def plot_distribution(
    centroids: np.ndarray,
    bins: int,
    title_prefix: str,
    output_path: Path | None,
    no_show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(centroids.ravel(), bins=bins, color="#3366cc", alpha=0.85)
    ax.set_title(f"{title_prefix}\nCentroid Value Distribution")
    ax.set_xlabel("Centroid value")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2, linewidth=0.6)
    save_or_show(fig, output_path, no_show)


def plot_norm_heatmap(
    centroids: np.ndarray,
    title_prefix: str,
    output_path: Path | None,
    no_show: bool,
) -> None:
    norms = np.linalg.norm(centroids, axis=-1)  # [num_groups, K]
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(norms, aspect="auto", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("L2 norm")
    ax.set_title(f"{title_prefix}\nCentroid Norms (groups x K)")
    ax.set_xlabel("Centroid index")
    ax.set_ylabel("Group index")
    save_or_show(fig, output_path, no_show)


def plot_group_heatmap(
    centroids: np.ndarray,
    group_id: int,
    title_prefix: str,
    output_path: Path | None,
    no_show: bool,
) -> None:
    num_groups = centroids.shape[0]
    if group_id < 0 or group_id >= num_groups:
        raise ValueError(f"group_id must be in [0, {num_groups - 1}], got {group_id}")

    group = centroids[group_id]  # [K, group_dim]
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(group, aspect="auto", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Centroid value")
    ax.set_title(f"{title_prefix}\nGroup {group_id} Centroids (K x group_dim)")
    ax.set_xlabel("Channel in group")
    ax.set_ylabel("Centroid index")
    save_or_show(fig, output_path, no_show)


def plot_global_pca_3d(
    centroids: np.ndarray,
    title_prefix: str,
    max_points: int,
    output_path: Path | None,
    no_show: bool,
) -> None:
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("[Skip] 3D PCA plot requires scikit-learn (`pip install scikit-learn`).")
        return

    num_groups, num_centroids, group_dim = centroids.shape
    flat = centroids.reshape(num_groups * num_centroids, group_dim)
    group_ids = np.repeat(np.arange(num_groups, dtype=np.int32), num_centroids)
    norm_values = np.linalg.norm(flat, axis=1)

    if flat.shape[0] > max_points:
        indices = np.linspace(0, flat.shape[0] - 1, num=max_points, dtype=np.int64)
        flat = flat[indices]
        group_ids = group_ids[indices]
        norm_values = norm_values[indices]
        print(f"[Info] PCA plot subsampled to {max_points} points.")

    if group_dim >= 3:
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(flat)
        ratio = pca.explained_variance_ratio_ * 100.0

        fig = plt.figure(figsize=(8.5, 7))
        ax = fig.add_subplot(111, projection="3d")
        points = ax.scatter(
            reduced[:, 0],
            reduced[:, 1],
            reduced[:, 2],
            c=group_ids,
            s=10 + 25 * (norm_values / (norm_values.max() + 1e-6)),
            cmap="turbo",
            alpha=0.75,
        )
        cbar = fig.colorbar(points, ax=ax, pad=0.1)
        cbar.set_label("Group index")
        ax.set_title(
            f"{title_prefix}\nGlobal 3D PCA of centroids "
            f"(var: {ratio[0]:.1f}% / {ratio[1]:.1f}% / {ratio[2]:.1f}%)"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        save_or_show(fig, output_path, no_show)
        return

    # Auto fallback for low-dimensional groups (e.g. 2c8b -> group_dim=2).
    if group_dim >= 2:
        print(f"[Info] group_dim={group_dim}, fallback to PCA2D.")
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(flat)
        ratio = pca.explained_variance_ratio_ * 100.0

        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        points = ax.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=group_ids,
            s=10 + 25 * (norm_values / (norm_values.max() + 1e-6)),
            cmap="turbo",
            alpha=0.75,
        )
        cbar = fig.colorbar(points, ax=ax)
        cbar.set_label("Group index")
        ax.set_title(
            f"{title_prefix}\nGlobal 2D PCA fallback "
            f"(var: {ratio[0]:.1f}% / {ratio[1]:.1f}%)"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2, linewidth=0.6)
        save_or_show(fig, output_path, no_show)
        return

    print(f"[Skip] PCA needs group_dim >= 2, got {group_dim}")


def main() -> None:
    args = parse_args()
    centroids = load_centroids(args.centroid_path)
    num_groups, num_centroids, group_dim = centroids.shape

    print(f"Loaded: {args.centroid_path}")
    print(f"Shape: {centroids.shape} = [num_groups, K, group_dim]")
    print(
        f"Stats: min={centroids.min():.6f}, max={centroids.max():.6f}, "
        f"mean={centroids.mean():.6f}, std={centroids.std():.6f}"
    )
    print(
        f"Visualizing mode={args.plot_mode}, group_id={args.group_id}, bins={args.bins}"
    )

    stem = args.centroid_path.stem
    title_prefix = f"{stem} ({num_groups} groups, K={num_centroids}, C={group_dim})"

    out_hist = None
    out_norm = None
    out_group = None
    out_pca3 = None
    if args.output_dir is not None:
        out_hist = args.output_dir / f"{stem}_hist.png"
        out_norm = args.output_dir / f"{stem}_norms_heatmap.png"
        out_group = args.output_dir / f"{stem}_group{args.group_id}_heatmap.png"
        out_pca3 = args.output_dir / f"{stem}_pca3d.png"

    if args.plot_mode == "all":
        plot_distribution(centroids, args.bins, title_prefix, out_hist, args.no_show)
        plot_norm_heatmap(centroids, title_prefix, out_norm, args.no_show)
        plot_group_heatmap(
            centroids, args.group_id, title_prefix, out_group, args.no_show
        )

    plot_global_pca_3d(
        centroids,
        title_prefix,
        args.pca_max_points,
        out_pca3,
        args.no_show,
    )


if __name__ == "__main__":
    main()
