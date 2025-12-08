"""Utilities for Coupled Quantization (CQ) KV-cache integration.

This module provides:
- CodebookManager: loads per-layer K/V centroids.
- quantize_cq / dequantize_cq helpers operating on grouped channels.
- CQCacheLayer / QuantizedDynamicCache: drop-in replacements for the Transformers cache
  that store quantized indices instead of full-precision tensors.
- enable_cq_kv_cache: monkey-patching helper that wires the quantized cache into a
  loaded Hugging Face Llama-* model.
"""

from __future__ import annotations

import os
import types
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import nn

from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.models.llama.configuration_llama import LlamaConfig


@dataclass
class CQQuantizationConfig:
    """Runtime knobs for CQ KV-cache integration."""

    codebook_dir: str
    layer_prefix: str = "layer"
    num_layers: Optional[int] = None
    device: Optional[torch.device] = None


class CodebookManager:
    """Loads Fisher-weighted CQ centroids per layer from disk."""

    def __init__(self, codebook_dir: str, layer_prefix: str = "layer", n_layers: Optional[int] = None):
        self.codebook_dir = codebook_dir
        self.layer_prefix = layer_prefix
        self._k_codebooks: list[torch.Tensor] = []
        self._v_codebooks: list[torch.Tensor] = []

        if not os.path.isdir(codebook_dir):
            raise FileNotFoundError(f"Codebook directory '{codebook_dir}' does not exist")

        layer_idx = 0
        while True:
            if n_layers is not None and layer_idx >= n_layers:
                break
            k_path = self._codebook_path("k", layer_idx)
            v_path = self._codebook_path("v", layer_idx)
            if not (os.path.exists(k_path) and os.path.exists(v_path)):
                if n_layers is None:
                    break
                raise FileNotFoundError(
                    f"Missing codebook .npy for layer {layer_idx}: expected {k_path} and {v_path}"
                )

            self._k_codebooks.append(self._load_numpy(k_path))
            self._v_codebooks.append(self._load_numpy(v_path))
            layer_idx += 1

        if not self._k_codebooks:
            raise RuntimeError(f"No codebooks found under {codebook_dir}")

    @property
    def num_layers(self) -> int:
        return len(self._k_codebooks)

    def _codebook_path(self, kind: str, layer_idx: int) -> str:
        filename = f"{kind}_centroids_fisher_{self.layer_prefix}{layer_idx}.npy"
        return os.path.join(self.codebook_dir, filename)

    @staticmethod
    def _load_numpy(path: str) -> torch.Tensor:
        data = np.load(path)
        return torch.from_numpy(data).to(torch.float32)

    def get_codebooks(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._k_codebooks[layer_idx].clone(), self._v_codebooks[layer_idx].clone()


def quantize_cq(input_tensor: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Quantize grouped channels using CQ codebooks.

    Args:
        input_tensor: [num_tokens, total_channels]
        centroids: [num_groups, num_centroids, group_channels]

    Returns:
        quantized indices shaped [num_tokens, num_groups] (uint8)
    """

    num_tokens, total_channels = input_tensor.shape
    num_groups, num_centroids, group_channels = centroids.shape
    assert (
        total_channels == num_groups * group_channels
    ), f"Mismatched dimensions: {total_channels} vs {num_groups} * {group_channels}"

    # Reshape input: [num_tokens, num_groups, group_channels]
    reshaped = input_tensor.view(num_tokens, num_groups, group_channels).float()
    
    # Ensure centroids are on the same device
    centroids = centroids.to(input_tensor.device)
    
    # Vectorized distance computation: avoid Python loop
    # reshaped: [num_tokens, num_groups, group_channels]
    # centroids: [num_groups, num_centroids, group_channels]
    # Compute L2 distance for all groups at once
    # Expand dimensions for broadcasting: [num_tokens, num_groups, 1, group_channels] - [1, num_groups, num_centroids, group_channels]
    reshaped_expanded = reshaped.unsqueeze(2)  # [num_tokens, num_groups, 1, group_channels]
    centroids_expanded = centroids.unsqueeze(0)  # [1, num_groups, num_centroids, group_channels]
    
    # Compute squared L2 distance (faster than torch.cdist)
    distances = torch.sum((reshaped_expanded - centroids_expanded) ** 2, dim=-1)  # [num_tokens, num_groups, num_centroids]
    
    # Find nearest centroid for each group
    quantized = torch.argmin(distances, dim=-1).to(torch.uint8)  # [num_tokens, num_groups]

    return quantized


def dequantize_cq(indices: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Reconstruct grouped channels from CQ indices.

    Args:
        indices: [num_tokens, num_groups] uint8 tensor.
        centroids: [num_groups, num_centroids, group_channels]

    Returns:
        Reconstructed tensor shaped [num_tokens, num_groups * group_channels]
    """

    num_tokens, num_groups = indices.shape
    _, _, group_channels = centroids.shape
    reconstructed_groups = []

    for group_idx in range(num_groups):
        group_centroids = centroids[group_idx]
        gather_idx = indices[:, group_idx].long()
        reconstructed_groups.append(group_centroids[gather_idx])

    return torch.cat(reconstructed_groups, dim=1)


class CQCacheLayer(CacheLayerMixin):
    """Cache layer that stores CQ-quantized indices instead of dense tensors."""

    is_sliding = False

    def __init__(self, k_codebook: torch.Tensor, v_codebook: torch.Tensor):
        super().__init__()
        self.k_codebook = k_codebook
        self.v_codebook = v_codebook
        self.num_groups = k_codebook.shape[0]
        self.group_channels = k_codebook.shape[2]
        self.hidden_dim = self.num_groups * self.group_channels

        self.k_indices: Optional[torch.Tensor] = None
        self.v_indices: Optional[torch.Tensor] = None
        self.batch_size: Optional[int] = None
        self.num_heads: Optional[int] = None
        self.head_dim: Optional[int] = None
        self.layer_device: Optional[torch.device] = None
        self.compute_device: Optional[torch.device] = None
        self.tensor_dtype: Optional[torch.dtype] = None
        self.total_seq_len: int = 0

    def lazy_initialization(self, key_states: torch.Tensor):
        self.batch_size, self.num_heads, _, self.head_dim = key_states.shape
        self.tensor_dtype = key_states.dtype
        self.layer_device = key_states.device
        self.compute_device = key_states.device

        assert (
            self.num_heads * self.head_dim == self.hidden_dim
        ), "Codebook hidden dim must match KV projection size"

        self.k_codebook = self.k_codebook.to(self.layer_device, dtype=torch.float32)
        self.v_codebook = self.v_codebook.to(self.layer_device, dtype=torch.float32)
        self.k_indices = torch.empty(
            (self.batch_size, 0, self.num_groups), dtype=torch.uint8, device=self.layer_device
        )
        self.v_indices = torch.empty_like(self.k_indices)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k_indices is None:
            self.lazy_initialization(key_states)

        seq_len = key_states.shape[2]
        flat_k = self._flatten_heads(key_states)
        flat_v = self._flatten_heads(value_states)

        new_k_indices = quantize_cq(flat_k, self.k_codebook)
        new_v_indices = quantize_cq(flat_v, self.v_codebook)
        new_k_indices = new_k_indices.view(self.batch_size, seq_len, self.num_groups)
        new_v_indices = new_v_indices.view(self.batch_size, seq_len, self.num_groups)

        self.k_indices = torch.cat([self.k_indices, new_k_indices], dim=1)
        self.v_indices = torch.cat([self.v_indices, new_v_indices], dim=1)
        self.total_seq_len += seq_len

        decoded_k = self._decode_indices(self.k_indices, kind="k")
        decoded_v = self._decode_indices(self.v_indices, kind="v")
        return decoded_k, decoded_v

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        query_length = cache_position.shape[0]
        kv_length = query_length + self.get_seq_length()
        return kv_length, 0

    def get_seq_length(self) -> int:
        return self.total_seq_len

    def get_max_cache_shape(self) -> int:
        return -1

    def reset(self):
        if self.k_indices is not None:
            self.k_indices = torch.empty(
                (self.batch_size, 0, self.num_groups), dtype=torch.uint8, device=self.layer_device
            )
            self.v_indices = torch.empty_like(self.k_indices)
        self.total_seq_len = 0

    def crop(self, max_length: int) -> None:
        if self.k_indices is None:
            return
        self.k_indices = self.k_indices[:, :max_length]
        self.v_indices = self.v_indices[:, :max_length]
        self.total_seq_len = min(self.total_seq_len, max_length)

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.k_indices is None:
            return
        self.k_indices = self.k_indices.repeat_interleave(repeats, dim=0)
        self.v_indices = self.v_indices.repeat_interleave(repeats, dim=0)
        self.batch_size = self.k_indices.shape[0]

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.k_indices is None:
            return
        self.k_indices = self.k_indices[indices]
        self.v_indices = self.v_indices[indices]
        self.batch_size = self.k_indices.shape[0]

    def offload(self):
        if self.k_indices is not None:
            self.k_indices = self.k_indices.to("cpu", non_blocking=True)
            self.v_indices = self.v_indices.to("cpu", non_blocking=True)
            self.layer_device = torch.device("cpu")

    def prefetch(self):
        if self.k_indices is not None and self.compute_device is not None:
            self.k_indices = self.k_indices.to(self.compute_device, non_blocking=True)
            self.v_indices = self.v_indices.to(self.compute_device, non_blocking=True)
            self.layer_device = self.compute_device

    def materialize(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.k_indices is None:
            return None, None
        return self._decode_indices(self.k_indices, kind="k"), self._decode_indices(self.v_indices, kind="v")

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _flatten_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, num_heads, seq_len, head_dim = tensor.shape
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        return tensor.view(batch * seq_len, num_heads * head_dim)

    def _decode_indices(self, indices: torch.Tensor, kind: str) -> torch.Tensor:
        codebook = self.k_codebook if kind == "k" else self.v_codebook
        flat = indices.reshape(-1, self.num_groups)
        decoded = dequantize_cq(flat, codebook)
        decoded = decoded.view(self.batch_size, -1, self.num_heads, self.head_dim)
        decoded = decoded.permute(0, 2, 1, 3).contiguous()
        return decoded.to(self.tensor_dtype)


class QuantizedDynamicCache(Cache):
    """Cache container backed by CQCacheLayer per decoder block."""

    def __init__(self, codebook_manager: CodebookManager, config: LlamaConfig):
        layers = []
        num_layers = min(codebook_manager.num_layers, config.num_hidden_layers)
        for layer_idx in range(num_layers):
            k_cb, v_cb = codebook_manager.get_codebooks(layer_idx)
            layers.append(CQCacheLayer(k_cb, v_cb))
        super().__init__(layers=layers)

    def __getitem__(self, layer_idx: int):  # type: ignore[override]
        if layer_idx < len(self.layers):
            return self.layers[layer_idx].materialize()
        raise KeyError(f"Cache only has {len(self.layers)} layers, got index {layer_idx}")


def enable_cq_kv_cache(model: nn.Module, config: CQQuantizationConfig) -> Callable[[], None]:
    """Enable CQ-quantized KV cache for a loaded Hugging Face Llama model.

    Returns a function that can be called to disable the monkey patch.
    """

    if not hasattr(model, "model"):
        raise ValueError("Expected a LlamaForCausalLM-like module with a `.model` attribute")

    llama_model = model.model
    num_layers = getattr(model.config, "num_hidden_layers", None)
    manager = CodebookManager(config.codebook_dir, config.layer_prefix, num_layers)

    def cache_factory() -> QuantizedDynamicCache:
        return QuantizedDynamicCache(manager, model.config)

    setattr(llama_model, "_cq_cache_factory", cache_factory)
    setattr(llama_model, "_cq_enabled", True)

    if hasattr(llama_model, "_cq_original_forward"):
        return getattr(llama_model, "_cq_disable")

    original_forward = llama_model.forward

    def patched_forward(self, *args, **kwargs):
        use_cache = kwargs.get("use_cache", None)
        past_key_values = kwargs.get("past_key_values", None)
        if getattr(self, "_cq_enabled", False):
            should_use_cache = use_cache if use_cache is not None else True
            if should_use_cache and past_key_values is None:
                kwargs["past_key_values"] = self._cq_cache_factory()
                kwargs.setdefault("use_cache", True)
        return original_forward(*args, **kwargs)

    llama_model.forward = types.MethodType(patched_forward, llama_model)

    def disable():
        llama_model.forward = original_forward
        setattr(llama_model, "_cq_enabled", False)

    setattr(llama_model, "_cq_disable", disable)
    setattr(llama_model, "_cq_original_forward", original_forward)
    return disable
