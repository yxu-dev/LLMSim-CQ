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
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


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


def quantize_cq(input_tensor: torch.Tensor, centroids: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
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

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    # Reshape input: [num_tokens, num_groups, group_channels]
    reshaped = input_tensor.view(num_tokens, num_groups, group_channels).float()

    # Keep computation in float32 for stability and predictable memory use
    centroids = centroids.to(input_tensor.device, dtype=torch.float32)
    centroid_norm = (centroids * centroids).sum(dim=-1).unsqueeze(0)  # [1, G, K]

    quantized_chunks = []
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        chunk = reshaped[start:end]  # [T, G, C]

        # Squared L2 with algebraic expansion:
        # ||x-c||^2 = ||x||^2 + ||c||^2 - 2 * x·c
        chunk_norm = (chunk * chunk).sum(dim=-1, keepdim=True)  # [T, G, 1]
        cross = torch.einsum("tgc,gkc->tgk", chunk, centroids)  # [T, G, K]
        distances = chunk_norm + centroid_norm - (2.0 * cross)  # [T, G, K]

        quantized_chunks.append(torch.argmin(distances, dim=-1).to(torch.uint8))

    return torch.cat(quantized_chunks, dim=0)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match query head count."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def _select_attention_interface(attn_module: nn.Module) -> Callable:
    """Pick eager/flash/sdpa backend when available, with eager fallback."""
    attn_impl = getattr(getattr(attn_module, "config", None), "_attn_implementation", "eager")
    if attn_impl == "eager":
        return _eager_attention_forward

    try:
        from transformers.models.llama.modeling_llama import ALL_ATTENTION_FUNCTIONS

        return ALL_ATTENTION_FUNCTIONS[attn_impl]
    except Exception:
        return _eager_attention_forward


def _compute_kv_rope_embeddings(attn_module: nn.Module, key_states: torch.Tensor) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Build RoPE embeddings for full KV length when cache stores pre-RoPE keys."""
    rotary_emb = getattr(attn_module, "rotary_emb", None)
    if rotary_emb is None:
        rotary_emb = getattr(attn_module, "_cq_rotary_emb", None)
    if rotary_emb is None:
        return None

    bsz, _, kv_seq_len, _ = key_states.shape
    kv_position_ids = torch.arange(kv_seq_len, device=key_states.device).unsqueeze(0).expand(bsz, -1)
    return rotary_emb(key_states, kv_position_ids)


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

    if hasattr(llama_model, "rotary_emb"):
        for layer in getattr(llama_model, "layers", []):
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                setattr(attn, "_cq_rotary_emb", llama_model.rotary_emb)

    if hasattr(llama_model, "_cq_original_forward"):
        return getattr(llama_model, "_cq_disable")

    original_forward = llama_model.forward

    original_attn_forwards: list[tuple[nn.Module, Callable]] = []
    for layer in getattr(llama_model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue

        original_attn_forward = attn.forward
        original_attn_forwards.append((attn, original_attn_forward))

        def make_patched_attention_forward(orig_forward: Callable) -> Callable:
            def patched_attention_forward(
                self,
                hidden_states: torch.Tensor,
                position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Cache] = None,
                cache_position: Optional[torch.LongTensor] = None,
                **kwargs,
            ):
                if not getattr(llama_model, "_cq_enabled", False):
                    return orig_forward(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        past_key_value=past_key_value,
                        cache_position=cache_position,
                        **kwargs,
                    )

                if position_embeddings is None:
                    position_embeddings = kwargs.get("position_embeddings", None)

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)

                query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                # IMPORTANT: update cache with pre-RoPE K/V to match pre-RoPE CQ codebook space.
                if past_key_value is not None:
                    cache_kwargs = {"cache_position": cache_position} if cache_position is not None else None
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, self.layer_idx, cache_kwargs
                    )

                if position_embeddings is None:
                    raise RuntimeError(
                        "Missing position_embeddings for CQ attention patch; cannot apply RoPE to query/key states."
                    )

                cos_q, sin_q = position_embeddings
                query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos_q, sin_q)

                kv_seq_len = key_states.shape[-2]
                q_seq_len = query_states.shape[-2]
                if kv_seq_len == q_seq_len:
                    _, key_states = apply_rotary_pos_emb(key_states, key_states, cos_q, sin_q)
                else:
                    kv_position_embeddings = _compute_kv_rope_embeddings(self, key_states)
                    if kv_position_embeddings is None:
                        # Best-effort fallback: rotate only newly appended window when full KV RoPE is unavailable.
                        _, key_tail = apply_rotary_pos_emb(
                            key_states[..., -q_seq_len:, :], key_states[..., -q_seq_len:, :], cos_q, sin_q
                        )
                        key_states = torch.cat([key_states[..., :-q_seq_len, :], key_tail], dim=-2)
                    else:
                        cos_k, sin_k = kv_position_embeddings
                        _, key_states = apply_rotary_pos_emb(key_states, key_states, cos_k, sin_k)

                attention_interface = _select_attention_interface(self)

                attn_kwargs = dict(kwargs)
                attn_kwargs.pop("position_embeddings", None)
                attn_kwargs.pop("position_ids", None)
                attn_kwargs.pop("cache_position", None)
                attn_kwargs.pop("past_key_value", None)
                attn_kwargs.pop("use_cache", None)

                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=getattr(self, "scaling", self.head_dim**-0.5),
                    **attn_kwargs,
                )

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, attn_weights

            return patched_attention_forward

        attn.forward = types.MethodType(make_patched_attention_forward(original_attn_forward), attn)

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
        for attn, orig_forward in original_attn_forwards:
            attn.forward = orig_forward
        llama_model.forward = original_forward
        setattr(llama_model, "_cq_enabled", False)

    setattr(llama_model, "_cq_disable", disable)
    setattr(llama_model, "_cq_original_forward", original_forward)
    setattr(llama_model, "_cq_original_attn_forwards", original_attn_forwards)
    return disable
