import torch

from lm_eval.quantization.cq_cache import CQCacheLayer, dequantize_cq, quantize_cq


def test_quantize_dequantize_shapes():
    torch.manual_seed(0)
    num_tokens = 16
    num_groups = 8
    group_channels = 4
    num_centroids = 16
    hidden_dim = num_groups * group_channels

    data = torch.randn(num_tokens, hidden_dim)
    codebook = torch.randn(num_groups, num_centroids, group_channels)

    indices = quantize_cq(data, codebook)
    assert indices.shape == (num_tokens, num_groups)
    assert indices.dtype == torch.uint8

    reconstructed = dequantize_cq(indices, codebook)
    assert reconstructed.shape == (num_tokens, hidden_dim)
    assert torch.isfinite(reconstructed).all()


def test_cq_cache_layer_update():
    torch.manual_seed(0)
    num_groups = 8
    group_channels = 4
    hidden_dim = num_groups * group_channels
    num_centroids = 16

    k_cb = torch.randn(num_groups, num_centroids, group_channels)
    v_cb = torch.randn(num_groups, num_centroids, group_channels)
    layer = CQCacheLayer(k_cb, v_cb)

    batch = 2
    num_heads = 4
    head_dim = hidden_dim // num_heads
    seq_len = 3

    key_states = torch.randn(batch, num_heads, seq_len, head_dim)
    value_states = torch.randn(batch, num_heads, seq_len, head_dim)

    decoded_k, decoded_v = layer.update(key_states, value_states)
    assert decoded_k.shape == key_states.shape
    assert decoded_v.shape == value_states.shape
    assert layer.get_seq_length() == seq_len

    layer.reset()
    assert layer.get_seq_length() == 0
