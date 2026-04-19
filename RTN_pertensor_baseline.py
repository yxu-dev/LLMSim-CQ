import torch
import torch.nn as nn

def quantize_rtn_symmetric_per_tensor(x: torch.Tensor, n_bits: int = 4):
    """
    x: float tensor
    returns:
        q: quantized integer tensor
        scale: scalar tensor
    """
    assert x.is_floating_point(), "Input must be floating point"
    if not isinstance(n_bits, int) or n_bits < 2 or n_bits > 8:
        raise ValueError("n_bits must be an integer in [2, 8] for int8 storage")

    # Strict symmetric signed range: [-qmax, qmax]
    # Example: 4-bit -> [-7, 7]
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -qmax

    max_val = x.abs().max()
    scale = max_val / qmax if max_val.item() > 0 else torch.tensor(1.0, device=x.device, dtype=x.dtype)

    q = torch.round(x / scale).clamp(qmin, qmax)

    # int8 is enough to hold 2/4/8-bit values
    q = q.to(torch.int8)

    return q, scale


def dequantize_rtn_symmetric_per_tensor(q: torch.Tensor, scale: torch.Tensor):
    return q.to(scale.dtype) * scale


def quantize_dequantize_rtn_symmetric_per_tensor(x: torch.Tensor, n_bits: int = 4):
    """Convenience helper: quantize then immediately dequantize."""
    q, scale = quantize_rtn_symmetric_per_tensor(x, n_bits=n_bits)
    return dequantize_rtn_symmetric_per_tensor(q, scale)


def apply_rtn_per_tensor_to_model(model: nn.Module, n_bits: int = 4) -> int:
    """
    Apply RTN per-tensor symmetric quantization to all Linear layer weights in-place.

    Returns:
        Number of linear layers processed.
    """
    num_processed = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear) and hasattr(module, "weight") and module.weight is not None:
                module.weight.data = quantize_dequantize_rtn_symmetric_per_tensor(
                    module.weight.data, n_bits=n_bits
                )
                num_processed += 1
    return num_processed