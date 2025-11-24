import math
import os
from typing import Any


import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F


# meta
TEXT_TOKEN = -1
IGNORE_TOKEN = -2


def get_attr_by_name(obj: Any, name: str) -> Any:
    """
    Get an attribute from an object using a dot notation string.
    e.g., get_attr_by_name(model, "layers.0.self_attn.q_proj") will return model.layers[0].self_attn.q_proj
    """
    levels = name.split(".")
    current = obj
    for level in levels:
        if level.isdigit():
            current = current[int(level)]
        else:
            current = getattr(current, level)
    return current


# Efficient implementation equivalent to the following:
def naive_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(query.device)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(
        query.dtype
    )

    return attn_weight @ value



def save_video_frames(video, output_path: str = "local/video_frames"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    to_pil = T.ToPILImage()
    for i, frame in enumerate(video[0]):
        frame_float = frame.to(torch.float32)
        frame_float = (frame_float + 1) / 2
        frame_float = torch.clamp(frame_float, 0, 1)
        frame_pil = to_pil(frame_float)
        frame_pil.save(os.path.join(output_path, f"frame_{i}.png"))


def save_video_frames_subfigures(video, output_path: str = "local/video_frames.jpg"):
    """
    Save the video frames as subfigures in a single image.
    """
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    num_frames = len(video[0])
    rows = int(np.sqrt(num_frames))
    cols = int(np.ceil(num_frames / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    to_pil = T.ToPILImage()
    for i, frame in enumerate(video[0]):
        frame_float = frame.to(torch.float32)
        frame_float = (frame_float + 1) / 2
        frame_float = torch.clamp(frame_float, 0, 1)
        frame_pil = to_pil(frame_float)

        axes[i].imshow(frame_pil)
        axes[i].axis("off")
        axes[i].set_title(f"Frame {i}")

    # Hide empty subplots
    for i in range(num_frames, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


class AverageMeter:
    """Computes and stores the average, current value, sum, and count."""
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all statistics."""
        self.val = 0.0      # current value
        self.avg = 0.0       # average
        self.sum = 0.0       # sum of all values
        self.count = 0.0     # number of updates

    def update(self, val, n=1):
        """Updates the meter with a new value.
        
        Args:
            val (float): New value to add.
            n (int): Weight of the new value (e.g., batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
