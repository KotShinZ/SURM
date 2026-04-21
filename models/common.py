import math

import torch
from torch import nn


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor


def packed_norm_ratio_from_lengths(
    x1: torch.Tensor,
    x2: torch.Tensor,
    lengths: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute per-segment ||x1 - x2|| / (eps + ||x1 + x2|| / 2) for packed sequences."""
    lengths = lengths.to(device=x1.device, dtype=torch.long)
    num_segments = lengths.shape[0]
    segment_ids = torch.repeat_interleave(
        torch.arange(num_segments, device=x1.device, dtype=torch.long),
        lengths,
    )

    diff_sq_per_token = (x1 - x2).to(torch.float32).square().sum(dim=-1)
    sum_sq_per_token = (x1 + x2).to(torch.float32).square().sum(dim=-1)

    diff_norm_sq = torch.zeros((num_segments,), device=x1.device, dtype=torch.float32)
    sum_norm_sq = torch.zeros((num_segments,), device=x1.device, dtype=torch.float32)
    diff_norm_sq.scatter_add_(0, segment_ids, diff_sq_per_token)
    sum_norm_sq.scatter_add_(0, segment_ids, sum_sq_per_token)

    return diff_norm_sq.sqrt() / (eps + sum_norm_sq.sqrt() / 2)
