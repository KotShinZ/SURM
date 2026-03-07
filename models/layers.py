from typing import Tuple, Optional
from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
import einops
import math

try:
    from flash_attn_interface import flash_attn_func
except ImportError:
    from flash_attn import flash_attn_func

from models.common import trunc_normal_init_


CosSin = Tuple[torch.Tensor, ...]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


def apply_rotary_pos_emb_2d(
    q: torch.Tensor, k: torch.Tensor,
    cos_row: torch.Tensor, sin_row: torch.Tensor,
    cos_col: torch.Tensor, sin_col: torch.Tensor,
):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos_row, sin_row, cos_col, sin_col: [seq_len, head_dim // 2]
    # The first half of head_dim carries row RoPE, the second half carries col RoPE.
    orig_dtype = q.dtype
    q = q.to(cos_row.dtype)
    k = k.to(cos_row.dtype)

    half = q.shape[-1] // 2
    q_row, q_col = q[..., :half], q[..., half:]
    k_row, k_col = k[..., :half], k[..., half:]

    cos_r = cos_row.unsqueeze(-2)  # [seq_len, 1, half_dim]
    sin_r = sin_row.unsqueeze(-2)
    cos_c = cos_col.unsqueeze(-2)
    sin_c = sin_col.unsqueeze(-2)

    q_row = q_row * cos_r + rotate_half(q_row) * sin_r
    k_row = k_row * cos_r + rotate_half(k_row) * sin_r
    q_col = q_col * cos_c + rotate_half(q_col) * sin_c
    k_col = k_col * cos_c + rotate_half(k_col) * sin_c

    q_embed = torch.cat([q_row, q_col], dim=-1)
    k_embed = torch.cat([k_row, k_col], dim=-1)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5)))
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class RotaryEmbedding2D(nn.Module):
    """2D Rotary Position Embeddings for grid-structured inputs (e.g. Sudoku, ARC-AGI).

    The first half of head_dim encodes the row position and the second half
    encodes the column position.  Puzzle-embedding tokens that precede the
    grid in the sequence are assigned position (row=0, col=0) so they
    receive no rotation (cos=1, sin=0).
    """

    def __init__(self, dim: int, grid_height: int, grid_width: int,
                 puzzle_emb_len: int, base: float, device=None):
        super().__init__()

        half_dim = dim // 2  # each spatial axis gets half the head dimensions

        # Frequency bands — same formula as standard RoPE but applied to half_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2, dtype=torch.float32, device=device) / half_dim))

        total_len = puzzle_emb_len + grid_height * grid_width

        # Row / column indices for every token position.
        # Puzzle-embedding prefix tokens get (row=0, col=0).
        row_ids = torch.zeros(total_len, dtype=torch.float32, device=device)
        col_ids = torch.zeros(total_len, dtype=torch.float32, device=device)

        grid_pos = torch.arange(grid_height * grid_width, dtype=torch.float32, device=device)
        row_ids[puzzle_emb_len:] = grid_pos // grid_width
        col_ids[puzzle_emb_len:] = grid_pos % grid_width

        # [total_len, half_dim//2] → [total_len, half_dim]  (same duplication as 1D RoPE)
        row_freqs = torch.outer(row_ids, inv_freq)
        col_freqs = torch.outer(col_ids, inv_freq)

        row_emb = torch.cat([row_freqs, row_freqs], dim=-1)
        col_emb = torch.cat([col_freqs, col_freqs], dim=-1)

        self.cos_row = nn.Buffer(row_emb.cos(), persistent=False)
        self.sin_row = nn.Buffer(row_emb.sin(), persistent=False)
        self.cos_col = nn.Buffer(col_emb.cos(), persistent=False)
        self.sin_col = nn.Buffer(col_emb.sin(), persistent=False)

    def forward(self):
        return self.cos_row, self.sin_row, self.cos_col, self.sin_col

class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False, attn_dropout=0.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.attn_dropout = attn_dropout

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, window_size=-1) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE (1D or 2D)
        if cos_sin is not None:
            if len(cos_sin) == 4:
                # 2D RoPE: (cos_row, sin_row, cos_col, sin_col)
                query, key = apply_rotary_pos_emb_2d(query, key, *cos_sin)
            else:
                # 1D RoPE: (cos, sin)
                cos, sin = cos_sin
                query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # flash attn
        dropout_p = self.attn_dropout if self.training else 0.0
        attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal, window_size=(window_size, window_size), dropout_p=dropout_p)
        if isinstance(attn_output, tuple):  # fa2 and fa3 compatibility
            attn_output = attn_output[0]

        # attn_output: [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float, mlp_dropout: float = 0.0):
        super().__init__()
        
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)
        self.mlp_dropout = nn.Dropout(mlp_dropout)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.mlp_dropout(F.silu(gate) * up))


class ConvSwiGLU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expansion: float,
        conv_kernel: int = 2,
        intermediate_size: Optional[int] = None,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()

        inter = intermediate_size if intermediate_size is not None else _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.inter = inter
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.dwconv = nn.Conv1d(
            in_channels=inter,
            out_channels=inter,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=inter,
            bias=True,
        ).to(dtype=torch.bfloat16)
        # self.dwattn = Attention(
        #     hidden_size=inter,
        #     head_dim=inter // 8,
        #     num_heads=8,
        #     num_key_value_heads=8,
        #     causal=False,
        # )
        self.conv_kernel = conv_kernel

        self.act = nn.SiLU()
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)
        self.mlp_dropout = nn.Dropout(mlp_dropout)

    def forward(self, x: torch.Tensor, timer: Optional[object] = None, prefix: str = ""):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        x_ffn = F.silu(gate) * up
        x_conv = self.dwconv(x_ffn.transpose(1, 2).to(self.dwconv.weight.dtype))
        # x_conv = self.dwattn(cos_sin = None, hidden_states=x_ffn, window_size=self.conv_kernel - 1)
        x_conv = x_conv[..., :up.size(1)]
        x_conv = self.act(x_conv)
        x_conv = x_conv.transpose(1, 2).contiguous()
        x_out = self.down_proj(self.mlp_dropout(x_conv))

        return x_out


class FullyLinearGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = round(expansion * hidden_size)

        self.up_proj = nn.Linear(hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(self.up_proj(x))


class LinearGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(gate + up)


class SiLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size), 256)

        self.up_proj = CastedLinear(hidden_size, inter, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        x = self.up_proj(x)
        x = F.silu(x)
        return self.down_proj(x)


class LinearSwish(nn.Module):
    def __init__(self, hidden_size: int, reverse=False):
        super().__init__()

        self.linear = CastedLinear(hidden_size, hidden_size, bias=False)
        self.reverse = reverse

    def forward(self, x):
        if self.reverse:
            return F.silu(self.linear(x))
        else:
            return self.linear(F.silu(x))


class ReLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size), 256)

        self.up_proj = CastedLinear(hidden_size, inter, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        x = self.up_proj(x)
        x = F.relu(x)
        return self.down_proj(x)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
