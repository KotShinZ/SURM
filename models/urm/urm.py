from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass, replace
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from models.common import packed_norm_ratio_from_lengths, trunc_normal_init_
from models.layers import (
    rms_norm,
    ConvSwiGLU,
    Attention,
    RotaryEmbedding,
    RotaryEmbedding2D,
    RotaryEmbedding3D,
    RotaryEmbedding4D,
    CosSin,
    CastedEmbedding,
    CastedLinear,
)
from models.sparse_embedding import CastedSparseEmbedding
from logger import global_logger

@dataclass
class URMCarry:
    current_hidden: torch.Tensor
    steps: Optional[torch.Tensor] = None
    halted: Optional[torch.Tensor] = None
    current_data: Optional[Dict[str, torch.Tensor]] = None


def _normalize_attention_window_sizes(
    attention_window_size: Union[int, List[int]],
    num_layers: int,
    config_name: str,
) -> List[int]:
    if isinstance(attention_window_size, (list, tuple)):
        layer_window_sizes = [int(window_size) for window_size in attention_window_size]
    else:
        layer_window_sizes = [int(attention_window_size)] * num_layers

    if len(layer_window_sizes) != num_layers:
        raise ValueError(
            f"{config_name} must have exactly {num_layers} entries, but got {len(layer_window_sizes)}"
        )

    normalized_window_sizes = []
    for layer_idx, window_size in enumerate(layer_window_sizes):
        if window_size == -1:
            normalized_window_sizes.append(-1)
            continue
        if window_size < 0:
            raise ValueError(
                f"{config_name}[{layer_idx}] must be -1 or a non-negative even integer, but got {window_size}"
            )
        if window_size % 2 != 0:
            raise ValueError(
                f"{config_name}[{layer_idx}] must be even so it can be split symmetrically, but got {window_size}"
            )
        normalized_window_sizes.append(window_size // 2)

    return normalized_window_sizes


class URMConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int
    num_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    grid_depth: int = 0  # Grid depth for 3D RoPE (0 = use 2D/1D RoPE)
    grid_io: int = 0  # Input/output slot axis for 4D RoPE (0 = use 3D/2D/1D RoPE)
    grid_height: int = 0  # Grid height for 2D RoPE (0 = use 1D RoPE)
    grid_width: int = 0   # Grid width  for 2D RoPE (0 = use 1D RoPE)
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0
    attention_type: str = "full"
    attention_window_size: Union[int, List[int]] = -1
    attention_window_size_2d: int = 1
    attention_topk: int = 0
    topk_sparsity: float = 0.0
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    loops: int
    L_cycles: int
    H_cycles: int
    forward_dtype: str = "bfloat16"
    use_act: bool = True
    input_embedding_noise_size: float = 0.0
    input_injection_enabled: bool = True
    noise_size: float = 0.0
    noise_seed: int = 42
    norm_diff_max: float = 0.2
    norm_diff_min: float = 0.1
    diff_L_loss_enabled: bool = False
    halt_norm_in_use_act: bool = False
    patch_io_enabled: bool = False
    patch_height: int = 2
    patch_width: int = 2
    patch_pre_embedding_size: int = 3
    variable_seq_lengths: bool = False
    profile: bool = False
    answer_only: bool = False
    answer_only_context_layers: int = 0


class URMBlock(nn.Module):
    def __init__(self, config: URMConfig, attention_window_size: int) -> None:
        super().__init__()
        puzzle_prefix_len = -(config.puzzle_emb_ndim // -config.hidden_size)
        attention_type = config.attention_type.lower()
        if attention_type in {"full", "swa"}:
            attention_type = "full" if attention_window_size == -1 else "swa"
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
            attn_dropout=config.attn_dropout,
            attention_type=attention_type,
            attention_window_size=attention_window_size,
            attention_window_size_2d=config.attention_window_size_2d,
            attention_topk=config.attention_topk,
            grid_height=config.grid_height,
            grid_width=config.grid_width,
            prefix_seq_len=puzzle_prefix_len,
            topk_sparsity=config.topk_sparsity,
        )
        self.mlp = ConvSwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
            mlp_dropout=config.mlp_dropout,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(
        self,
        cos_sin: CosSin,
        hidden_states: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_output = self.self_attn(
            cos_sin=cos_sin,
            hidden_states=hidden_states,
            window_size=-1,
            sequence_lengths=sequence_lengths,
        )
        hidden_states = rms_norm(hidden_states + attn_output, variance_epsilon=self.norm_eps)
        mlp_output = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_output, variance_epsilon=self.norm_eps)
        return hidden_states

    def forward_packed(
        self,
        cos_sin: CosSin,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        attn_output = self.self_attn.forward_packed(
            cos_sin=cos_sin,
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            window_size=-1,
        )
        hidden_states = rms_norm(hidden_states + attn_output, variance_epsilon=self.norm_eps)
        mlp_output = self.mlp.forward_packed(hidden_states, cu_seqlens)
        hidden_states = rms_norm(hidden_states + mlp_output, variance_epsilon=self.norm_eps)
        return hidden_states

    def forward_cross_packed(
        self,
        query_cos_sin: CosSin,
        key_value_cos_sin: CosSin,
        query_states: torch.Tensor,
        key_value_states: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        attn_output = self.self_attn.forward_cross_packed(
            query_cos_sin=query_cos_sin,
            key_value_cos_sin=key_value_cos_sin,
            query_states=query_states,
            key_value_states=key_value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
        query_states = rms_norm(query_states + attn_output, variance_epsilon=self.norm_eps)
        mlp_output = self.mlp.forward_packed(query_states, cu_seqlens_q)
        query_states = rms_norm(query_states + mlp_output, variance_epsilon=self.norm_eps)
        return query_states


class URM_Inner(nn.Module):
    def __init__(self, config: URMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.patch_area = self.config.patch_height * self.config.patch_width
        self.inner_seq_len = self.config.seq_len
        self.inner_grid_depth = self.config.grid_depth
        self.inner_grid_io = self.config.grid_io
        self.inner_grid_height = self.config.grid_height
        self.inner_grid_width = self.config.grid_width
        self.padded_grid_height = self.config.grid_height
        self.padded_grid_width = self.config.grid_width
        self.padded_seq_len = self.config.seq_len

        if self.config.variable_seq_lengths and self.config.patch_io_enabled:
            raise ValueError("variable_seq_lengths is not supported with patch_io_enabled")

        if self.config.patch_io_enabled:
            if self.config.grid_height > 0 or self.config.grid_width > 0:
                self.inner_grid_height = -(self.config.grid_height // -self.config.patch_height)
                self.inner_grid_width = -(self.config.grid_width // -self.config.patch_width)
                self.inner_seq_len = self.inner_grid_height * self.inner_grid_width
                self.padded_grid_height = self.inner_grid_height * self.config.patch_height
                self.padded_grid_width = self.inner_grid_width * self.config.patch_width
            else:
                self.inner_seq_len = -(self.config.seq_len // -self.patch_area)
                self.padded_seq_len = self.inner_seq_len * self.patch_area

        config_updates = dict(
            seq_len=self.inner_seq_len,
            grid_depth=self.inner_grid_depth,
            grid_io=self.inner_grid_io,
            grid_height=self.inner_grid_height,
            grid_width=self.inner_grid_width,
        )
        self.inner_config = (
            self.config.model_copy(update=config_updates)
            if hasattr(self.config, "model_copy")
            else self.config.copy(update=config_updates)
        )

        if self.config.patch_io_enabled:
            patch_dim = self.patch_area * self.config.patch_pre_embedding_size
            self.pre_embedding = CastedLinear(self.config.vocab_size, self.config.patch_pre_embedding_size, bias=False)
            self.embed_tokens = CastedLinear(patch_dim, self.config.hidden_size, bias=False)
            self.lm_head = CastedLinear(self.config.hidden_size, patch_dim, bias=False)
            self.post_head = CastedLinear(self.config.patch_pre_embedding_size, self.config.vocab_size, bias=False)
        else:
            self.embed_tokens = CastedEmbedding(
                self.config.vocab_size,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
            self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)

        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        if (
            self.inner_config.grid_depth > 0
            and self.inner_config.grid_io > 0
            and self.inner_config.grid_height > 0
            and self.inner_config.grid_width > 0
        ):
            self.rotary_emb = RotaryEmbedding4D(
                dim=self.inner_config.hidden_size // self.inner_config.num_heads,
                grid_depth=self.inner_config.grid_depth,
                grid_io=self.inner_config.grid_io,
                grid_height=self.inner_config.grid_height,
                grid_width=self.inner_config.grid_width,
                puzzle_emb_len=self.puzzle_emb_len,
                base=self.inner_config.rope_theta,
            )
        elif (
            self.inner_config.grid_depth > 0
            and self.inner_config.grid_height > 0
            and self.inner_config.grid_width > 0
        ):
            self.rotary_emb = RotaryEmbedding3D(
                dim=self.inner_config.hidden_size // self.inner_config.num_heads,
                grid_depth=self.inner_config.grid_depth,
                grid_height=self.inner_config.grid_height,
                grid_width=self.inner_config.grid_width,
                puzzle_emb_len=self.puzzle_emb_len,
                base=self.inner_config.rope_theta,
            )
        elif self.inner_config.grid_height > 0 and self.inner_config.grid_width > 0:
            self.rotary_emb = RotaryEmbedding2D(
                dim=self.inner_config.hidden_size // self.inner_config.num_heads,
                grid_height=self.inner_config.grid_height,
                grid_width=self.inner_config.grid_width,
                puzzle_emb_len=self.puzzle_emb_len,
                base=self.inner_config.rope_theta,
            )
        else:
            self.rotary_emb = RotaryEmbedding(
                dim=self.inner_config.hidden_size // self.inner_config.num_heads,
                max_position_embeddings=self.inner_seq_len + self.puzzle_emb_len,
                base=self.inner_config.rope_theta,
            )

        self.layer_attention_window_sizes = _normalize_attention_window_sizes(
            self.inner_config.attention_window_size,
            self.config.num_layers,
            "URMConfig.attention_window_size",
        )
        self.layers = nn.ModuleList(
            [
                URMBlock(self.inner_config, attention_window_size=self.layer_attention_window_sizes[layer_idx])
                for layer_idx in range(self.config.num_layers)
            ]
        )
        if self.config.answer_only_context_layers < 0:
            raise ValueError(
                f"answer_only_context_layers must be >= 0, got {self.config.answer_only_context_layers}"
            )
        self.context_layers = nn.ModuleList(
            [
                URMBlock(self.inner_config, attention_window_size=-1)
                for _ in range(self.config.answer_only_context_layers)
            ]
        )

        self.init_hidden = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)
            
        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = torch.Generator(device=generator_device).manual_seed(self.config.noise_seed)

    def _one_hot_inputs(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim == 3:
            if input.shape[-1] != self.config.vocab_size:
                raise ValueError("one-hot inputs must have vocab_size as the last dimension")
            return input.to(self.forward_dtype)
        return F.one_hot(input.to(torch.long), num_classes=self.config.vocab_size).to(self.forward_dtype)

    def _patchify(self, pixels: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = pixels.shape
        if seq_len != self.config.seq_len:
            raise ValueError("patch IO input sequence length must match config.seq_len")

        if self.config.grid_height > 0 and self.config.grid_width > 0:
            pixels = pixels.view(batch_size, self.config.grid_height, self.config.grid_width, channels)
            if self.padded_grid_height != self.config.grid_height or self.padded_grid_width != self.config.grid_width:
                pixels = F.pad(
                    pixels,(0,0,0,self.padded_grid_width - self.config.grid_width,
                        0,self.padded_grid_height - self.config.grid_height,),
                )
            patches = pixels.reshape(
                batch_size,
                self.inner_grid_height,
                self.config.patch_height,
                self.inner_grid_width,
                self.config.patch_width,
                channels,
            )
            return patches.permute(0, 1, 3, 2, 4, 5).reshape(
                batch_size,
                self.inner_seq_len,
                self.patch_area * channels,
            )

        if self.padded_seq_len != self.config.seq_len:
            pixels = F.pad(pixels, (0, 0, 0, self.padded_seq_len - self.config.seq_len))
        return pixels.reshape(batch_size, self.inner_seq_len, self.patch_area * channels)

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, patch_dim = patches.shape
        channels = self.config.patch_pre_embedding_size
        if seq_len != self.inner_seq_len or patch_dim != self.patch_area * channels:
            raise ValueError("patch IO head output has an unexpected shape")

        if self.config.grid_height > 0 and self.config.grid_width > 0:
            pixels = patches.reshape(
                batch_size,
                self.inner_grid_height,
                self.inner_grid_width,
                self.config.patch_height,
                self.config.patch_width,
                channels,
            )
            pixels = pixels.permute(0, 1, 3, 2, 4, 5).reshape(
                batch_size,
                self.padded_grid_height,
                self.padded_grid_width,
                channels,
            )
            return pixels[:, : self.config.grid_height, : self.config.grid_width].reshape(
                batch_size,
                self.config.seq_len,
                channels,
            )

        return patches.reshape(batch_size, self.padded_seq_len, channels)[:, : self.config.seq_len]

    def _output_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.lm_head(hidden_states)[:, self.puzzle_emb_len:]
        if self.config.patch_io_enabled:
            output = self.post_head(self._unpatchify(output))
        return output

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        if self.config.patch_io_enabled:
            pixels = self.pre_embedding(self._one_hot_inputs(input))
            embedding = self.embed_tokens(self._patchify(pixels))
        else:
            embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2,
            )
        embedding = self.embed_scale * embedding

        if self.config.input_embedding_noise_size > 0:
            noise = torch.randn(
                embedding.shape,
                generator=self.generator,
                dtype=embedding.dtype,
                device=embedding.device,
                layout=embedding.layout,
            )
            embedding = embedding + noise * self.config.input_embedding_noise_size

        return embedding

    def _packed_lengths(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["seq_lengths"].to(device=batch["inputs"].device, dtype=torch.long)

    def _pack_rotary_axes(
        self,
        axes: CosSin,
        batch: Dict[str, torch.Tensor],
        token_indices: torch.Tensor,
    ) -> CosSin:
        if self.puzzle_emb_len == 0:
            return axes

        total_len = token_indices.shape[0] + batch["puzzle_identifiers"].shape[0] * self.puzzle_emb_len
        prefix_indices = self._packed_prefix_indices(batch)
        packed_axes = []
        for cos, sin in zip(axes[::2], axes[1::2]):
            flat_cos = cos.new_empty((total_len, cos.shape[-1]))
            flat_sin = sin.new_empty((total_len, sin.shape[-1]))
            flat_cos[token_indices] = cos
            flat_sin[token_indices] = sin
            flat_cos[prefix_indices] = 1
            flat_sin[prefix_indices] = 0
            packed_axes.extend([flat_cos, flat_sin])
        return tuple(packed_axes)

    @torch.compiler.disable
    def _packed_cu_seqlens(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, int]:
        lengths = self._packed_lengths(batch) + self.puzzle_emb_len
        cu_seqlens = F.pad(torch.cumsum(lengths, dim=0), (1, 0)).to(torch.int32)
        max_seqlen = int(lengths.max().item()) if lengths.numel() else 0
        return cu_seqlens, max_seqlen

    def _packed_data_token_indices(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        num_tokens = batch["inputs"].shape[0]
        if self.puzzle_emb_len == 0:
            return torch.arange(num_tokens, device=batch["inputs"].device, dtype=torch.long)

        lengths = self._packed_lengths(batch)
        seq_ids = torch.repeat_interleave(
            torch.arange(lengths.shape[0], device=batch["inputs"].device, dtype=torch.long),
            lengths,
        )
        token_positions = torch.arange(num_tokens, device=batch["inputs"].device, dtype=torch.long)
        return token_positions + seq_ids * self.puzzle_emb_len + self.puzzle_emb_len

    def _packed_prefix_indices(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = batch["puzzle_identifiers"].shape[0]
        if self.puzzle_emb_len == 0 or batch_size == 0:
            return torch.empty((0,), device=batch["inputs"].device, dtype=torch.long)

        data_offsets = batch["seq_offsets"][:-1].to(device=batch["inputs"].device, dtype=torch.long)
        seq_offsets = data_offsets + torch.arange(batch_size, device=batch["inputs"].device, dtype=torch.long) * self.puzzle_emb_len
        prefix_offsets = torch.arange(self.puzzle_emb_len, device=batch["inputs"].device, dtype=torch.long)
        return (seq_offsets[:, None] + prefix_offsets[None, :]).reshape(-1)

    def _input_embeddings_packed(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.patch_io_enabled:
            raise ValueError("Packed variable-length data is not supported with patch_io_enabled")

        token_embedding = self.embed_tokens(batch["inputs"].to(torch.int32))
        token_indices = self._packed_data_token_indices(batch)

        if self.puzzle_emb_len > 0:
            puzzle_embedding = self.puzzle_emb(batch["puzzle_identifiers"])
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            puzzle_embedding = puzzle_embedding.view(-1, self.config.hidden_size)

            total_len = token_embedding.shape[0] + batch["puzzle_identifiers"].shape[0] * self.puzzle_emb_len
            embedding = token_embedding.new_empty((total_len, self.config.hidden_size))
            embedding[token_indices] = token_embedding
            embedding[self._packed_prefix_indices(batch)] = puzzle_embedding
        else:
            embedding = token_embedding

        embedding = self.embed_scale * embedding

        if self.config.input_embedding_noise_size > 0:
            noise = torch.randn(
                embedding.shape,
                generator=self.generator,
                dtype=embedding.dtype,
                device=embedding.device,
                layout=embedding.layout,
            )
            embedding = embedding + noise * self.config.input_embedding_noise_size

        return embedding, token_indices

    def _rotary_cos_sin_packed(self, batch: Dict[str, torch.Tensor], token_indices: torch.Tensor) -> CosSin:
        if isinstance(self.rotary_emb, RotaryEmbedding4D):
            position_ids = batch["position_ids"].to(device=batch["inputs"].device, dtype=torch.long)
            return self._pack_rotary_axes(self.rotary_emb.lookup(position_ids), batch, token_indices)

        if isinstance(self.rotary_emb, RotaryEmbedding3D):
            position_ids = batch["position_ids"].to(device=batch["inputs"].device, dtype=torch.long)
            return self._pack_rotary_axes(self.rotary_emb.lookup(position_ids), batch, token_indices)

        if isinstance(self.rotary_emb, RotaryEmbedding2D):
            position_ids = batch["position_ids"].to(device=batch["inputs"].device, dtype=torch.long)
            data_flat_ids = self.puzzle_emb_len + position_ids[:, 0] * self.rotary_emb.grid_width + position_ids[:, 1]
            if self.puzzle_emb_len > 0:
                flat_ids = data_flat_ids.new_empty((token_indices.shape[0] + batch["puzzle_identifiers"].shape[0] * self.puzzle_emb_len,))
                flat_ids[token_indices] = data_flat_ids
                prefix_ids = torch.arange(self.puzzle_emb_len, device=batch["inputs"].device, dtype=torch.long)
                flat_ids[self._packed_prefix_indices(batch)] = prefix_ids.repeat(batch["puzzle_identifiers"].shape[0])
            else:
                flat_ids = data_flat_ids

            return (
                self.rotary_emb.cos_row[flat_ids],
                self.rotary_emb.sin_row[flat_ids],
                self.rotary_emb.cos_col[flat_ids],
                self.rotary_emb.sin_col[flat_ids],
            )

        lengths = self._packed_lengths(batch)
        seq_ids = torch.repeat_interleave(
            torch.arange(lengths.shape[0], device=batch["inputs"].device, dtype=torch.long),
            lengths,
        )
        data_offsets = batch["seq_offsets"][:-1].to(device=batch["inputs"].device, dtype=torch.long)
        token_positions = torch.arange(batch["inputs"].shape[0], device=batch["inputs"].device, dtype=torch.long)
        data_pos_ids = self.puzzle_emb_len + token_positions - data_offsets[seq_ids]
        if self.puzzle_emb_len > 0:
            flat_ids = data_pos_ids.new_empty((token_indices.shape[0] + batch["puzzle_identifiers"].shape[0] * self.puzzle_emb_len,))
            flat_ids[token_indices] = data_pos_ids
            prefix_ids = torch.arange(self.puzzle_emb_len, device=batch["inputs"].device, dtype=torch.long)
            flat_ids[self._packed_prefix_indices(batch)] = prefix_ids.repeat(batch["puzzle_identifiers"].shape[0])
        else:
            flat_ids = data_pos_ids

        cos, sin = self.rotary_emb()
        return cos[flat_ids], sin[flat_ids]

    def _slice_packed_cos_sin(self, cos_sin: CosSin, indices: torch.Tensor) -> CosSin:
        return tuple(axis[indices] for axis in cos_sin)

    def _packed_answer_mask(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "answer_mask" in batch:
            return batch["answer_mask"].to(device=batch["inputs"].device, dtype=torch.bool)
        return batch["labels"] != -100

    def _packed_answer_metadata(
        self,
        batch: Dict[str, torch.Tensor],
        token_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        lengths = self._packed_lengths(batch)
        answer_data_mask = self._packed_answer_mask(batch)
        answer_lengths = torch.zeros(
            (lengths.shape[0],),
            device=batch["inputs"].device,
            dtype=torch.long,
        )
        if lengths.numel() > 0:
            segment_ids = torch.repeat_interleave(
                torch.arange(lengths.shape[0], device=batch["inputs"].device, dtype=torch.long),
                lengths,
            )
            answer_lengths.scatter_add_(0, segment_ids, answer_data_mask.to(torch.long))

        answer_indices = token_indices[answer_data_mask]
        if "label_seq_lengths" in batch:
            answer_labels = batch["labels"]
            answer_lengths = batch["label_seq_lengths"].to(device=batch["inputs"].device, dtype=torch.long)
        else:
            answer_labels = batch["labels"][answer_data_mask]
        full_lengths = lengths + self.puzzle_emb_len
        context_lengths = full_lengths - answer_lengths
        cu_answer = F.pad(torch.cumsum(answer_lengths.to(torch.int32), dim=0), (1, 0))
        cu_context = F.pad(torch.cumsum(context_lengths.to(torch.int32), dim=0), (1, 0))
        max_answer_len = int(answer_lengths.max().item()) if answer_lengths.numel() else 0
        max_context_len = int(context_lengths.max().item()) if context_lengths.numel() else 0
        return (
            answer_data_mask,
            answer_indices,
            answer_labels,
            answer_lengths,
            cu_answer,
            cu_context,
            max_answer_len,
            max_context_len,
        )

    def _packed_context_indices(
        self,
        hidden_states: torch.Tensor,
        answer_indices: torch.Tensor,
    ) -> torch.Tensor:
        context_mask = torch.ones(
            (hidden_states.shape[0],),
            device=hidden_states.device,
            dtype=torch.bool,
        )
        context_mask[answer_indices] = False
        return torch.nonzero(context_mask, as_tuple=False).flatten()

    def empty_carry(self, batch_size: int, seq_len: Optional[int] = None) -> URMCarry:
        carry_seq_len = self.inner_seq_len if seq_len is None else seq_len
        return URMCarry(
            current_hidden=torch.empty(
                batch_size,
                carry_seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: URMCarry) -> URMCarry:
        new_hidden = torch.where(
            reset_flag.view(-1, 1, 1),
            self.init_hidden,
            carry.current_hidden
        )
        return replace(carry, current_hidden=new_hidden)

    def _rotary_cos_sin(self, batch: Dict[str, torch.Tensor]):
        if "position_ids" in batch and isinstance(
            self.rotary_emb,
            (RotaryEmbedding2D, RotaryEmbedding3D, RotaryEmbedding4D),
        ):
            return self.rotary_emb(batch["position_ids"], prefix_seq_len=self.puzzle_emb_len)
        if isinstance(self.rotary_emb, (RotaryEmbedding3D, RotaryEmbedding4D)):
            raise ValueError(f"{type(self.rotary_emb).__name__} requires explicit position_ids in the batch.")
        return self.rotary_emb()

    def _sequence_lengths(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.config.variable_seq_lengths or "seq_lengths" not in batch:
            return None
        return batch["seq_lengths"].to(torch.int32) + self.puzzle_emb_len

    def _inject_inputs(self, hidden_states: torch.Tensor, input_embeddings: torch.Tensor) -> torch.Tensor:
        if not self.config.input_injection_enabled:
            return hidden_states
        return hidden_states + input_embeddings

    def forward(
        self,
        carry: URMCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[URMCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        if self.config.variable_seq_lengths:
            return self.forward_packed(carry, batch)

        seq_info = dict(
            cos_sin=self._rotary_cos_sin(batch),
            sequence_lengths=self._sequence_lengths(batch),
        )
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        hidden_states = carry.current_hidden
        if self.config.H_cycles > 1:
            with torch.no_grad():
                for _ in range(self.config.H_cycles - 1):
                    for _ in range(self.config.L_cycles):
                        hidden_states = self._inject_inputs(hidden_states, input_embeddings)
                        for layer in self.layers:
                            hidden_states = layer(hidden_states=hidden_states, **seq_info)
                        if self.config.noise_size > 0:
                            noise = torch.randn(
                                hidden_states.shape,
                                generator=self.generator,
                                dtype=hidden_states.dtype,
                                device=hidden_states.device,
                                layout=hidden_states.layout
                            )
                            hidden_states = hidden_states + noise * self.config.noise_size

        # Gradient norm logging for unrolled layers
        _log_grads = global_logger.is_log and self.training
        if _log_grads:
            _grad_norms = {}
            _total_unrolled = self.config.L_cycles * len(self.layers)
            def _make_grad_hook(idx, container, total):
                def hook(grad):
                    container[idx] = grad.detach().norm().item()
                    if len(container) == total:
                        norm_tensor = torch.tensor([container[i] for i in range(total)])
                        global_logger.store("grad_norm_per_layer", norm_tensor)
                return hook
        
        _unrolled_idx = 0
        diff_L = torch.zeros_like(hidden_states) if self.config.diff_L_loss_enabled else None
        for _ in range(self.config.L_cycles):
            hidden_states = self._inject_inputs(hidden_states, input_embeddings)
            for layer in self.layers:
                pre_hidden_states = hidden_states
                hidden_states = layer(hidden_states=pre_hidden_states, **seq_info)
                if diff_L is not None:
                    diff_L = diff_L + torch.abs(hidden_states - pre_hidden_states)
                if self.config.noise_size > 0:
                    noise = torch.randn(
                        hidden_states.shape,
                        generator=self.generator,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                        layout=hidden_states.layout
                    )
                    hidden_states = hidden_states + noise * self.config.noise_size
                if _log_grads:
                    hidden_states.register_hook(_make_grad_hook(_unrolled_idx, _grad_norms, _total_unrolled))
                    _unrolled_idx += 1

        new_carry = replace(carry, current_hidden=hidden_states.detach())
        output = self._output_logits(hidden_states)
        q_logits = self.q_head(hidden_states[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), diff_L

    def forward_packed(
        self,
        carry: URMCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[URMCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        if self.config.answer_only:
            return self.forward_answer_only_packed(carry, batch)

        input_embeddings, token_indices = self._input_embeddings_packed(batch)
        cu_seqlens, max_seqlen = self._packed_cu_seqlens(batch)
        cos_sin = self._rotary_cos_sin_packed(batch, token_indices)

        hidden_states = carry.current_hidden
        if hidden_states.shape[0] != input_embeddings.shape[0]:
            raise RuntimeError(
                f"Packed carry/input length mismatch: carry={hidden_states.shape[0]} input={input_embeddings.shape[0]}"
            )

        if self.config.H_cycles > 1:
            with torch.no_grad():
                for _ in range(self.config.H_cycles - 1):
                    for _ in range(self.config.L_cycles):
                        hidden_states = self._inject_inputs(hidden_states, input_embeddings)
                        for layer in self.layers:
                            hidden_states = layer.forward_packed(
                                cos_sin=cos_sin,
                                hidden_states=hidden_states,
                                cu_seqlens=cu_seqlens,
                                max_seqlen=max_seqlen,
                            )
                        if self.config.noise_size > 0:
                            noise = torch.randn(
                                hidden_states.shape,
                                generator=self.generator,
                                dtype=hidden_states.dtype,
                                device=hidden_states.device,
                                layout=hidden_states.layout
                            )
                            hidden_states = hidden_states + noise * self.config.noise_size

        _log_grads = global_logger.is_log and self.training
        if _log_grads:
            _grad_norms = {}
            _total_unrolled = self.config.L_cycles * len(self.layers)
            def _make_grad_hook(idx, container, total):
                def hook(grad):
                    container[idx] = grad.detach().norm().item()
                    if len(container) == total:
                        norm_tensor = torch.tensor([container[i] for i in range(total)])
                        global_logger.store("grad_norm_per_layer", norm_tensor)
                return hook

        _unrolled_idx = 0
        diff_L = torch.zeros_like(hidden_states) if self.config.diff_L_loss_enabled else None
        for _ in range(self.config.L_cycles):
            hidden_states = self._inject_inputs(hidden_states, input_embeddings)
            for layer in self.layers:
                pre_hidden_states = hidden_states
                hidden_states = layer.forward_packed(
                    cos_sin=cos_sin,
                    hidden_states=pre_hidden_states,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
                if diff_L is not None:
                    diff_L = diff_L + torch.abs(hidden_states - pre_hidden_states)
                if self.config.noise_size > 0:
                    noise = torch.randn(
                        hidden_states.shape,
                        generator=self.generator,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                        layout=hidden_states.layout
                    )
                    hidden_states = hidden_states + noise * self.config.noise_size
                if _log_grads:
                    hidden_states.register_hook(_make_grad_hook(_unrolled_idx, _grad_norms, _total_unrolled))
                    _unrolled_idx += 1

        new_carry = replace(carry, current_hidden=hidden_states.detach())
        output = self.lm_head(hidden_states[token_indices])
        q_logits = self.q_head(hidden_states[cu_seqlens[:-1].to(torch.long)]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), diff_L

    def forward_answer_only_packed(
        self,
        carry: URMCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[URMCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        if not self.config.variable_seq_lengths:
            raise ValueError("answer_only is currently implemented for packed variable-length batches only.")

        input_embeddings, token_indices = self._input_embeddings_packed(batch)
        cu_seqlens, max_seqlen = self._packed_cu_seqlens(batch)
        cos_sin = self._rotary_cos_sin_packed(batch, token_indices)
        (
            _answer_data_mask,
            answer_indices,
            _answer_labels,
            _answer_lengths,
            cu_answer,
            cu_context,
            max_answer_len,
            max_context_len,
        ) = self._packed_answer_metadata(batch, token_indices)

        hidden_states = carry.current_hidden
        if hidden_states.shape[0] != input_embeddings.shape[0]:
            raise RuntimeError(
                f"Packed carry/input length mismatch: carry={hidden_states.shape[0]} input={input_embeddings.shape[0]}"
            )
        if answer_indices.numel() == 0:
            new_carry = replace(carry, current_hidden=hidden_states.detach())
            output = self.lm_head(hidden_states.new_empty((0, hidden_states.shape[-1])))
            q_logits = self.q_head(hidden_states[cu_seqlens[:-1].to(torch.long)]).to(torch.float32)
            return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), None

        context_indices = self._packed_context_indices(hidden_states, answer_indices)
        context_states = self._inject_inputs(
            hidden_states[context_indices],
            input_embeddings[context_indices],
        )
        context_cos_sin = self._slice_packed_cos_sin(cos_sin, context_indices)
        for layer in self.context_layers:
            context_states = layer.forward_packed(
                cos_sin=context_cos_sin,
                hidden_states=context_states,
                cu_seqlens=cu_context,
                max_seqlen=max_context_len,
            )

        def run_answer_layers(states: torch.Tensor) -> torch.Tensor:
            answer_cos_sin = self._slice_packed_cos_sin(cos_sin, answer_indices)
            for _ in range(self.config.L_cycles):
                states = states.clone()
                states[answer_indices] = self._inject_inputs(
                    states[answer_indices],
                    input_embeddings[answer_indices],
                )
                for layer in self.layers:
                    kv_states = states.clone()
                    kv_states[context_indices] = context_states
                    answer_states = states[answer_indices]
                    pre_answer_states = answer_states
                    answer_states = layer.forward_cross_packed(
                        query_cos_sin=answer_cos_sin,
                        key_value_cos_sin=cos_sin,
                        query_states=pre_answer_states,
                        key_value_states=kv_states,
                        cu_seqlens_q=cu_answer,
                        cu_seqlens_k=cu_seqlens,
                        max_seqlen_q=max_answer_len,
                        max_seqlen_k=max_seqlen,
                    )
                    if self.config.noise_size > 0:
                        noise = torch.randn(
                            answer_states.shape,
                            generator=self.generator,
                            dtype=answer_states.dtype,
                            device=answer_states.device,
                            layout=answer_states.layout,
                        )
                        answer_states = answer_states + noise * self.config.noise_size
                    states = states.clone()
                    states[answer_indices] = answer_states
            return states

        if self.config.H_cycles > 1:
            with torch.no_grad():
                for _ in range(self.config.H_cycles - 1):
                    hidden_states = run_answer_layers(hidden_states)

        hidden_states = run_answer_layers(hidden_states)
        new_carry = replace(carry, current_hidden=hidden_states.detach())
        output = self.lm_head(hidden_states[answer_indices])
        q_logits = self.q_head(hidden_states[cu_seqlens[:-1].to(torch.long)]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), None


class URM(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = URMConfig(**config_dict)
        self.inner = URM_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> URMCarry:
        batch_size = batch["puzzle_identifiers"].shape[0] if self.config.variable_seq_lengths else batch["inputs"].shape[0]
        if self.config.variable_seq_lengths:
            return URMCarry(
                current_hidden=torch.empty(
                    0,
                    self.config.hidden_size,
                    dtype=self.inner.forward_dtype,
                    device=batch["inputs"].device,
                ),
                steps=torch.zeros((batch_size,), dtype=torch.int32, device=batch["inputs"].device),
                halted=torch.ones((batch_size,), dtype=torch.bool, device=batch["inputs"].device),
                current_data={k: torch.empty_like(v) for k, v in batch.items()},
            )

        seq_len = batch["inputs"].shape[1] if self.config.variable_seq_lengths else None
        base = self.inner.empty_carry(batch_size, seq_len=seq_len)
        return URMCarry(
            current_hidden=base.current_hidden,
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    @torch.compiler.disable
    def _merge_packed_current_data(self, carry: URMCarry, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert carry.current_data is not None

        halted = carry.halted.detach().cpu().tolist()
        batch_offsets = batch["seq_offsets"].detach().cpu().tolist()
        current_offsets = carry.current_data["seq_offsets"].detach().cpu().tolist()
        batch_size = len(halted)

        token_keys = [
            key for key in ("inputs", "position_ids", "source_inputs", "answer_mask")
            if key in batch and key in carry.current_data
        ]
        label_token_keys = [
            key for key in ("labels",)
            if key in batch and key in carry.current_data
        ]
        token_chunks = {key: [] for key in token_keys}
        label_token_chunks = {key: [] for key in label_token_keys}
        lengths = []
        label_lengths = []
        puzzle_identifiers = []
        seq_shapes = []
        label_seq_shapes = []

        for idx in range(batch_size):
            use_batch = bool(halted[idx])
            source = batch if use_batch else carry.current_data
            offsets = batch_offsets if use_batch else current_offsets
            label_offsets = source.get("label_seq_offsets", source["seq_offsets"]).detach().cpu().tolist()
            start, end = int(offsets[idx]), int(offsets[idx + 1])
            label_start, label_end = int(label_offsets[idx]), int(label_offsets[idx + 1])

            for key in token_keys:
                token_chunks[key].append(source[key][start:end])
            for key in label_token_keys:
                label_token_chunks[key].append(source[key][label_start:label_end])
            lengths.append(source["seq_lengths"][idx])
            label_lengths.append(source.get("label_seq_lengths", source["seq_lengths"])[idx])
            puzzle_identifiers.append(source["puzzle_identifiers"][idx])
            if "seq_shapes" in source:
                seq_shapes.append(source["seq_shapes"][idx])
            if "label_seq_shapes" in source:
                label_seq_shapes.append(source["label_seq_shapes"][idx])

        device = batch["inputs"].device
        merged: Dict[str, torch.Tensor] = {}
        for key, chunks in token_chunks.items():
            merged[key] = torch.cat(chunks, dim=0) if chunks else batch[key].new_empty((0,) + batch[key].shape[1:])
        for key, chunks in label_token_chunks.items():
            merged[key] = torch.cat(chunks, dim=0) if chunks else batch[key].new_empty((0,) + batch[key].shape[1:])

        if lengths:
            merged["seq_lengths"] = torch.stack(lengths).to(device=device, dtype=torch.int32)
            merged["label_seq_lengths"] = torch.stack(label_lengths).to(device=device, dtype=torch.int32)
            merged["puzzle_identifiers"] = torch.stack(puzzle_identifiers).to(device=device, dtype=batch["puzzle_identifiers"].dtype)
            if seq_shapes:
                merged["seq_shapes"] = torch.stack(seq_shapes).to(device=device, dtype=batch["seq_shapes"].dtype)
            if label_seq_shapes:
                merged["label_seq_shapes"] = torch.stack(label_seq_shapes).to(device=device, dtype=batch["label_seq_shapes"].dtype)
        else:
            merged["seq_lengths"] = batch["seq_lengths"].new_empty((0,))
            merged["label_seq_lengths"] = batch.get("label_seq_lengths", batch["seq_lengths"]).new_empty((0,))
            merged["puzzle_identifiers"] = batch["puzzle_identifiers"].new_empty((0,))

        merged["seq_offsets"] = F.pad(torch.cumsum(merged["seq_lengths"].to(torch.int32), dim=0), (1, 0))
        merged["label_seq_offsets"] = F.pad(torch.cumsum(merged["label_seq_lengths"].to(torch.int32), dim=0), (1, 0))
        return merged

    @torch.compiler.disable
    def _reset_packed_carry(self, reset_flag: torch.Tensor, carry: URMCarry, current_data: Dict[str, torch.Tensor]) -> URMCarry:
        assert carry.current_data is not None

        reset = reset_flag.detach().cpu().tolist()
        current_lengths = current_data["seq_lengths"].detach().cpu().tolist()
        old_hidden_lengths = (carry.current_data["seq_lengths"] + self.inner.puzzle_emb_len).detach().cpu().tolist()
        old_hidden_offsets = [0]
        for length in old_hidden_lengths:
            old_hidden_offsets.append(old_hidden_offsets[-1] + int(length))

        hidden_chunks = []
        for idx, should_reset in enumerate(reset):
            new_len = int(current_lengths[idx]) + self.inner.puzzle_emb_len
            if should_reset:
                hidden_chunks.append(self.inner.init_hidden.expand(new_len, -1))
            else:
                hidden_chunks.append(carry.current_hidden[old_hidden_offsets[idx]:old_hidden_offsets[idx + 1]])

        if hidden_chunks:
            current_hidden = torch.cat(hidden_chunks, dim=0)
        else:
            current_hidden = torch.empty(
                0,
                self.config.hidden_size,
                dtype=self.inner.forward_dtype,
                device=current_data["inputs"].device,
            )

        return replace(carry, current_hidden=current_hidden, current_data=current_data)
        
    def _pad_to_seq_len(self, key: str, value: torch.Tensor, target_seq_len: int) -> torch.Tensor:
        if value.ndim < 2 or value.shape[1] == target_seq_len:
            return value
        if value.shape[1] > target_seq_len:
            return value[:, :target_seq_len]

        pad_shape = list(value.shape)
        pad_shape[1] = target_seq_len - value.shape[1]
        pad_value = -100 if key == "labels" else 0
        return torch.cat(
            [value, torch.full(pad_shape, pad_value, dtype=value.dtype, device=value.device)],
            dim=1,
        )

    def _align_variable_batch(self, carry: URMCarry, batch: Dict[str, torch.Tensor]) -> Tuple[URMCarry, Dict[str, torch.Tensor]]:
        if not self.config.variable_seq_lengths or carry.current_data is None:
            return carry, batch

        if bool(carry.halted.all().item()):
            target_seq_len = batch["inputs"].shape[1]
        else:
            target_seq_len = max(carry.current_data["inputs"].shape[1], batch["inputs"].shape[1])
        target_hidden_len = target_seq_len + self.inner.puzzle_emb_len

        aligned_batch = {
            k: self._pad_to_seq_len(k, v, target_seq_len)
            for k, v in batch.items()
        }
        aligned_current_data = {
            k: self._pad_to_seq_len(k, v, target_seq_len)
            for k, v in carry.current_data.items()
        }
        aligned_hidden = carry.current_hidden
        if aligned_hidden.shape[1] != target_hidden_len:
            if aligned_hidden.shape[1] > target_hidden_len:
                aligned_hidden = aligned_hidden[:, :target_hidden_len]
            else:
                pad_shape = list(aligned_hidden.shape)
                pad_shape[1] = target_hidden_len - aligned_hidden.shape[1]
                aligned_hidden = torch.cat(
                    [
                        aligned_hidden,
                        torch.zeros(pad_shape, dtype=aligned_hidden.dtype, device=aligned_hidden.device),
                    ],
                    dim=1,
                )

        return replace(carry, current_hidden=aligned_hidden, current_data=aligned_current_data), aligned_batch

    def norm_func(self, x1, x2, seq_lengths: Optional[torch.Tensor] = None):
        #return torch.norm(x1 - x2, dim=(1,2))
        if seq_lengths is not None and x1.ndim == 2:
            lengths = seq_lengths.to(x1.device) + self.inner.puzzle_emb_len
            return packed_norm_ratio_from_lengths(x1, x2, lengths)

        if seq_lengths is not None:
            lengths = (seq_lengths.to(x1.device) + self.inner.puzzle_emb_len).clamp(max=x1.shape[1])
            mask = torch.arange(x1.shape[1], device=x1.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1)
            diff_norm = torch.norm(torch.where(mask, x1 - x2, torch.zeros_like(x1)), dim=(1, 2))
            sum_norm = torch.norm(torch.where(mask, x1 + x2, torch.zeros_like(x1)), dim=(1, 2))
            return diff_norm / (1e-7 + sum_norm / 2)
        return torch.norm(x1 - x2, dim=(1,2)) / (1e-7 + torch.norm(x1 + x2, dim=(1,2)) / 2)

    def forward(
        self,
        carry: URMCarry,
        batch: Dict[str, torch.Tensor],
        compute_target_q=False
    ) -> Tuple[URMCarry, Dict[str, torch.Tensor]]:
        if self.config.variable_seq_lengths:
            return self._forward_packed(carry, batch, compute_target_q=compute_target_q)

        carry, batch = self._align_variable_batch(carry, batch)
        new_carry = self.inner.reset_carry(carry.halted, carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v
            )
            for k, v in carry.current_data.items()
        }

        new_carry2, logits, (q_halt_logits, q_continue_logits), diff_L = self.inner(new_carry, new_current_data)
        
        hidden_diff_norm = self.norm_func(
            new_carry2.current_hidden.detach(),
            new_carry.current_hidden.detach(),
            new_current_data.get("seq_lengths") if self.config.variable_seq_lengths else None,
        )
        sum_norm_with_steps = torch.bincount(new_carry2.steps.cpu(), weights=hidden_diff_norm.cpu(), minlength=self.config.loops + 1) # (loops + 1,)
        steps_count = torch.bincount(new_carry2.steps.cpu(), minlength=self.config.loops + 1) # (loops + 1,)
        mean_norm_with_steps = sum_norm_with_steps / steps_count.clamp_min(1)
        # print(mean_norm_with_steps)
        if global_logger.is_log:
            global_logger.store("mean_norm_with_steps", mean_norm_with_steps)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }
        if diff_L is not None:
            outputs["diff_L"] = diff_L

        with torch.no_grad():
            new_steps = new_steps + 1
            halted = (new_steps >= self.config.loops)

            if self.training and (self.config.loops > 1):
                halted = halted | (q_halt_logits > 0)

                # Exploration
                halt_exploration_prob = 0.1
                min_halt_steps = (torch.rand_like(q_halt_logits) < halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.loops + 1)
                halted = halted & (new_steps >= min_halt_steps)
                
                if (self.config.use_act == False or self.config.halt_norm_in_use_act == True) and self.training == True:  
                    #print("Hidden diff norm:", hidden_diff_norm)
                    norm_diff_max = self.config.norm_diff_max
                    norm_diff_min = self.config.norm_diff_min
                    
                    if norm_diff_max != norm_diff_min:
                        norm_diff_threshold = torch.rand_like(hidden_diff_norm) * (norm_diff_max - norm_diff_min) + norm_diff_min
                    else:
                        norm_diff_threshold = torch.full_like(hidden_diff_norm, norm_diff_max)
                    # print("Hidden diff norm:", hidden_diff_norm)
                    # print("Norm diff threshold:", norm_diff_threshold+ self.config.attn_dropout)
                    halted = halted | (hidden_diff_norm < (norm_diff_threshold + self.config.attn_dropout))
                # halted = torch.ones_like(halted)

        return (
            URMCarry(
                current_hidden=new_carry2.current_hidden,
                steps=new_steps,
                halted=halted,
                current_data=new_current_data,
            ),
            outputs,
        )

    def _forward_packed(
        self,
        carry: URMCarry,
        batch: Dict[str, torch.Tensor],
        compute_target_q=False,
    ) -> Tuple[URMCarry, Dict[str, torch.Tensor]]:
        new_current_data = self._merge_packed_current_data(carry, batch)
        new_carry = self._reset_packed_carry(carry.halted, carry, new_current_data)
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_carry2, logits, (q_halt_logits, q_continue_logits), diff_L = self.inner(new_carry, new_current_data)

        hidden_diff_norm = self.norm_func(
            new_carry2.current_hidden.detach(),
            new_carry.current_hidden.detach(),
            new_current_data.get("seq_lengths"),
        )
        sum_norm_with_steps = torch.bincount(new_carry2.steps.cpu(), weights=hidden_diff_norm.cpu(), minlength=self.config.loops + 1)
        steps_count = torch.bincount(new_carry2.steps.cpu(), minlength=self.config.loops + 1)
        mean_norm_with_steps = sum_norm_with_steps / steps_count.clamp_min(1)
        if global_logger.is_log:
            global_logger.store("mean_norm_with_steps", mean_norm_with_steps)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }
        if self.config.answer_only:
            token_indices = self.inner._packed_data_token_indices(new_current_data)
            (
                _answer_data_mask,
                _answer_indices,
                answer_labels,
                answer_lengths,
                _cu_answer,
                _cu_context,
                _max_answer_len,
                _max_context_len,
            ) = self.inner._packed_answer_metadata(new_current_data, token_indices)
            outputs["loss_labels"] = answer_labels
            outputs["loss_seq_lengths"] = answer_lengths.to(torch.int32)
        if diff_L is not None:
            outputs["diff_L"] = diff_L

        with torch.no_grad():
            new_steps = new_steps + 1
            halted = (new_steps >= self.config.loops)

            if self.training and (self.config.loops > 1):
                halted = halted | (q_halt_logits > 0)

                halt_exploration_prob = 0.1
                min_halt_steps = (torch.rand_like(q_halt_logits) < halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.loops + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if (self.config.use_act == False or self.config.halt_norm_in_use_act == True) and self.training == True:
                    norm_diff_max = self.config.norm_diff_max
                    norm_diff_min = self.config.norm_diff_min

                    if norm_diff_max != norm_diff_min:
                        norm_diff_threshold = torch.rand_like(hidden_diff_norm) * (norm_diff_max - norm_diff_min) + norm_diff_min
                    else:
                        norm_diff_threshold = torch.full_like(hidden_diff_norm, norm_diff_max)
                    halted = halted | (hidden_diff_norm < (norm_diff_threshold + self.config.attn_dropout))

        return (
            URMCarry(
                current_hidden=new_carry2.current_hidden,
                steps=new_steps,
                halted=halted,
                current_data=new_current_data,
            ),
            outputs,
        )
