from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass, replace
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from models.common import packed_norm_ratio_from_lengths, trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100

@dataclass
class TRMInnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TRMCarry:
    inner_carry: TRMInnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


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


class TRMConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    attention_window_size: Union[int, List[int]] = -1

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    attn_dropout: float = 0.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float
    use_act: bool = True
    norm_diff_max: float = 0.2
    norm_diff_min: float = 0.1
    halt_norm_in_use_act: bool = False

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense
    variable_seq_lengths: bool = False
    profile: bool = False

class TRMBlock(nn.Module):
    def __init__(self, config: TRMConfig, attention_window_size: int) -> None:
        super().__init__()

        self.config = config
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        if self.config.mlp_t:
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
                attn_dropout=config.attn_dropout,
                attention_type="full" if attention_window_size == -1 else "swa",
                attention_window_size=attention_window_size,
                prefix_seq_len=self.puzzle_emb_len,
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

        
    def forward(
        self,
        cos_sin: CosSin,
        hidden_states: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None,
        compute_target_q=False,
    ) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            hidden_states = rms_norm(
                hidden_states
                + self.self_attn(
                    cos_sin=cos_sin,
                    hidden_states=hidden_states,
                    sequence_lengths=sequence_lengths,
                ),
                variance_epsilon=self.norm_eps,
            )
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

    def forward_packed(
        self,
        cos_sin: CosSin,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        compute_target_q=False,
    ) -> torch.Tensor:
        if self.config.mlp_t:
            raise ValueError("Packed variable-length TRM does not support mlp_t=True")

        hidden_states = rms_norm(
            hidden_states
            + self.self_attn.forward_packed(
                cos_sin=cos_sin,
                hidden_states=hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            ),
            variance_epsilon=self.norm_eps,
        )
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TRMReasoningModule(nn.Module):
    def __init__(self, layers: List[TRMBlock]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, compute_target_q=False, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states

    def forward_packed(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, compute_target_q=False, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer.forward_packed(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TRM_Inner(nn.Module):
    def __init__(self, config: TRMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.layer_attention_window_sizes = _normalize_attention_window_sizes(
            self.config.attention_window_size,
            self.config.L_layers,
            "TRMConfig.attention_window_size",
        )
        self.L_level = TRMReasoningModule(
            layers=[
                TRMBlock(self.config, attention_window_size=self.layer_attention_window_sizes[layer_idx])
                for layer_idx in range(self.config.L_layers)
            ]
        )

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (
                embedding + self.embed_pos.embedding_weight[: embedding.shape[1]].to(self.forward_dtype)
            )

        # Scale
        return self.embed_scale * embedding

    def _packed_lengths(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["seq_lengths"].to(device=batch["inputs"].device, dtype=torch.long)

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

    def _packed_flat_position_ids(self, batch: Dict[str, torch.Tensor], token_indices: torch.Tensor) -> torch.Tensor:
        lengths = self._packed_lengths(batch)
        seq_ids = torch.repeat_interleave(
            torch.arange(lengths.shape[0], device=batch["inputs"].device, dtype=torch.long),
            lengths,
        )
        data_offsets = batch["seq_offsets"][:-1].to(device=batch["inputs"].device, dtype=torch.long)
        token_positions = torch.arange(batch["inputs"].shape[0], device=batch["inputs"].device, dtype=torch.long)
        data_pos_ids = self.puzzle_emb_len + token_positions - data_offsets[seq_ids]
        if self.puzzle_emb_len == 0:
            return data_pos_ids

        flat_ids = data_pos_ids.new_empty((token_indices.shape[0] + batch["puzzle_identifiers"].shape[0] * self.puzzle_emb_len,))
        flat_ids[token_indices] = data_pos_ids
        prefix_ids = torch.arange(self.puzzle_emb_len, device=batch["inputs"].device, dtype=torch.long)
        flat_ids[self._packed_prefix_indices(batch)] = prefix_ids.repeat(batch["puzzle_identifiers"].shape[0])
        return flat_ids

    def _input_embeddings_packed(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
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

        if self.config.pos_encodings == "learned":
            flat_ids = self._packed_flat_position_ids(batch, token_indices)
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight[flat_ids].to(self.forward_dtype))

        return self.embed_scale * embedding, token_indices

    def _rotary_cos_sin_packed(self, batch: Dict[str, torch.Tensor], token_indices: torch.Tensor) -> Optional[CosSin]:
        if not hasattr(self, "rotary_emb"):
            return None
        flat_ids = self._packed_flat_position_ids(batch, token_indices)
        cos, sin = self.rotary_emb()
        return cos[flat_ids], sin[flat_ids]

    def empty_carry(self, batch_size: int, seq_len: Optional[int] = None):
        carry_seq_len = self.config.seq_len if seq_len is None else seq_len
        return TRMInnerCarry(
            z_H=torch.empty(batch_size, carry_seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, carry_seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TRMInnerCarry):
        return TRMInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def _sequence_lengths(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.config.variable_seq_lengths or "seq_lengths" not in batch:
            return None
        return batch["seq_lengths"].to(torch.int32) + self.puzzle_emb_len

    def forward(
        self,
        carry: TRMInnerCarry,
        batch: Dict[str, torch.Tensor],
        compute_target_q: bool = False,
    ) -> Tuple[ TRMInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor] ]:
        if self.config.variable_seq_lengths:
            return self.forward_packed(carry, batch, compute_target_q=compute_target_q)

        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
            sequence_lengths=self._sequence_lengths(batch),
        )

        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        it = 0
        z_H, z_L = carry.z_H, carry.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = TRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])

    def forward_packed(
        self,
        carry: TRMInnerCarry,
        batch: Dict[str, torch.Tensor],
        compute_target_q: bool = False,
    ) -> Tuple[ TRMInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor] ]:
        input_embeddings, token_indices = self._input_embeddings_packed(batch)
        cu_seqlens, max_seqlen = self._packed_cu_seqlens(batch)
        seq_info = dict(
            cos_sin=self._rotary_cos_sin_packed(batch, token_indices),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        z_H, z_L = carry.z_H, carry.z_L
        if z_H.shape[0] != input_embeddings.shape[0] or z_L.shape[0] != input_embeddings.shape[0]:
            raise RuntimeError(
                f"Packed carry/input length mismatch: z_H={z_H.shape[0]} z_L={z_L.shape[0]} input={input_embeddings.shape[0]}"
            )

        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level.forward_packed(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level.forward_packed(z_H, z_L, **seq_info)

        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level.forward_packed(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level.forward_packed(z_H, z_L, **seq_info)

        new_carry = TRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H[token_indices])
        q_logits = self.q_head(z_H[cu_seqlens[:-1].to(torch.long)]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TRM(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TRMConfig(**config_dict)
        self.inner = TRM_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["puzzle_identifiers"].shape[0] if self.config.variable_seq_lengths else batch["inputs"].shape[0]
        if self.config.variable_seq_lengths:
            empty_hidden = torch.empty(
                0,
                self.config.hidden_size,
                dtype=self.inner.forward_dtype,
                device=batch["inputs"].device,
            )
            return TRMCarry(
                inner_carry=TRMInnerCarry(z_H=empty_hidden, z_L=empty_hidden),
                steps=torch.zeros((batch_size, ), dtype=torch.int32, device=batch["inputs"].device),
                halted=torch.ones((batch_size, ), dtype=torch.bool, device=batch["inputs"].device),
                current_data={k: torch.empty_like(v) for k, v in batch.items()}
            )

        seq_len = batch["inputs"].shape[1] if self.config.variable_seq_lengths else None

        return TRMCarry(
            inner_carry=self.inner.empty_carry(batch_size, seq_len=seq_len),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    @torch.compiler.disable
    def _merge_packed_current_data(self, carry: TRMCarry, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        halted = carry.halted.detach().cpu().tolist()
        batch_offsets = batch["seq_offsets"].detach().cpu().tolist()
        current_offsets = carry.current_data["seq_offsets"].detach().cpu().tolist()
        batch_size = len(halted)

        token_keys = [
            key for key in ("inputs", "labels", "position_ids", "source_inputs")
            if key in batch and key in carry.current_data
        ]
        token_chunks = {key: [] for key in token_keys}
        lengths = []
        puzzle_identifiers = []

        for idx in range(batch_size):
            use_batch = bool(halted[idx])
            source = batch if use_batch else carry.current_data
            offsets = batch_offsets if use_batch else current_offsets
            start, end = int(offsets[idx]), int(offsets[idx + 1])

            for key in token_keys:
                token_chunks[key].append(source[key][start:end])
            lengths.append(source["seq_lengths"][idx])
            puzzle_identifiers.append(source["puzzle_identifiers"][idx])

        merged: Dict[str, torch.Tensor] = {}
        for key, chunks in token_chunks.items():
            merged[key] = torch.cat(chunks, dim=0) if chunks else batch[key].new_empty((0,) + batch[key].shape[1:])

        if lengths:
            merged["seq_lengths"] = torch.stack(lengths).to(device=batch["inputs"].device, dtype=torch.int32)
            merged["puzzle_identifiers"] = torch.stack(puzzle_identifiers).to(
                device=batch["inputs"].device,
                dtype=batch["puzzle_identifiers"].dtype,
            )
        else:
            merged["seq_lengths"] = batch["seq_lengths"].new_empty((0,))
            merged["puzzle_identifiers"] = batch["puzzle_identifiers"].new_empty((0,))

        merged["seq_offsets"] = F.pad(torch.cumsum(merged["seq_lengths"].to(torch.int32), dim=0), (1, 0))
        return merged

    @torch.compiler.disable
    def _reset_packed_carry(self, reset_flag: torch.Tensor, carry: TRMCarry, current_data: Dict[str, torch.Tensor]) -> TRMInnerCarry:
        reset = reset_flag.detach().cpu().tolist()
        current_lengths = current_data["seq_lengths"].detach().cpu().tolist()
        old_hidden_lengths = (carry.current_data["seq_lengths"] + self.inner.puzzle_emb_len).detach().cpu().tolist()
        old_hidden_offsets = [0]
        for length in old_hidden_lengths:
            old_hidden_offsets.append(old_hidden_offsets[-1] + int(length))

        z_h_chunks = []
        z_l_chunks = []
        for idx, should_reset in enumerate(reset):
            new_len = int(current_lengths[idx]) + self.inner.puzzle_emb_len
            if should_reset:
                z_h_chunks.append(self.inner.H_init.expand(new_len, -1))
                z_l_chunks.append(self.inner.L_init.expand(new_len, -1))
            else:
                z_h_chunks.append(carry.inner_carry.z_H[old_hidden_offsets[idx]:old_hidden_offsets[idx + 1]])
                z_l_chunks.append(carry.inner_carry.z_L[old_hidden_offsets[idx]:old_hidden_offsets[idx + 1]])

        if z_h_chunks:
            return TRMInnerCarry(z_H=torch.cat(z_h_chunks, dim=0), z_L=torch.cat(z_l_chunks, dim=0))

        empty_hidden = torch.empty(
            0,
            self.config.hidden_size,
            dtype=self.inner.forward_dtype,
            device=current_data["inputs"].device,
        )
        return TRMInnerCarry(z_H=empty_hidden, z_L=empty_hidden)

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

    def _pad_hidden(self, value: torch.Tensor, target_hidden_len: int) -> torch.Tensor:
        if value.shape[1] == target_hidden_len:
            return value
        if value.shape[1] > target_hidden_len:
            return value[:, :target_hidden_len]

        pad_shape = list(value.shape)
        pad_shape[1] = target_hidden_len - value.shape[1]
        return torch.cat(
            [value, torch.zeros(pad_shape, dtype=value.dtype, device=value.device)],
            dim=1,
        )

    def _align_variable_batch(self, carry: TRMCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TRMCarry, Dict[str, torch.Tensor]]:
        if not self.config.variable_seq_lengths:
            return carry, batch

        if bool(carry.halted.all().item()):
            target_seq_len = batch["inputs"].shape[1]
        else:
            target_seq_len = max(carry.current_data["inputs"].shape[1], batch["inputs"].shape[1])
        target_hidden_len = target_seq_len + self.inner.puzzle_emb_len
        aligned_batch = {k: self._pad_to_seq_len(k, v, target_seq_len) for k, v in batch.items()}
        aligned_current_data = {k: self._pad_to_seq_len(k, v, target_seq_len) for k, v in carry.current_data.items()}
        aligned_inner = TRMInnerCarry(
            z_H=self._pad_hidden(carry.inner_carry.z_H, target_hidden_len),
            z_L=self._pad_hidden(carry.inner_carry.z_L, target_hidden_len),
        )

        return replace(carry, inner_carry=aligned_inner, current_data=aligned_current_data), aligned_batch

    def norm_func(self, x1: torch.Tensor, x2: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        return torch.norm(x1 - x2, dim=(1, 2)) / (1e-7 + torch.norm(x1 + x2, dim=(1, 2)) / 2)

    def carry_diff_norm(
        self,
        x1: TRMInnerCarry,
        x2: TRMInnerCarry,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z_h_diff = self.norm_func(x1.z_H.detach(), x2.z_H.detach(), seq_lengths)
        z_l_diff = self.norm_func(x1.z_L.detach(), x2.z_L.detach(), seq_lengths)
        return torch.maximum(z_h_diff, z_l_diff)
        
    def forward(self, carry: TRMCarry, batch: Dict[str, torch.Tensor], compute_target_q=False) -> Tuple[TRMCarry, Dict[str, torch.Tensor]]:
        if self.config.variable_seq_lengths:
            return self._forward_packed(carry, batch, compute_target_q=compute_target_q)

        carry, batch = self._align_variable_batch(carry, batch)

        # Update data, carry (removing halted sequences)
        reset_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(reset_inner_carry, new_current_data)
        hidden_diff_norm = self.carry_diff_norm(
            new_inner_carry,
            reset_inner_carry,
            new_current_data.get("seq_lengths") if self.config.variable_seq_lengths else None,
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
                if self.config.use_act:
                    if self.config.no_ACT_continue:
                        halted = halted | (q_halt_logits > 0)
                    else:
                        halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if (self.config.use_act == False or self.config.halt_norm_in_use_act == True) and self.training == True:
                    norm_diff_max = self.config.norm_diff_max
                    norm_diff_min = self.config.norm_diff_min

                    if norm_diff_max != norm_diff_min:
                        norm_diff_threshold = torch.rand_like(hidden_diff_norm) * (norm_diff_max - norm_diff_min) + norm_diff_min
                    else:
                        norm_diff_threshold = torch.full_like(hidden_diff_norm, norm_diff_max)
                    halted = halted | (hidden_diff_norm < (norm_diff_threshold + self.config.attn_dropout))

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TRMCarry(new_inner_carry, new_steps, halted, new_current_data), outputs

    def _forward_packed(self, carry: TRMCarry, batch: Dict[str, torch.Tensor], compute_target_q=False) -> Tuple[TRMCarry, Dict[str, torch.Tensor]]:
        new_current_data = self._merge_packed_current_data(carry, batch)
        reset_inner_carry = self._reset_packed_carry(carry.halted, carry, new_current_data)
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(reset_inner_carry, new_current_data)
        hidden_diff_norm = self.carry_diff_norm(
            new_inner_carry,
            reset_inner_carry,
            new_current_data.get("seq_lengths"),
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                if self.config.use_act:
                    if self.config.no_ACT_continue:
                        halted = halted | (q_halt_logits > 0)
                    else:
                        halted = halted | (q_halt_logits > q_continue_logits)

                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if (self.config.use_act == False or self.config.halt_norm_in_use_act == True) and self.training == True:
                    norm_diff_max = self.config.norm_diff_max
                    norm_diff_min = self.config.norm_diff_min

                    if norm_diff_max != norm_diff_min:
                        norm_diff_threshold = torch.rand_like(hidden_diff_norm) * (norm_diff_max - norm_diff_min) + norm_diff_min
                    else:
                        norm_diff_threshold = torch.full_like(hidden_diff_norm, norm_diff_max)
                    halted = halted | (hidden_diff_norm < (norm_diff_threshold + self.config.attn_dropout))

                if not self.config.no_ACT_continue:
                    next_inner_carry, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(new_inner_carry, new_current_data)
                    del next_inner_carry
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TRMCarry(new_inner_carry, new_steps, halted, new_current_data), outputs
