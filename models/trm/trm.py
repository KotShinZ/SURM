from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, replace
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
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
    def __init__(self, config: TRMConfig) -> None:
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

class TRMReasoningModule(nn.Module):
    def __init__(self, layers: List[TRMBlock]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, compute_target_q=False, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
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
        self.L_level = TRMReasoningModule(layers=[TRMBlock(self.config) for _i in range(self.config.L_layers)])

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
        batch_size = batch["inputs"].shape[0]
        seq_len = batch["inputs"].shape[1] if self.config.variable_seq_lengths else None

        return TRMCarry(
            inner_carry=self.inner.empty_carry(batch_size, seq_len=seq_len),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

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
