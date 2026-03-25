from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from models.common import trunc_normal_init_
from models.layers import rms_norm, ConvSwiGLU, Attention, RotaryEmbedding, RotaryEmbedding2D, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from logger import global_logger

@dataclass
class URMInnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class URMCarry:
    inner_carry: URMInnerCarry
    steps: Optional[torch.Tensor] = None
    halted: Optional[torch.Tensor] = None
    current_data: Optional[Dict[str, torch.Tensor]] = None


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
    grid_height: int = 0  # Grid height for 2D RoPE (0 = use 1D RoPE)
    grid_width: int = 0   # Grid width  for 2D RoPE (0 = use 1D RoPE)
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0
    topk_sparsity: float = 0.0
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    loops: int
    L_cycles: int
    H_cycles: int
    forward_dtype: str = "bfloat16"
    use_act: bool = True
    noise_size: float = 0.0
    noise_seed: int = 42
    norm_diff_max: float = 0.2
    norm_diff_min: float = 0.1


class URMBlock(nn.Module):
    def __init__(self, config: URMConfig) -> None:
        super().__init__()
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
            attn_dropout=config.attn_dropout,
            topk_sparsity=config.topk_sparsity,
        )
        self.mlp = ConvSwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
            mlp_dropout=config.mlp_dropout,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_output = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states, window_size=-1)
        hidden_states = rms_norm(hidden_states + attn_output, variance_epsilon=self.norm_eps)
        mlp_output = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_output, variance_epsilon=self.norm_eps)
        return hidden_states


class URM_Inner(nn.Module):
    def __init__(self, config: URMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

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

        if self.config.grid_height > 0 and self.config.grid_width > 0:
            self.rotary_emb = RotaryEmbedding2D(
                dim=self.config.hidden_size // self.config.num_heads,
                grid_height=self.config.grid_height,
                grid_width=self.config.grid_width,
                puzzle_emb_len=self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        else:
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )

        self.layers = nn.ModuleList([URMBlock(self.config) for _ in range(self.config.num_layers)])

        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)
            
        self.generator = torch.Generator(device="cuda").manual_seed(self.config.noise_seed)

    def reasoning_module(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states

    def _maybe_add_noise(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.noise_size <= 0:
            return hidden_states
        noise = torch.randn(
            hidden_states.shape,
            generator=self.generator,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            layout=hidden_states.layout
        )
        return hidden_states + noise * self.config.noise_size

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
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
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int) -> URMInnerCarry:
        return URMInnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
            z_L=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: URMInnerCarry) -> URMInnerCarry:
        new_z_H = torch.where(
            reset_flag.view(-1, 1, 1),
            self.H_init,
            carry.z_H
        )
        new_z_L = torch.where(
            reset_flag.view(-1, 1, 1),
            self.L_init,
            carry.z_L
        )
        return URMInnerCarry(z_H=new_z_H, z_L=new_z_L)

    def forward(
        self,
        carry: URMInnerCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[URMInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(cos_sin=self.rotary_emb())
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        z_H = carry.z_H
        z_L = carry.z_L
        if self.config.H_cycles > 1:
            with torch.no_grad():
                for _ in range(self.config.H_cycles - 1):
                    for _ in range(self.config.L_cycles):
                        z_L = self.reasoning_module(z_L, z_H + input_embeddings, **seq_info)
                        z_L = self._maybe_add_noise(z_L)

                    z_H = self.reasoning_module(z_H, z_L, **seq_info)
                    z_H = self._maybe_add_noise(z_H)

        # Gradient norm logging for unrolled layers
        _log_grads = global_logger.is_log and self.training
        if _log_grads:
            _grad_norms = {}
            _total_unrolled = (self.config.L_cycles + 1) * len(self.layers)
            def _make_grad_hook(idx, container, total):
                def hook(grad):
                    container[idx] = grad.detach().norm().item()
                    if len(container) == total:
                        norm_tensor = torch.tensor([container[i] for i in range(total)])
                        global_logger.store("grad_norm_per_layer", norm_tensor)
                return hook

        _unrolled_idx = 0
        for _ in range(self.config.L_cycles):
            z_L = z_L + z_H + input_embeddings
            for layer in self.layers:
                z_L = layer(hidden_states=z_L, **seq_info)
                z_L = self._maybe_add_noise(z_L)
                if _log_grads:
                    z_L.register_hook(_make_grad_hook(_unrolled_idx, _grad_norms, _total_unrolled))
                    _unrolled_idx += 1

        z_H = z_H + z_L
        for layer in self.layers:
            z_H = layer(hidden_states=z_H, **seq_info)
            z_H = self._maybe_add_noise(z_H)
            if _log_grads:
                z_H.register_hook(_make_grad_hook(_unrolled_idx, _grad_norms, _total_unrolled))
                _unrolled_idx += 1

        new_carry = URMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class URM(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = URMConfig(**config_dict)
        self.inner = URM_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> URMCarry:
        batch_size = batch["inputs"].shape[0]
        base = self.inner.empty_carry(batch_size)
        return URMCarry(
            inner_carry=base,
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )
        
    def norm_func(self, x1, x2):
        #return torch.norm(x1 - x2, dim=(1,2))
        return torch.norm(x1 - x2, dim=(1,2)) / (1e-7 + torch.norm(x1 + x2, dim=(1,2)) / 2)

    def forward(
        self,
        carry: URMCarry,
        batch: Dict[str, torch.Tensor],
        compute_target_q=False
    ) -> Tuple[URMCarry, Dict[str, torch.Tensor]]:

        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v
            )
            for k, v in carry.current_data.items()
        }

        new_inner_carry2, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)
        
        hidden_diff_norm = self.norm_func(new_inner_carry2.z_H.detach(), new_inner_carry.z_H.detach())
        step_bucket = (new_steps + 1).cpu()
        sum_norm_with_steps = torch.bincount(step_bucket, weights=hidden_diff_norm.cpu(), minlength=self.config.loops + 1) # (loops + 1,)
        steps_count = torch.bincount(step_bucket, minlength=self.config.loops + 1) # (loops + 1,)
        mean_norm_with_steps = sum_norm_with_steps / steps_count.clamp_min(1)
        # print(mean_norm_with_steps)
        if global_logger.is_log:
            global_logger.store("mean_norm_with_steps", mean_norm_with_steps)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            halted = (new_steps >= self.config.loops)

            if self.training and (self.config.loops > 1):
                halted = halted | (q_halt_logits > 0)

                # Exploration
                halt_exploration_prob = 0.1
                min_halt_steps = (torch.rand_like(q_halt_logits) < halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.loops + 1)
                halted = halted & (new_steps >= min_halt_steps)
                
                if self.config.use_act == False and self.training == True:  
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

        return (
            URMCarry(
                inner_carry=new_inner_carry2,
                steps=new_steps,
                halted=halted,
                current_data=new_current_data,
            ),
            outputs,
        )
