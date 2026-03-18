from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, replace
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
class URMCarry:
    current_hidden: torch.Tensor
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

    def _full_attnres_mix(
        self,
        sources: List[torch.Tensor],   # len=S, each [B, T, D]
        query: torch.Tensor,           # [D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full AttnRes mixing.

        sources[i]: [B, T, D]
        V         : [S, B, T, D]
        logits    : [S, B, T]
        alpha     : [S, B, T]
        mixed     : [B, T, D]
        """
        if len(sources) == 0:
            raise ValueError("sources must contain at least one tensor.")

        V = torch.stack(sources, dim=0)  # [S, B, T, D]
        K = rms_norm(V, variance_epsilon=self.norm_eps)  # [S, B, T, D]

        q = query.to(dtype=K.dtype, device=K.device)     # [D]
        logits = torch.einsum("d,sbtd->sbt", q, K)       # [S, B, T]
        alpha = torch.softmax(logits, dim=0)             # [S, B, T]

        mixed = torch.einsum("sbt,sbtd->btd", alpha, V)  # [B, T, D]
        return mixed, alpha

    def forward(
        self,
        cos_sin: CosSin,
        sources: List[torch.Tensor],   # source accumulation is local to one L_cycle
        attn_query: torch.Tensor,      # [D] for this (cycle, layer)
        mlp_query: torch.Tensor,       # [D] for this (cycle, layer)
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Full AttnRes block.

        Inputs:
            sources:
                Local source list for the current L_cycle only.
                Each element has shape [B, T, D].

            attn_query:
                [D], cycle-layer-specific query for the attention sublayer.

            mlp_query:
                [D], cycle-layer-specific query for the MLP sublayer.

        Returns:
            hidden_states:
                [B, T, D] = output of the MLP sublayer

            sources:
                original sources + [attn_output, mlp_output]
        """
        # 1) Build self-attention input from all previous sources in this cycle
        attn_input, _ = self._full_attnres_mix(
            sources=sources,
            query=attn_query,
        )  # [B, T, D]

        # 2) Attention sublayer output
        attn_output = self.self_attn(
            cos_sin=cos_sin,
            hidden_states=attn_input,
            window_size=-1,
        )  # [B, T, D]

        # Add attention output as a new source
        new_sources = list(sources)
        new_sources.append(attn_output)

        # 3) Build MLP input from updated source list
        mlp_input, _ = self._full_attnres_mix(
            sources=new_sources,
            query=mlp_query,
        )  # [B, T, D]

        # 4) MLP sublayer output
        mlp_output = self.mlp(mlp_input)  # [B, T, D]

        # Add MLP output as a new source
        new_sources.append(mlp_output)

        hidden_states = mlp_output
        return hidden_states, new_sources

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

        # Full AttnRes queries with per-cycle and per-layer specificity
        # attn_queries: [L_cycles, num_layers, D]
        # mlp_queries : [L_cycles, num_layers, D]
        self.attn_queries = nn.Parameter(
            torch.zeros(self.config.L_cycles, self.config.num_layers, self.config.hidden_size)
        )
        self.mlp_queries = nn.Parameter(
            torch.zeros(self.config.L_cycles, self.config.num_layers, self.config.hidden_size)
        )

        self.init_hidden = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)
            self.attn_queries.zero_()
            self.mlp_queries.zero_()

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

    def empty_carry(self, batch_size: int) -> URMCarry:
        return URMCarry(
            current_hidden=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
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

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.noise_size > 0:
            return x + (torch.randn_like(x) * self.config.noise_size * 2 - self.config.noise_size)
        return x

    def _run_one_l_cycle(
        self,
        hidden_states: torch.Tensor,    # [B, T, D]
        input_embeddings: torch.Tensor, # [B, T, D]
        seq_info: Dict[str, CosSin],
        cycle_idx: int,
        log_grads: bool = False,
        grad_container: Optional[Dict[int, float]] = None,
        total_unrolled: Optional[int] = None,
        unrolled_offset: int = 0,
    ) -> Tuple[torch.Tensor, int]:
        """
        Run one local L_cycle with Full AttnRes.

        Source accumulation is done ONLY inside this function:
            sources = [cycle_input]
            then append attn_output, mlp_output within the cycle.
        """
        hidden_states = hidden_states + input_embeddings  # [B, T, D]
        sources: List[torch.Tensor] = [hidden_states]     # local sources for this cycle only

        def _make_grad_hook(idx, container, total):
            def hook(grad):
                container[idx] = grad.detach().norm().item()
                if len(container) == total:
                    norm_tensor = torch.tensor([container[i] for i in range(total)])
                    global_logger.store("grad_norm_per_layer", norm_tensor)
            return hook

        local_idx = 0
        for layer_idx, layer in enumerate(self.layers):
            attn_query = self.attn_queries[cycle_idx, layer_idx]  # [D]
            mlp_query = self.mlp_queries[cycle_idx, layer_idx]    # [D]

            hidden_states, sources = layer(
                cos_sin=seq_info["cos_sin"],
                sources=sources,
                attn_query=attn_query,
                mlp_query=mlp_query,
            )

            hidden_states = self._add_noise(hidden_states)

            # Keep the final source aligned with the noisy hidden state
            if self.config.noise_size > 0:
                sources[-1] = hidden_states

            if log_grads:
                hidden_states.register_hook(
                    _make_grad_hook(unrolled_offset + local_idx, grad_container, total_unrolled)
                )
            local_idx += 1

        return hidden_states, local_idx

    def forward(
        self,
        carry: URMCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[URMCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(cos_sin=self.rotary_emb())
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        hidden_states = carry.current_hidden

        if self.config.H_cycles > 1:
            with torch.no_grad():
                for _ in range(self.config.H_cycles - 1):
                    for cycle_idx in range(self.config.L_cycles):
                        hidden_states, _ = self._run_one_l_cycle(
                            hidden_states=hidden_states,
                            input_embeddings=input_embeddings,
                            seq_info=seq_info,
                            cycle_idx=cycle_idx,
                            log_grads=False,
                        )

        _log_grads = global_logger.is_log and self.training
        if _log_grads:
            _grad_norms = {}
            _total_unrolled = self.config.L_cycles * len(self.layers)
        else:
            _grad_norms = None
            _total_unrolled = None

        _unrolled_idx = 0
        for cycle_idx in range(self.config.L_cycles):
            hidden_states, used_layers = self._run_one_l_cycle(
                hidden_states=hidden_states,
                input_embeddings=input_embeddings,
                seq_info=seq_info,
                cycle_idx=cycle_idx,
                log_grads=_log_grads,
                grad_container=_grad_norms,
                total_unrolled=_total_unrolled,
                unrolled_offset=_unrolled_idx,
            )
            _unrolled_idx += used_layers

        new_carry = replace(carry, current_hidden=hidden_states.detach())
        output = self.lm_head(hidden_states)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(hidden_states[:, 0]).to(torch.float32)
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
            current_hidden=base.current_hidden,
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

        new_carry2, logits, (q_halt_logits, q_continue_logits) = self.inner(new_carry, new_current_data)
        
        hidden_diff_norm = self.norm_func(new_carry2.current_hidden.detach(), new_carry.current_hidden.detach())
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
                    norm_diff_max = getattr(getattr(self.config, "config", None), "norm_diff_max", 0.1)
                    norm_diff_min = getattr(getattr(self.config, "config", None), "norm_diff_min", 0.01)
                    if self.config.attn_dropout == 0.0:
                        norm_diff_max = 0.01
                        norm_diff_min = 0.005
                    if norm_diff_max != norm_diff_min:
                        norm_diff_threshold = torch.rand_like(hidden_diff_norm) * (norm_diff_max - norm_diff_min) + norm_diff_min
                    else:
                        norm_diff_threshold = torch.full_like(hidden_diff_norm, norm_diff_max)
                    # print("Hidden diff norm:", hidden_diff_norm)
                    # print("Norm diff threshold:", norm_diff_threshold+ self.config.attn_dropout)
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
