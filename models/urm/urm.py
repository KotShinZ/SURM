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
    attention_type: str = "full"
    attention_window_size: int = -1
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


class URMBlock(nn.Module):
    def __init__(self, config: URMConfig) -> None:
        super().__init__()
        puzzle_prefix_len = -(config.puzzle_emb_ndim // -config.hidden_size)
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
            attn_dropout=config.attn_dropout,
            attention_type=config.attention_type,
            attention_window_size=config.attention_window_size,
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


class URM_Inner(nn.Module):
    def __init__(self, config: URMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.patch_area = self.config.patch_height * self.config.patch_width
        self.inner_seq_len = self.config.seq_len
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

        if self.inner_config.grid_height > 0 and self.inner_config.grid_width > 0:
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

        self.layers = nn.ModuleList([URMBlock(self.inner_config) for _ in range(self.config.num_layers)])

        self.init_hidden = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)
            
        self.generator = torch.Generator(device="cuda").manual_seed(self.config.noise_seed)

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
        if "position_ids" in batch and isinstance(self.rotary_emb, RotaryEmbedding2D):
            return self.rotary_emb(batch["position_ids"], prefix_seq_len=self.puzzle_emb_len)
        return self.rotary_emb()

    def _sequence_lengths(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.config.variable_seq_lengths or "seq_lengths" not in batch:
            return None
        return batch["seq_lengths"].to(torch.int32) + self.puzzle_emb_len

    def forward(
        self,
        carry: URMCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[URMCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
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
                        hidden_states = hidden_states + input_embeddings # + (torch.randn_like(hidden_states) * 2 - 1)
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
            hidden_states = hidden_states + input_embeddings # + (torch.randn_like(hidden_states) * 2 - 1)
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
        seq_len = batch["inputs"].shape[1] if self.config.variable_seq_lengths else None
        base = self.inner.empty_carry(batch_size, seq_len=seq_len)
        return URMCarry(
            current_hidden=base.current_hidden,
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
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
