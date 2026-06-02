from typing import Tuple, List, Dict, Literal, Optional, Union
from dataclasses import dataclass, replace
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel, model_validator
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
    current_low_hidden: Optional[torch.Tensor] = None


@dataclass
class URMLayerInferenceCache:
    key: torch.Tensor
    value: torch.Tensor
    seqlens: torch.Tensor
    conv_hidden_state: Optional[torch.Tensor] = None
    
    def update(self, key: torch.Tensor, value: torch.Tensor, conv_hidden_state: Optional[torch.Tensor] = None) -> "URMLayerInferenceCache":
        updated_key = torch.cat([self.key, key], dim=1)
        updated_value = torch.cat([self.value, value], dim=1)
        updated_seqlens = self.seqlens + key.shape[1]
        return replace(self, key=updated_key, value=updated_value, seqlens=updated_seqlens, conv_hidden_state=conv_hidden_state or self.conv_hidden_state)


@dataclass
class URMInferenceCache:
    layers: List[List[URMLayerInferenceCache]]
    batch_size: int
    max_cache_len: int

def _normalize_attention_window_sizes(
    attention_window_size: Union[int, List[int]],
    num_layers: int,
) -> List[int]:
    if isinstance(attention_window_size, (list, tuple)):
        layer_window_sizes = [int(window_size) for window_size in attention_window_size]
    else:
        layer_window_sizes = [int(attention_window_size)] * num_layers

    return [-1 if window_size == -1 else window_size // 2 for window_size in layer_window_sizes]


ForwardMode = Literal["standard", "answer_only", "prefix_lm", "casual", "causal"]


class URMConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int
    num_layers: int
    H_layers: int = 0
    L_layers: int = 0
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
    grad_H_cycles: int = 1
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
    grad_logging_enabled: bool = True
    forward_mode: ForwardMode = "standard"
    answer_only: bool = False
    answer_only_context_layers: int = 0
    prefix_lm: bool = False
    label_separate: bool = False
    SeparateMode: str = "D"
    separate_mode: Optional[str] = None
    label_separate_C_noise_scale: float = 1.0
    prelude_layers: int = 0
    coda_layers: int = 0
    num_memory_tokens: int = 0
    answer_initial_mode: Literal["default", "black", "random", "noised_label_C", "noised_label_D"] = "default"
    answer_initial_random_std: float = 1.0
    answer_initial_C_noise_distribution: Literal["uniform", "beta"] = "beta"
    answer_initial_C_noise_scale: float = 1.0
    answer_initial_C_beta_alpha: float = 1.0
    answer_initial_C_beta_beta: float = 2.0
    answer_initial_D_ratio_distribution: Literal["constant", "normal", "uniform"] = "uniform"
    answer_initial_D_ratio_min: float = 0.0
    answer_initial_D_ratio_max: float = 1.0
    answer_initial_D_ratio_mean: float = 0.5
    answer_initial_D_ratio_std: float = 0.25
    answer_initial_random_token_min: int = 2
    answer_initial_random_token_max: Optional[int] = None
    answer_initial_pad_token_id: int = 0
    answer_initial_eos_token_id: int = 1
    answer_initial_use_labels_in_eval: bool = False

    @model_validator(mode="after")
    def _normalize_forward_mode(self):
        if self.forward_mode == "causal":
            self.forward_mode = "casual"
        if self.forward_mode == "standard":
            if self.prefix_lm:
                self.forward_mode = "prefix_lm"
            elif self.answer_only:
                self.forward_mode = "answer_only"

        if self.forward_mode == "prefix_lm":
            self.answer_only = True
            self.prefix_lm = True
        elif self.forward_mode == "answer_only":
            self.answer_only = True
            self.prefix_lm = False
        elif self.forward_mode == "casual":
            self.answer_only = False
            self.prefix_lm = False
        return self


class URMBlock(nn.Module):
    def __init__(self, config: URMConfig, attention_window_size: int) -> None:
        super().__init__()
        prefix_seq_len = -(config.puzzle_emb_ndim // -config.hidden_size) + config.num_memory_tokens
        attention_type = config.attention_type.lower()
        if attention_type in {"full", "swa"}:
            attention_type = "full" if attention_window_size == -1 else "swa"
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=config.forward_mode == "casual",
            attn_dropout=config.attn_dropout,
            attention_type=attention_type,
            attention_window_size=attention_window_size,
            attention_window_size_2d=config.attention_window_size_2d,
            attention_topk=config.attention_topk,
            grid_height=config.grid_height,
            grid_width=config.grid_width,
            prefix_seq_len=prefix_seq_len,
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
        mlp_output, _conv_hidden_state = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_output, variance_epsilon=self.norm_eps)
        return hidden_states

    def forward_packed(
        self,
        cos_sin: CosSin,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cache: Optional[URMLayerInferenceCache] = None,
    ) -> Tuple[torch.Tensor, Optional[URMLayerInferenceCache]]:
        if cache is not None:
            attn_output, key, value = self.self_attn.forward_decode(
                cos_sin=cos_sin,
                hidden_states=hidden_states,
                key_cache=cache.key,
                value_cache=cache.value,
                cache_seqlens=cache.seqlens,
            )
        else:   
            attn_output, key, value = self.self_attn.forward_packed(
                cos_sin=cos_sin,
                hidden_states=hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                window_size=-1,
            )
        hidden_states = rms_norm(hidden_states + attn_output, variance_epsilon=self.norm_eps)
        mlp_output, conv_hidden_state = self.mlp.forward_packed(hidden_states, cu_seqlens, cache.conv_hidden_state if cache is not None else None)
        hidden_states = rms_norm(hidden_states + mlp_output, variance_epsilon=self.norm_eps)
        
        new_cache = None
        if cache is not None:
            new_cache = cache.update(key=key, value=value, conv_hidden_state=conv_hidden_state)
        
        return hidden_states, new_cache
    
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
        causal: bool = False,
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
            causal=causal,
        )
        query_states = rms_norm(query_states + attn_output, variance_epsilon=self.norm_eps)
        mlp_output, _conv_hidden_state = self.mlp.forward_packed(query_states, cu_seqlens_q)
        query_states = rms_norm(query_states + mlp_output, variance_epsilon=self.norm_eps)
        return query_states


class URM_Inner(nn.Module):
    
#region Init
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
        self.use_hrm = self.config.H_layers > 0
        self.L_layers = self.config.L_layers if self.use_hrm and self.config.L_layers > 0 else self.config.num_layers
            
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

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
        self.prefix_seq_len = self.puzzle_emb_len + self.config.num_memory_tokens

        if self.config.num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(
                trunc_normal_init_(
                    torch.empty(self.config.num_memory_tokens,self.config.hidden_size,dtype=self.forward_dtype,),
                    std=embed_init_std,
                )
            )

        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        self.init_rope_nD()

        self.layer_attention_window_sizes = _normalize_attention_window_sizes(
            self.inner_config.attention_window_size,
            self.L_layers,
        )
        self.layers = nn.ModuleList(
            [
                URMBlock(self.inner_config, attention_window_size=self.layer_attention_window_sizes[layer_idx])
                for layer_idx in range(self.L_layers)
            ]
        )
        
        if self.use_hrm:
            self.init_hrm()
            
        if self.config.patch_io_enabled:
            self.init_patch_io()
            
        self.prelude_layers = nn.ModuleList(
            [
                URMBlock(self.inner_config, attention_window_size=-1)
                for _ in range(self.config.prelude_layers)
            ]
        )
        self.coda_layers = nn.ModuleList(
            [
                URMBlock(self.inner_config, attention_window_size=-1)
                for _ in range(self.config.coda_layers)
            ]
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
        
    def init_rope_nD(self):
        if (self.inner_config.grid_depth > 0 and self.inner_config.grid_io > 0 and self.inner_config.grid_height > 0 and self.inner_config.grid_width > 0):
            self.rotary_emb = RotaryEmbedding4D(
                dim=self.inner_config.hidden_size // self.inner_config.num_heads,
                grid_depth=self.inner_config.grid_depth,
                grid_io=self.inner_config.grid_io,
                grid_height=self.inner_config.grid_height,
                grid_width=self.inner_config.grid_width,
                puzzle_emb_len=self.prefix_seq_len,
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
                puzzle_emb_len=self.prefix_seq_len,
                base=self.inner_config.rope_theta,
            )
        elif self.inner_config.grid_height > 0 and self.inner_config.grid_width > 0:
            self.rotary_emb = RotaryEmbedding2D(
                dim=self.inner_config.hidden_size // self.inner_config.num_heads,
                grid_height=self.inner_config.grid_height,
                grid_width=self.inner_config.grid_width,
                puzzle_emb_len=self.prefix_seq_len,
                base=self.inner_config.rope_theta,
            )
        else:
            self.rotary_emb = RotaryEmbedding(
                dim=self.inner_config.hidden_size // self.inner_config.num_heads,
                max_position_embeddings=self.inner_seq_len + self.prefix_seq_len,
                base=self.inner_config.rope_theta,
            )

    def init_hrm(self):
        self.H_layer_attention_window_sizes = []
        self.H_layers = nn.ModuleList()
        self.H_layer_attention_window_sizes = _normalize_attention_window_sizes(
                self.inner_config.attention_window_size,
                self.config.H_layers,
            )
        self.H_layers = nn.ModuleList(
            [
                URMBlock(self.inner_config, attention_window_size=self.H_layer_attention_window_sizes[layer_idx])
                for layer_idx in range(self.config.H_layers)
            ]
        )
        self.low_init_hidden = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

    def init_patch_io(self):
        if self.config.grid_height > 0 or self.config.grid_width > 0:
            self.inner_grid_height = -(self.config.grid_height // -self.config.patch_height)
            self.inner_grid_width = -(self.config.grid_width // -self.config.patch_width)
            self.inner_seq_len = self.inner_grid_height * self.inner_grid_width
            self.padded_grid_height = self.inner_grid_height * self.config.patch_height
            self.padded_grid_width = self.inner_grid_width * self.config.patch_width
        else:
            self.inner_seq_len = -(self.config.seq_len // -self.patch_area)
            self.padded_seq_len = self.inner_seq_len * self.patch_area
            
        patch_dim = self.patch_area * self.config.patch_pre_embedding_size
        self.pre_embedding = CastedLinear(self.config.vocab_size, self.config.patch_pre_embedding_size, bias=False)
        self.embed_tokens = CastedLinear(patch_dim, self.config.hidden_size, bias=False)
        self.lm_head = CastedLinear(self.config.hidden_size, patch_dim, bias=False)
        self.post_head = CastedLinear(self.config.patch_pre_embedding_size, self.config.vocab_size, bias=False)
#endregion

#region Patchify
    ### convert (B, T, C) to (B, T // patch_size, C)
    def _patchify(self, pixels: torch.Tensor) -> torch.Tensor:
        batch_size, _, channels = pixels.shape

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

    ### unconvert (B, T, C) to (B, T // patch_size, C)
    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch_size = patches.shape[0]
        channels = self.config.patch_pre_embedding_size

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
#endregion

    def _memory_embeddings(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.config.num_memory_tokens == 0:
            return None
        return self.memory_tokens.to(device=device).unsqueeze(0).expand(batch_size, -1, -1)


#region Label Embedding Noise
    def _separate_mode(self) -> str:
        return str(self.config.separate_mode or self.config.SeparateMode).upper()
    
    def _label_separate_C_enabled(self) -> bool:
        return bool(self.config.label_separate and self._separate_mode() == "C")

    def _sample_label_separate_C_alphas(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        if any(dim == 0 for dim in shape):
            return torch.empty(shape, dtype=torch.float32, device=device)
        rng_kwargs = {}
        generator_device = getattr(self.generator, "device", None)
        if generator_device is not None and torch.device(generator_device).type == device.type:
            rng_kwargs["generator"] = self.generator
        u = torch.rand(
            shape,
            dtype=torch.float32,
            device=device,
            **rng_kwargs,
        )
        u = u.clamp(torch.finfo(torch.float32).eps, 1.0 - torch.finfo(torch.float32).eps)
        return 1.0 - torch.sqrt(1.0 - u)

    def _label_separate_C_noise(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        scale = float(self.config.label_separate_C_noise_scale)
        if scale == 0 or any(dim == 0 for dim in shape):
            return torch.zeros(shape, dtype=dtype, device=device)
        rng_kwargs = {}
        generator_device = getattr(self.generator, "device", None)
        if generator_device is not None and torch.device(generator_device).type == device.type:
            rng_kwargs["generator"] = self.generator
        return torch.randn(shape, dtype=dtype, device=device, **rng_kwargs) * scale

    def _label_separate_C_mixed_embeddings(self, labels: torch.Tensor) -> torch.Tensor:
        if self.training:
            safe_labels = torch.where(labels >= 0, labels, torch.zeros_like(labels)).to(torch.int32)
            label_embeddings = self.embed_scale * self.embed_tokens(safe_labels)
            noise = self._label_separate_C_noise(
                tuple(label_embeddings.shape),
                label_embeddings.device,
                label_embeddings.dtype,
            )
            alpha = self._sample_label_separate_C_alphas(
                tuple(label_embeddings.shape[:-1]),
                label_embeddings.device,
            ).unsqueeze(-1)
            return alpha.to(label_embeddings.dtype) * label_embeddings + (1.0 - alpha).to(label_embeddings.dtype) * noise

        return self._label_separate_C_noise(
            (*tuple(labels.shape), self.config.hidden_size),
            labels.device,
            self.forward_dtype,
        )

    def _apply_label_separate_C_fixed(
        self,
        embedding: torch.Tensor,
        batch: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if batch is None or not self._label_separate_C_enabled() or "answer_mask" not in batch:
            return embedding

        answer_mask = batch["answer_mask"].to(device=embedding.device, dtype=torch.bool)
        if not bool(answer_mask.any().item()):
            return embedding

        labels = batch.get("labels")
        if labels is None:
            labels = torch.zeros(answer_mask.shape, dtype=torch.long, device=embedding.device)
        else:
            labels = labels.to(device=embedding.device, dtype=torch.long)

        mixed = self._label_separate_C_mixed_embeddings(labels)
        data_embedding = embedding[:, self.prefix_seq_len : self.prefix_seq_len + answer_mask.shape[1]]
        data_embedding = torch.where(answer_mask.unsqueeze(-1), mixed.to(data_embedding.dtype), data_embedding)
        embedding = embedding.clone()
        embedding[:, self.prefix_seq_len : self.prefix_seq_len + answer_mask.shape[1]] = data_embedding
        return embedding

    def _apply_label_separate_C_packed(
        self,
        embedding: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        if not self._label_separate_C_enabled() or "answer_mask" not in batch:
            return embedding

        answer_mask = batch["answer_mask"].to(device=batch["inputs"].device, dtype=torch.bool)
        if not bool(answer_mask.any().item()):
            return embedding

        labels = batch.get("labels")
        answer_count = int(answer_mask.sum().item())
        if labels is None:
            answer_labels = torch.zeros((answer_count,), dtype=torch.long, device=batch["inputs"].device)
        else:
            labels = labels.to(device=batch["inputs"].device, dtype=torch.long)
            if labels.numel() == answer_mask.numel():
                answer_labels = labels[answer_mask]
            elif labels.numel() == answer_count:
                answer_labels = labels
            else:
                answer_labels = torch.zeros((answer_count,), dtype=torch.long, device=batch["inputs"].device)

        answer_indices = token_indices[answer_mask]
        mixed = self._label_separate_C_mixed_embeddings(answer_labels)
        embedding = embedding.clone()
        embedding[answer_indices] = mixed.to(embedding.dtype)
        return embedding
#endregion

    def _input_embeddings(
        self,
        input: torch.Tensor,
        puzzle_identifiers: torch.Tensor,
        batch: Optional[Dict[str, torch.Tensor]] = None,
    ):
        if self.config.patch_io_enabled:
            if input.ndim == 3:
                _input = input.to(self.forward_dtype)
            else:
                _input = F.one_hot(input.to(torch.long), num_classes=self.config.vocab_size).to(self.forward_dtype)
            pixels = self.pre_embedding(_input)
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
        memory_embedding = self._memory_embeddings(embedding.shape[0], embedding.device)
        if memory_embedding is not None:
            embedding = torch.cat(
                (
                    embedding[:, : self.puzzle_emb_len],
                    memory_embedding,
                    embedding[:, self.puzzle_emb_len :],
                ),
                dim=-2,
            )
        embedding = self.embed_scale * embedding
        embedding = self._apply_label_separate_C_fixed(embedding, batch)

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
        if self.prefix_seq_len == 0:
            return axes

        total_len = token_indices.shape[0] + batch["puzzle_identifiers"].shape[0] * self.prefix_seq_len
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
        lengths = self._packed_lengths(batch) + self.prefix_seq_len
        cu_seqlens = F.pad(torch.cumsum(lengths, dim=0), (1, 0)).to(torch.int32)
        max_seqlen = int(lengths.max().item()) if lengths.numel() else 0
        return cu_seqlens, max_seqlen

    def _packed_data_token_indices(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        num_tokens = batch["inputs"].shape[0]
        if self.prefix_seq_len == 0:
            return torch.arange(num_tokens, device=batch["inputs"].device, dtype=torch.long)

        lengths = self._packed_lengths(batch)
        seq_ids = torch.repeat_interleave(
            torch.arange(lengths.shape[0], device=batch["inputs"].device, dtype=torch.long),
            lengths,
        )
        token_positions = torch.arange(num_tokens, device=batch["inputs"].device, dtype=torch.long)
        return token_positions + seq_ids * self.prefix_seq_len + self.prefix_seq_len

    def _packed_prefix_indices(
        self,
        batch: Dict[str, torch.Tensor],
        start: int = 0,
        length: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size = batch["puzzle_identifiers"].shape[0]
        length = self.prefix_seq_len if length is None else length
        if length == 0 or batch_size == 0:
            return torch.empty((0,), device=batch["inputs"].device, dtype=torch.long)

        data_offsets = batch["seq_offsets"][:-1].to(device=batch["inputs"].device, dtype=torch.long)
        seq_offsets = (
            data_offsets
            + torch.arange(batch_size, device=batch["inputs"].device, dtype=torch.long) * self.prefix_seq_len
            + start
        )
        prefix_offsets = torch.arange(length, device=batch["inputs"].device, dtype=torch.long)
        return (seq_offsets[:, None] + prefix_offsets[None, :]).reshape(-1)

    def _input_embeddings_packed(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        token_embedding = self.embed_tokens(batch["inputs"].to(torch.int32))
        token_indices = self._packed_data_token_indices(batch)

        if self.prefix_seq_len > 0:
            total_len = token_embedding.shape[0] + batch["puzzle_identifiers"].shape[0] * self.prefix_seq_len
            embedding = token_embedding.new_empty((total_len, self.config.hidden_size))
            embedding[token_indices] = token_embedding
            if self.puzzle_emb_len > 0:
                puzzle_embedding = self.puzzle_emb(batch["puzzle_identifiers"])
                pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
                if pad_count > 0:
                    puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
                embedding[self._packed_prefix_indices(batch, length=self.puzzle_emb_len)] = puzzle_embedding.view(-1, self.config.hidden_size)
            if self.config.num_memory_tokens > 0:
                batch_size = batch["puzzle_identifiers"].shape[0]
                memory_embedding = self.memory_tokens.to(device=token_embedding.device).expand(batch_size, -1, -1)
                embedding[self._packed_prefix_indices(batch, start=self.puzzle_emb_len, length=self.config.num_memory_tokens)] = memory_embedding.reshape(-1, self.config.hidden_size)
        else:
            embedding = token_embedding

        embedding = self.embed_scale * embedding
        embedding = self._apply_label_separate_C_packed(embedding, batch, token_indices)

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
            data_flat_ids = self.prefix_seq_len + position_ids[:, 0] * self.rotary_emb.grid_width + position_ids[:, 1]
            if self.prefix_seq_len > 0:
                flat_ids = data_flat_ids.new_empty((token_indices.shape[0] + batch["puzzle_identifiers"].shape[0] * self.prefix_seq_len,))
                flat_ids[token_indices] = data_flat_ids
                prefix_ids = torch.arange(self.prefix_seq_len, device=batch["inputs"].device, dtype=torch.long)
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
        data_pos_ids = self.prefix_seq_len + token_positions - data_offsets[seq_ids]
        if self.prefix_seq_len > 0:
            flat_ids = data_pos_ids.new_empty((token_indices.shape[0] + batch["puzzle_identifiers"].shape[0] * self.prefix_seq_len,))
            flat_ids[token_indices] = data_pos_ids
            prefix_ids = torch.arange(self.prefix_seq_len, device=batch["inputs"].device, dtype=torch.long)
            flat_ids[self._packed_prefix_indices(batch)] = prefix_ids.repeat(batch["puzzle_identifiers"].shape[0])
        else:
            flat_ids = data_pos_ids

        cos, sin = self.rotary_emb()
        cos = cos[flat_ids]
        sin = sin[flat_ids]
        if self.config.num_memory_tokens > 0:
            memory_indices = self._packed_prefix_indices(batch, start=self.puzzle_emb_len, length=self.config.num_memory_tokens)
            cos = cos.clone()
            sin = sin.clone()
            cos[memory_indices] = 1
            sin[memory_indices] = 0
        return cos, sin

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
        full_lengths = lengths + self.prefix_seq_len
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
        low_hidden = None
        if self.use_hrm:
            low_hidden = torch.empty(
                batch_size,
                carry_seq_len + self.prefix_seq_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            )
        return URMCarry(
            current_hidden=torch.empty(
                batch_size,
                carry_seq_len + self.prefix_seq_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
            current_low_hidden=low_hidden,
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: URMCarry) -> URMCarry:
        new_hidden = torch.where(
            reset_flag.view(-1, 1, 1),
            self.init_hidden,
            carry.current_hidden
        )
        new_low_hidden = carry.current_low_hidden
        if self.use_hrm:
            if new_low_hidden is None:
                raise RuntimeError("HRM mode requires current_low_hidden in the carry.")
            new_low_hidden = torch.where(
                reset_flag.view(-1, 1, 1),
                self.low_init_hidden,
                new_low_hidden,
            )
        return replace(carry, current_hidden=new_hidden, current_low_hidden=new_low_hidden)

    def _rotary_cos_sin(self, batch: Dict[str, torch.Tensor]):
        if "position_ids" in batch and isinstance(
            self.rotary_emb,
            (RotaryEmbedding2D, RotaryEmbedding3D, RotaryEmbedding4D),
        ):
            return self.rotary_emb(batch["position_ids"], prefix_seq_len=self.prefix_seq_len)
        cos_sin = self.rotary_emb()
        if self.config.num_memory_tokens == 0:
            return cos_sin

        masked_axes = []
        memory_start = self.puzzle_emb_len
        memory_end = self.prefix_seq_len
        for cos, sin in zip(cos_sin[::2], cos_sin[1::2]):
            cos = cos.clone()
            sin = sin.clone()
            cos[memory_start:memory_end] = 1
            sin[memory_start:memory_end] = 0
            masked_axes.extend([cos, sin])
        return tuple(masked_axes)

    def _sequence_lengths(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.config.variable_seq_lengths or "seq_lengths" not in batch:
            return None
        return batch["seq_lengths"].to(torch.int32) + self.prefix_seq_len

#region loops
    def _inject_inputs(self, hidden_states: torch.Tensor, input_embeddings: torch.Tensor) -> torch.Tensor:
        if not self.config.input_injection_enabled:
            return hidden_states
        return hidden_states + input_embeddings

    def _add_noise(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.noise_size <= 0:
            return hidden_states
        noise = torch.randn(
            hidden_states.shape,
            generator=self.generator,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            layout=hidden_states.layout,
        )
        return hidden_states + noise * self.config.noise_size

    def _run_token_layers(
        self,
        hidden_states: torch.Tensor,
        layers: nn.ModuleList,
        cos_sin: CosSin,
        sequence_lengths: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        cache: Optional[List[URMLayerInferenceCache]] = None,
    ) -> torch.Tensor:
        for i, layer in enumerate(layers):
            if cu_seqlens is not None:
                if max_seqlen is None:
                    raise RuntimeError("Packed token layers require max_seqlen.")
                hidden_states, _cache = layer.forward_packed(
                    cos_sin=cos_sin,
                    hidden_states=hidden_states,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    cache=cache[i] if cache is not None else None,
                )
            else:
                if cache is not None:
                    raise RuntimeError("Fixed-length token layers do not support inference caches.")
                hidden_states = layer.forward(
                    cos_sin=cos_sin,
                    hidden_states=hidden_states,
                    sequence_lengths=sequence_lengths,
                )
        return hidden_states

    def _run_recurrent_layers(
        self,
        hidden_states: torch.Tensor,
        input_embeddings: torch.Tensor,
        layers: nn.ModuleList,
        cos_sin: CosSin,
        sequence_lengths: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        diff_L: Optional[torch.Tensor] = None,
        grad_hook_factory=None,
        unrolled_idx: int = 0,
        force_injection: bool = False,
        add_noise: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        hidden_states = hidden_states + input_embeddings if force_injection else self._inject_inputs(hidden_states, input_embeddings)
        for layer in layers:
            pre_hidden_states = hidden_states
            if cu_seqlens is not None:
                if max_seqlen is None:
                    raise RuntimeError("Packed recurrent layers require max_seqlen.")
                hidden_states, _cache = layer.forward_packed(
                    cos_sin=cos_sin,
                    hidden_states=pre_hidden_states,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
            else:
                hidden_states = layer.forward(
                    cos_sin=cos_sin,
                    hidden_states=pre_hidden_states,
                    sequence_lengths=sequence_lengths,
                )
            if diff_L is not None:
                diff_L = diff_L + torch.abs(hidden_states - pre_hidden_states)
            if add_noise:
                hidden_states = self._add_noise(hidden_states)
            if grad_hook_factory is not None:
                hidden_states.register_hook(grad_hook_factory(unrolled_idx))
                unrolled_idx += 1
        return hidden_states, diff_L, unrolled_idx

    def _loop_layers(
        self,
        hidden_states: Optional[torch.Tensor],
        low_hidden_states: Optional[torch.Tensor],
        input_embeddings: torch.Tensor,
        H_layers: Optional[nn.ModuleList],
        L_layers: nn.ModuleList,
        H_cycles: int,
        L_cycles: int,
        seq_info: Dict[str, Union[torch.Tensor, int, CosSin, None]],
        diff_L: Optional[torch.Tensor] = None,
        grad_hook_factory=None,
        unrolled_idx: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], int]:
        no_grad_H_cycles = H_cycles - self.config.grad_H_cycles
        with torch.no_grad():
            for _ in range(no_grad_H_cycles):
                for _ in range(L_cycles):
                    if H_layers is not None:
                        layer_input = hidden_states + input_embeddings if self.config.input_injection_enabled else hidden_states
                        low_hidden_states, _, _ = self._run_recurrent_layers(
                            hidden_states=low_hidden_states,
                            input_embeddings=layer_input,
                            layers=L_layers,
                            force_injection=True,
                        )
                    else:
                        hidden_states, _, _ = self._run_recurrent_layers(
                            hidden_states=hidden_states,
                            input_embeddings=input_embeddings,
                            layers=L_layers,
                            add_noise=False,
                            **seq_info,
                        )
                        hidden_states = self._add_noise(hidden_states)
                if H_layers is not None:
                    hidden_states, _, _ = self._run_recurrent_layers(
                        hidden_states=hidden_states,
                        input_embeddings=low_hidden_states,
                        layers=H_layers,
                        force_injection=True,
                        **seq_info,
                    )

        for _ in range(self.config.grad_H_cycles):
            if H_layers is not None:
                for _ in range(L_cycles):
                    layer_input = hidden_states + input_embeddings if self.config.input_injection_enabled else hidden_states
                    low_hidden_states, diff_L, unrolled_idx = self._run_recurrent_layers(
                        hidden_states=low_hidden_states,
                        input_embeddings=layer_input,
                        layers=L_layers,
                        diff_L=diff_L,
                        grad_hook_factory=grad_hook_factory,
                        unrolled_idx=unrolled_idx,
                        force_injection=True,
                        **seq_info,
                    )
                hidden_states, _, unrolled_idx = self._run_recurrent_layers(
                    hidden_states=hidden_states,
                    input_embeddings=low_hidden_states,
                    layers=H_layers,
                    grad_hook_factory=grad_hook_factory,
                    unrolled_idx=unrolled_idx,
                    force_injection=True,
                    **seq_info,
                )
            else:
                for _ in range(L_cycles):
                    hidden_states, diff_L, unrolled_idx = self._run_recurrent_layers(
                        hidden_states=hidden_states,
                        input_embeddings=input_embeddings,
                        layers=L_layers,
                        diff_L=diff_L,
                        grad_hook_factory=grad_hook_factory,
                        unrolled_idx=unrolled_idx,
                        **seq_info,
                    )

        return hidden_states, low_hidden_states if H_layers is not None else None, diff_L, unrolled_idx
#endregion
  
    def _prefix_lm_key_value_indices(
        self,
        context_indices: torch.Tensor,
        answer_indices: torch.Tensor,
        cu_context: torch.Tensor,
        cu_answer: torch.Tensor,
    ) -> torch.Tensor:
        context_offsets = cu_context.detach().cpu().tolist()
        answer_offsets = cu_answer.detach().cpu().tolist()
        chunks = []
        for sample_idx in range(len(context_offsets) - 1):
            context_start, context_end = int(context_offsets[sample_idx]), int(context_offsets[sample_idx + 1])
            answer_start, answer_end = int(answer_offsets[sample_idx]), int(answer_offsets[sample_idx + 1])
            if context_end > context_start:
                chunks.append(context_indices[context_start:context_end])
            if answer_end > answer_start:
                chunks.append(answer_indices[answer_start:answer_end])
        if not chunks:
            return context_indices.new_empty((0,))
        return torch.cat(chunks, dim=0)

    def _run_prefix_lm_token_layers_packed(
        self,
        hidden_states: torch.Tensor,
        layers: nn.ModuleList,
        cos_sin: CosSin,
        context_indices: torch.Tensor,
        answer_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_context: torch.Tensor,
        cu_answer: torch.Tensor,
        max_seqlen: int,
        max_context_len: int,
        max_answer_len: int,
        add_noise: bool = False,
    ) -> torch.Tensor:
        if len(layers) == 0:
            return hidden_states
        if answer_indices.numel() == 0:
            return self._run_token_layers(
                hidden_states=hidden_states,
                layers=layers,
                cos_sin=cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        context_cos_sin = self._slice_packed_cos_sin(cos_sin, context_indices)
        answer_cos_sin = self._slice_packed_cos_sin(cos_sin, answer_indices)
        key_value_indices = self._prefix_lm_key_value_indices(
            context_indices,
            answer_indices,
            cu_context,
            cu_answer,
        )
        key_value_cos_sin = self._slice_packed_cos_sin(cos_sin, key_value_indices)

        for layer in layers:
            next_states = hidden_states.clone()
            context_states = hidden_states[context_indices]
            if context_states.numel() > 0:
                context_states, _cache = layer.forward_packed(
                    cos_sin=context_cos_sin,
                    hidden_states=context_states,
                    cu_seqlens=cu_context,
                    max_seqlen=max_context_len,
                )
                next_states[context_indices] = context_states

            answer_states = hidden_states[answer_indices]
            if answer_states.numel() > 0:
                key_value_states = next_states[key_value_indices]
                answer_states = layer.forward_cross_packed(
                    query_cos_sin=answer_cos_sin,
                    key_value_cos_sin=key_value_cos_sin,
                    query_states=answer_states,
                    key_value_states=key_value_states,
                    cu_seqlens_q=cu_answer,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_answer_len,
                    max_seqlen_k=max_seqlen,
                    causal=True,
                )
                next_states[answer_indices] = answer_states

            hidden_states = self._add_noise(next_states) if add_noise else next_states

        return hidden_states
    
    def get_grad_hook_factory(self):
        if not self.config.grad_logging_enabled or not global_logger.is_log or not self.training:
            return None

        grad_norms = {}
        layers_per_H_cycle = self.config.L_cycles * len(self.layers)
        if self.use_hrm:
            layers_per_H_cycle += len(self.H_layers)
        total_unrolled = self.config.grad_H_cycles * layers_per_H_cycle

        def make_grad_hook(idx):
            def hook(grad):
                grad_norms[idx] = grad.detach().norm().item()
                if len(grad_norms) == total_unrolled:
                    norm_tensor = torch.tensor([grad_norms[i] for i in range(total_unrolled)])
                    global_logger.store("grad_norm_per_layer", norm_tensor)
            return hook

        return make_grad_hook

#region Some forward 
    def forward(
        self,
        carry: URMCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[URMCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        if self.config.forward_mode == "prefix_lm":
            return self.forward_prefix_lm_packed(carry, batch)
        if self.config.forward_mode == "answer_only":
            return self.forward_answer_only_packed(carry, batch)
        
        # create input
        is_packed = self.config.variable_seq_lengths
        if is_packed:
            input_embeddings, token_indices = self._input_embeddings_packed(batch)
            cu_seqlens, max_seqlen = self._packed_cu_seqlens(batch)
            cos_sin = self._rotary_cos_sin_packed(batch, token_indices)
            seq_info = dict(
                cos_sin=cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        else:
            input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"], batch=batch)
            token_indices, cu_seqlens, max_seqlen = None, None, None
            cos_sin = self._rotary_cos_sin(batch)
            seq_info = dict(
                cos_sin=cos_sin,
                sequence_lengths=self._sequence_lengths(batch),
            )
        input_embeddings = self._run_token_layers(
            hidden_states=input_embeddings,
            layers=self.prelude_layers,
            **seq_info,
        )
        grad_hook_factory = self.get_grad_hook_factory()

        hidden_states = carry.current_hidden
        low_hidden_states = carry.current_low_hidden
        _unrolled_idx = 0
        diff_L = torch.zeros_like(hidden_states) if self.config.diff_L_loss_enabled else None

        hidden_states, low_hidden_states, diff_L, _unrolled_idx = self._loop_layers(
            hidden_states=hidden_states if self.use_hrm else None,
            low_hidden_states=low_hidden_states if self.use_hrm else hidden_states,
            input_embeddings=input_embeddings,
            H_layers=self.H_layers if self.use_hrm else None,
            L_layers=self.layers,
            H_cycles=self.config.H_cycles,
            L_cycles=self.config.L_cycles,
            seq_info=seq_info,
            diff_L=diff_L,
            grad_hook_factory=grad_hook_factory,
            unrolled_idx=_unrolled_idx,
        )

        # detach and store carry
        new_carry = replace(
            carry,
            current_hidden=hidden_states.detach(),
            current_low_hidden=low_hidden_states.detach() if low_hidden_states is not None else None,
        )
        
        # coda layers
        head_hidden_states = self._run_token_layers(
            hidden_states=hidden_states,
            layers=self.coda_layers,
            **seq_info,
        )

        if is_packed:
            assert token_indices is not None and cu_seqlens is not None
            output = self.lm_head(head_hidden_states[token_indices])
            q_logits = self.q_head(head_hidden_states[cu_seqlens[:-1].to(torch.long)]).to(torch.float32)
        else:
            output = self.lm_head(head_hidden_states)[:, self.prefix_seq_len:]
            q_logits = self.q_head(head_hidden_states[:, 0]).to(torch.float32)

        if self.config.patch_io_enabled:
            output = self.post_head(self._unpatchify(output))
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), diff_L

    forward_packed = forward

    def forward_prefix_lm_packed(
        self,
        carry: URMCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[URMCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
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
        context_indices = self._packed_context_indices(input_embeddings, answer_indices)

        input_embeddings = self._run_prefix_lm_token_layers_packed(
            hidden_states=input_embeddings,
            layers=self.prelude_layers,
            cos_sin=cos_sin,
            context_indices=context_indices,
            answer_indices=answer_indices,
            cu_seqlens=cu_seqlens,
            cu_context=cu_context,
            cu_answer=cu_answer,
            max_seqlen=max_seqlen,
            max_context_len=max_context_len,
            max_answer_len=max_answer_len,
        )

        hidden_states = carry.current_hidden
        if hidden_states.shape[0] != input_embeddings.shape[0]:
            raise RuntimeError(
                f"Packed carry/input length mismatch: carry={hidden_states.shape[0]} input={input_embeddings.shape[0]}"
            )

        if len(self.context_layers) > 0 and context_indices.numel() > 0:
            context_states = self._inject_inputs(
                hidden_states[context_indices],
                input_embeddings[context_indices],
            )
            context_cos_sin = self._slice_packed_cos_sin(cos_sin, context_indices)
            for layer in self.context_layers:
                context_states, _cache = layer.forward_packed(
                    cos_sin=context_cos_sin,
                    hidden_states=context_states,
                    cu_seqlens=cu_context,
                    max_seqlen=max_context_len,
                )
            hidden_states = hidden_states.clone()
            hidden_states[context_indices] = context_states

        def run_cycles(states: torch.Tensor, *, add_noise: bool) -> torch.Tensor:
            for _ in range(self.config.L_cycles):
                states = self._inject_inputs(states, input_embeddings)
                states = self._run_prefix_lm_token_layers_packed(
                    hidden_states=states,
                    layers=self.layers,
                    cos_sin=cos_sin,
                    context_indices=context_indices,
                    answer_indices=answer_indices,
                    cu_seqlens=cu_seqlens,
                    cu_context=cu_context,
                    cu_answer=cu_answer,
                    max_seqlen=max_seqlen,
                    max_context_len=max_context_len,
                    max_answer_len=max_answer_len,
                    add_noise=add_noise,
                )
            return states

        no_grad_H_cycles = self.config.H_cycles - self.config.grad_H_cycles
        if no_grad_H_cycles > 0:
            with torch.no_grad():
                for _ in range(no_grad_H_cycles):
                    hidden_states = run_cycles(hidden_states, add_noise=False)
                    hidden_states = self._add_noise(hidden_states)

        for _ in range(self.config.grad_H_cycles):
            hidden_states = run_cycles(hidden_states, add_noise=True)

        new_carry = replace(carry, current_hidden=hidden_states.detach())
        head_hidden_states = self._run_prefix_lm_token_layers_packed(
            hidden_states=hidden_states,
            layers=self.coda_layers,
            cos_sin=cos_sin,
            context_indices=context_indices,
            answer_indices=answer_indices,
            cu_seqlens=cu_seqlens,
            cu_context=cu_context,
            cu_answer=cu_answer,
            max_seqlen=max_seqlen,
            max_context_len=max_context_len,
            max_answer_len=max_answer_len,
        )
        output = self.lm_head(head_hidden_states[answer_indices])
        q_logits = self.q_head(head_hidden_states[cu_seqlens[:-1].to(torch.long)]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), None

    def forward_answer_only_packed(
        self,
        carry: URMCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[URMCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        input_embeddings, token_indices = self._input_embeddings_packed(batch)
        cu_seqlens, max_seqlen = self._packed_cu_seqlens(batch)
        cos_sin = self._rotary_cos_sin_packed(batch, token_indices)
        input_embeddings = self._run_token_layers(
            hidden_states=input_embeddings,
            layers=self.prelude_layers,
            cos_sin=cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
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
            head_hidden_states = self._run_token_layers(
                hidden_states=hidden_states,
                layers=self.coda_layers,
                cos_sin=cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            output = self.lm_head(head_hidden_states.new_empty((0, head_hidden_states.shape[-1])))
            q_logits = self.q_head(head_hidden_states[cu_seqlens[:-1].to(torch.long)]).to(torch.float32)
            return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), None

        context_indices = self._packed_context_indices(hidden_states, answer_indices)
        context_states = self._inject_inputs(
            hidden_states[context_indices],
            input_embeddings[context_indices],
        )
        context_cos_sin = self._slice_packed_cos_sin(cos_sin, context_indices)
        for layer in self.context_layers:
            context_states, _cache = layer.forward_packed(
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

        no_grad_H_cycles = self.config.H_cycles - self.config.grad_H_cycles
        if no_grad_H_cycles > 0:
            with torch.no_grad():
                for _ in range(no_grad_H_cycles):
                    hidden_states = run_answer_layers(hidden_states)

        for _ in range(self.config.grad_H_cycles):
            hidden_states = run_answer_layers(hidden_states)
        new_carry = replace(carry, current_hidden=hidden_states.detach())
        head_hidden_states = self._run_token_layers(
            hidden_states=hidden_states,
            layers=self.coda_layers,
            cos_sin=cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        output = self.lm_head(head_hidden_states[answer_indices])
        q_logits = self.q_head(head_hidden_states[cu_seqlens[:-1].to(torch.long)]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), None
#endregion

class URM(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = URMConfig(**config_dict)
        self.inner = URM_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def _rng_kwargs(self, device: torch.device) -> Dict[str, torch.Generator]:
        generator_device = getattr(self.inner.generator, "device", None)
        if generator_device is not None and torch.device(generator_device).type == device.type:
            return {"generator": self.inner.generator}
        return {}

    def _effective_answer_initial_mode(self) -> str:
        return self.config.answer_initial_mode

    def _use_label_answer_initial(self) -> bool:
        return self.training or self.config.answer_initial_use_labels_in_eval

    def _empty_initial_hidden(self, shape: Tuple[int, ...], device: torch.device, low: bool = False) -> torch.Tensor:
        mode = self._effective_answer_initial_mode()
        dtype = self.inner.forward_dtype
        if any(dim == 0 for dim in shape):
            return torch.empty(shape, dtype=dtype, device=device)
        if mode == "default":
            init_hidden = self.inner.low_init_hidden if low else self.inner.init_hidden
            return init_hidden.to(device=device, dtype=dtype).expand(shape)
        if mode == "black":
            return torch.zeros(shape, dtype=dtype, device=device)
        if mode == "random":
            hidden = torch.randn(
                shape,
                dtype=dtype,
                device=device,
                **self._rng_kwargs(device),
            )
            return hidden * self.config.answer_initial_random_std
        return torch.zeros(shape, dtype=dtype, device=device)

    def _answer_initial_eps(self, shape: Tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        scale = float(self.config.answer_initial_C_noise_scale)
        if scale == 0 or any(dim == 0 for dim in shape):
            return torch.zeros(shape, dtype=dtype, device=device)
        return torch.randn(
            shape,
            dtype=dtype,
            device=device,
            **self._rng_kwargs(device),
        ) * scale

    def _sample_C_alphas(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        if any(dim == 0 for dim in shape):
            return torch.empty(shape, dtype=torch.float32, device=device)
        if self.config.answer_initial_C_noise_distribution == "uniform":
            return torch.rand(
                shape,
                dtype=torch.float32,
                device=device,
                **self._rng_kwargs(device),
            )

        alpha = float(self.config.answer_initial_C_beta_alpha)
        beta = float(self.config.answer_initial_C_beta_beta)
        u = torch.rand(
            shape,
            dtype=torch.float32,
            device=device,
            **self._rng_kwargs(device),
        ).clamp(torch.finfo(torch.float32).eps, 1.0 - torch.finfo(torch.float32).eps)
        if math.isclose(beta, 1.0):
            return u.pow(1.0 / alpha)
        if math.isclose(alpha, 1.0):
            return 1.0 - u.pow(1.0 / beta)

        concentration1 = torch.full(shape, alpha, dtype=torch.float32, device=device)
        concentration0 = torch.full(shape, beta, dtype=torch.float32, device=device)
        return torch.distributions.Beta(concentration1, concentration0).sample()

    def _mix_C_label_embeddings(self, label_embeddings: torch.Tensor) -> torch.Tensor:
        alpha = self._sample_C_alphas(
            tuple(label_embeddings.shape[:-1]),
            label_embeddings.device,
        ).unsqueeze(-1)
        noise_weight = torch.sqrt((1.0 - alpha.square()).clamp_min(0.0))
        eps = self._answer_initial_eps(
            tuple(label_embeddings.shape),
            label_embeddings.device,
            label_embeddings.dtype,
        )
        return alpha.to(label_embeddings.dtype) * label_embeddings + noise_weight.to(label_embeddings.dtype) * eps

    def _sample_D_ratios(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        distribution = self.config.answer_initial_D_ratio_distribution
        if distribution == "constant":
            ratio = float(self.config.answer_initial_D_ratio_max)
            return torch.full(shape, ratio, dtype=torch.float32, device=device)
        if distribution == "normal":
            ratios = torch.randn(
                shape,
                dtype=torch.float32,
                device=device,
                **self._rng_kwargs(device),
            )
            ratios = ratios * float(self.config.answer_initial_D_ratio_std)
            ratios = ratios + float(self.config.answer_initial_D_ratio_mean)
            return ratios.clamp(0.0, 1.0)

        ratio_min = float(self.config.answer_initial_D_ratio_min)
        ratio_max = float(self.config.answer_initial_D_ratio_max)
        if ratio_min == ratio_max:
            return torch.full(shape, ratio_min, dtype=torch.float32, device=device)
        ratios = torch.rand(
            shape,
            dtype=torch.float32,
            device=device,
            **self._rng_kwargs(device),
        )
        return ratios * (ratio_max - ratio_min) + ratio_min

    def _random_token_candidates(self, device: torch.device) -> torch.Tensor:
        token_max = self.config.answer_initial_random_token_max
        if token_max is None:
            token_max = self.config.vocab_size - 1
        token_min = int(self.config.answer_initial_random_token_min)
        token_max = int(token_max)

        candidates = torch.arange(token_min, token_max + 1, dtype=torch.long, device=device)
        candidates = candidates[
            (candidates != int(self.config.answer_initial_pad_token_id))
            & (candidates != int(self.config.answer_initial_eos_token_id))
        ]
        candidates = candidates[(0 <= candidates) & (candidates < self.config.vocab_size)]
        return candidates

    def _random_non_special_tokens(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        candidates = self._random_token_candidates(device)
        candidate_indices = torch.randint(
            candidates.numel(),
            shape,
            dtype=torch.long,
            device=device,
            **self._rng_kwargs(device),
        )
        return candidates[candidate_indices]

    def _fixed_answer_mask(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "answer_mask" in batch:
            return batch["answer_mask"].to(device=batch["inputs"].device, dtype=torch.bool)
        return batch["labels"] != -100

    def _label_embeddings(self, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = labels.to(device=labels.device, dtype=torch.long)
        valid = (labels >= 0) & (labels < self.config.vocab_size)
        safe_labels = torch.where(valid, labels, torch.zeros_like(labels))
        embeddings = self.inner.embed_tokens(safe_labels.to(torch.int32))
        return self.inner.embed_scale * embeddings, valid

    def _initial_answer_embeddings(
        self,
        labels: torch.Tensor,
        answer_mask: torch.Tensor,
        replace_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mode = self._effective_answer_initial_mode()
        if mode == "noised_label_C":
            embeddings, valid_labels = self._label_embeddings(labels)
            embeddings = self._mix_C_label_embeddings(embeddings)
            return embeddings, answer_mask & valid_labels

        if mode != "noised_label_D":
            embeddings, valid_labels = self._label_embeddings(labels)
            return embeddings, answer_mask & valid_labels

        labels = labels.to(device=labels.device, dtype=torch.long)
        valid_labels = (labels >= 0) & (labels < self.config.vocab_size)
        safe_labels = torch.where(valid_labels, labels, torch.zeros_like(labels))
        if replace_probs is None:
            replace_probs = self._sample_D_ratios(tuple(safe_labels.shape), safe_labels.device)
        replace_probs = replace_probs.to(device=safe_labels.device, dtype=torch.float32)
        replace_mask = answer_mask & valid_labels & (
            torch.rand(
                tuple(safe_labels.shape),
                dtype=torch.float32,
                device=safe_labels.device,
                **self._rng_kwargs(safe_labels.device),
            ) < replace_probs
        )
        random_tokens = self._random_non_special_tokens(tuple(safe_labels.shape), safe_labels.device)
        corrupted = torch.where(replace_mask, random_tokens, safe_labels)
        embeddings = self.inner.embed_scale * self.inner.embed_tokens(corrupted.to(torch.int32))
        return embeddings, answer_mask & valid_labels

    def _make_fixed_initial_hidden(self, batch: Dict[str, torch.Tensor], low: bool = False) -> torch.Tensor:
        batch_size = batch["inputs"].shape[0]
        seq_len = batch["inputs"].shape[1]
        mode = self._effective_answer_initial_mode()
        hidden = self._empty_initial_hidden(
            (batch_size, seq_len + self.inner.prefix_seq_len, self.config.hidden_size),
            batch["inputs"].device,
            low=low,
        )
        if mode not in {"noised_label_C", "noised_label_D"}:
            return hidden

        answer_mask = self._fixed_answer_mask(batch)
        valid_mask = answer_mask
        if mode == "noised_label_C" and not self._use_label_answer_initial():
            embeddings = self._answer_initial_eps(
                (batch_size, seq_len, self.config.hidden_size),
                batch["inputs"].device,
                hidden.dtype,
            )
        elif mode == "noised_label_D" and not self._use_label_answer_initial():
            random_tokens = self._random_non_special_tokens((batch_size, seq_len), batch["inputs"].device)
            embeddings = self.inner.embed_scale * self.inner.embed_tokens(random_tokens.to(torch.int32))
        else:
            replace_probs = None
            if mode == "noised_label_D":
                replace_probs = self._sample_D_ratios(
                    (batch_size,),
                    batch["inputs"].device,
                ).view(batch_size, 1)
            embeddings, valid_mask = self._initial_answer_embeddings(batch["labels"], answer_mask, replace_probs)
        data_hidden = hidden[:, self.inner.prefix_seq_len : self.inner.prefix_seq_len + seq_len]
        data_hidden = torch.where(valid_mask.unsqueeze(-1), embeddings.to(hidden.dtype), data_hidden)
        hidden = hidden.clone()
        hidden[:, self.inner.prefix_seq_len : self.inner.prefix_seq_len + seq_len] = data_hidden
        return hidden

    def _packed_answer_labels(self, batch: Dict[str, torch.Tensor], answer_data_mask: torch.Tensor) -> torch.Tensor:
        labels = batch["labels"]
        if labels.numel() == batch["inputs"].numel():
            return labels[answer_data_mask]
        return labels

    def _make_packed_initial_hidden(self, batch: Dict[str, torch.Tensor], low: bool = False) -> torch.Tensor:
        lengths = self.inner._packed_lengths(batch)
        mode = self._effective_answer_initial_mode()
        total_len = int((lengths + self.inner.prefix_seq_len).sum().item())
        hidden = self._empty_initial_hidden(
            (total_len, self.config.hidden_size),
            batch["inputs"].device,
            low=low,
        )
        if mode not in {"noised_label_C", "noised_label_D"}:
            return hidden

        answer_data_mask = self.inner._packed_answer_mask(batch)
        answer_indices = self.inner._packed_data_token_indices(batch)[answer_data_mask]
        if answer_indices.numel() == 0:
            return hidden
        if mode == "noised_label_C" and not self._use_label_answer_initial():
            embeddings = self._answer_initial_eps(
                (answer_indices.numel(), self.config.hidden_size),
                batch["inputs"].device,
                hidden.dtype,
            )
            valid_mask = torch.ones((answer_indices.numel(),), dtype=torch.bool, device=answer_indices.device)
        elif mode == "noised_label_D" and not self._use_label_answer_initial():
            random_tokens = self._random_non_special_tokens((answer_indices.numel(),), batch["inputs"].device)
            embeddings = self.inner.embed_scale * self.inner.embed_tokens(random_tokens.to(torch.int32))
            valid_mask = torch.ones((answer_indices.numel(),), dtype=torch.bool, device=answer_indices.device)
        else:
            answer_labels = self._packed_answer_labels(batch, answer_data_mask)
            replace_probs = None
            if mode == "noised_label_D":
                seq_ids = torch.repeat_interleave(
                    torch.arange(lengths.shape[0], device=batch["inputs"].device, dtype=torch.long),
                    lengths,
                )
                sample_ratios = self._sample_D_ratios((lengths.shape[0],), batch["inputs"].device)
                replace_probs = sample_ratios[seq_ids[answer_data_mask]]
            embeddings, valid_mask = self._initial_answer_embeddings(
                answer_labels,
                torch.ones_like(answer_labels, dtype=torch.bool, device=answer_labels.device),
                replace_probs,
            )
        hidden = hidden.clone()
        hidden[answer_indices[valid_mask]] = embeddings[valid_mask].to(hidden.dtype)
        return hidden

    def _make_initial_low_hidden(self, batch: Dict[str, torch.Tensor], packed: bool) -> Optional[torch.Tensor]:
        if not self.inner.use_hrm:
            return None
        if packed:
            return self._make_packed_initial_hidden(batch, low=True)
        return self._make_fixed_initial_hidden(batch, low=True)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> URMCarry:
        batch_size = batch["puzzle_identifiers"].shape[0] if self.config.variable_seq_lengths else batch["inputs"].shape[0]
        if self.config.variable_seq_lengths:
            initial_hidden = self._make_packed_initial_hidden(batch)
            return URMCarry(
                current_hidden=initial_hidden,
                steps=torch.zeros((batch_size,), dtype=torch.int32, device=batch["inputs"].device),
                halted=torch.ones((batch_size,), dtype=torch.bool, device=batch["inputs"].device),
                current_data=dict(batch),
                current_low_hidden=self._make_initial_low_hidden(batch, packed=True),
            )

        seq_len = batch["inputs"].shape[1] if self.config.variable_seq_lengths else None
        base = self.inner.empty_carry(batch_size, seq_len=seq_len)
        initial_hidden = self._make_fixed_initial_hidden(batch)
        initial_low_hidden = self._make_initial_low_hidden(batch, packed=False)
        return URMCarry(
            current_hidden=initial_hidden,
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=batch["inputs"].device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=batch["inputs"].device),
            current_data=dict(batch),
            current_low_hidden=initial_low_hidden if initial_low_hidden is not None else base.current_low_hidden,
        )

    def _reset_fixed_carry(
        self,
        reset_flag: torch.Tensor,
        carry: URMCarry,
        current_data: Dict[str, torch.Tensor],
    ) -> URMCarry:
        initial_hidden = self._make_fixed_initial_hidden(current_data)
        new_hidden = torch.where(
            reset_flag.view(-1, 1, 1),
            initial_hidden,
            carry.current_hidden,
        )
        new_low_hidden = carry.current_low_hidden
        if self.inner.use_hrm:
            if new_low_hidden is None:
                raise RuntimeError("HRM mode requires current_low_hidden in the carry.")
            initial_low_hidden = self._make_fixed_initial_hidden(current_data, low=True)
            new_low_hidden = torch.where(
                reset_flag.view(-1, 1, 1),
                initial_low_hidden,
                new_low_hidden,
            )
        return replace(carry, current_hidden=new_hidden, current_low_hidden=new_low_hidden)

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
        current_hidden_lengths = [int(length) + self.inner.prefix_seq_len for length in current_lengths]
        current_hidden_offsets = [0]
        for length in current_hidden_lengths:
            current_hidden_offsets.append(current_hidden_offsets[-1] + int(length))

        old_hidden_offsets = None
        if not all(reset):
            old_hidden_lengths = (carry.current_data["seq_lengths"] + self.inner.prefix_seq_len).detach().cpu().tolist()
            old_hidden_offsets = [0]
            for length in old_hidden_lengths:
                old_hidden_offsets.append(old_hidden_offsets[-1] + int(length))

        hidden_chunks = []
        low_hidden_chunks = [] if self.inner.use_hrm else None
        if self.inner.use_hrm and carry.current_low_hidden is None:
            raise RuntimeError("HRM mode requires current_low_hidden in the carry.")
        initial_hidden = self._make_packed_initial_hidden(current_data)
        initial_low_hidden = self._make_packed_initial_hidden(current_data, low=True) if low_hidden_chunks is not None else None
        for idx, should_reset in enumerate(reset):
            if should_reset:
                hidden_chunks.append(initial_hidden[current_hidden_offsets[idx]:current_hidden_offsets[idx + 1]])
                if low_hidden_chunks is not None:
                    assert initial_low_hidden is not None
                    low_hidden_chunks.append(initial_low_hidden[current_hidden_offsets[idx]:current_hidden_offsets[idx + 1]])
            else:
                assert old_hidden_offsets is not None
                hidden_chunks.append(carry.current_hidden[old_hidden_offsets[idx]:old_hidden_offsets[idx + 1]])
                if low_hidden_chunks is not None:
                    low_hidden_chunks.append(carry.current_low_hidden[old_hidden_offsets[idx]:old_hidden_offsets[idx + 1]])

        if hidden_chunks:
            current_hidden = torch.cat(hidden_chunks, dim=0)
            current_low_hidden = torch.cat(low_hidden_chunks, dim=0) if low_hidden_chunks is not None else None
        else:
            current_hidden = torch.empty(
                0,
                self.config.hidden_size,
                dtype=self.inner.forward_dtype,
                device=current_data["inputs"].device,
            )
            current_low_hidden = (
                torch.empty(
                    0,
                    self.config.hidden_size,
                    dtype=self.inner.forward_dtype,
                    device=current_data["inputs"].device,
                )
                if low_hidden_chunks is not None
                else None
            )

        return replace(
            carry,
            current_hidden=current_hidden,
            current_low_hidden=current_low_hidden,
            current_data=current_data,
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
        target_hidden_len = target_seq_len + self.inner.prefix_seq_len

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
            lengths = seq_lengths.to(x1.device) + self.inner.prefix_seq_len
            return packed_norm_ratio_from_lengths(x1, x2, lengths)

        if seq_lengths is not None:
            lengths = (seq_lengths.to(x1.device) + self.inner.prefix_seq_len).clamp(max=x1.shape[1])
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
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v
            )
            for k, v in carry.current_data.items()
        }
        new_carry = self._reset_fixed_carry(carry.halted, carry, new_current_data)

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
                current_low_hidden=new_carry2.current_low_hidden,
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
        if self.config.forward_mode in {"answer_only", "prefix_lm"}:
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
                current_low_hidden=new_carry2.current_low_hidden,
            ),
            outputs,
        )
