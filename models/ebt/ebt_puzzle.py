"""Energy-Based Transformer (EBT) for non-autoregressive puzzle tasks.

Implements the EBT approach (arXiv:2507.02092) adapted for puzzle reasoning:
predictions are iteratively refined via MCMC gradient descent on a learned
energy function.  Compatible with the existing ``pretrain.py`` training loop
through the ``EBTLossHead`` wrapper.
"""

from typing import Tuple, Dict, Set, Any, Optional
from dataclasses import dataclass
from contextlib import nullcontext
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel, ConfigDict

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm,
    ConvSwiGLU,
    RotaryEmbedding,
    CosSin,
    CastedEmbedding,
    CastedLinear,
    apply_rotary_pos_emb,
)
from models.sparse_embedding import CastedSparseEmbedding
from models.losses import IGNORE_LABEL_ID, stablemax_cross_entropy, softmax_cross_entropy


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class EBTPuzzleConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Architecture (matching URM interface)
    batch_size: int
    seq_len: int
    vocab_size: int
    num_layers: int
    hidden_size: int
    num_heads: int
    expansion: float = 4.0
    num_puzzle_identifiers: int = 0
    puzzle_emb_ndim: int = 0
    causal: bool = False
    pos_encodings: str = "rope"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    forward_dtype: str = "bfloat16"

    # EBT-specific
    mcmc_num_steps: int = 12
    mcmc_step_size: float = 100.0
    mcmc_step_size_learnable: bool = True
    truncate_mcmc: bool = True
    denoising_initial_condition: str = "zeros"  # "zeros" or "random_noise"
    clamp_grad_max: float = 0.0  # 0 = disabled

    # Inference step count (pretrain.py reads this via ``_get_loop_config``)
    loops: int = 16

    # Disable torch.compile – autograd.grad with create_graph is incompatible
    profile: bool = True


# ---------------------------------------------------------------------------
# Carry state
# ---------------------------------------------------------------------------

@dataclass
class EBTCarry:
    """Minimal carry: EBT does all MCMC steps in a single forward call."""
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Attention (standard PyTorch – supports higher-order gradients)
# ---------------------------------------------------------------------------

class EBTAttention(nn.Module):
    """Bidirectional multi-head attention using manual matmul.

    Flash-attention does not support ``create_graph=True`` in
    ``autograd.grad``, so we fall back to standard PyTorch ops here.
    """

    def __init__(self, hidden_size: int, head_dim: int, num_heads: int, num_key_value_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.scale = 1.0 / math.sqrt(head_dim)

        self.qkv_proj = CastedLinear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False,
        )
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, window_size: int = -1) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(B, S, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        q = qkv[:, :, : self.num_heads]
        k = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads]
        v = qkv[:, :, self.num_heads + self.num_key_value_heads :]

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # (B, num_heads, S, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.output_size)
        return self.o_proj(attn_output)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class EBTBlock(nn.Module):
    """Transformer block: Attention + ConvSwiGLU with post-norm residuals."""

    def __init__(self, config: EBTPuzzleConfig):
        super().__init__()
        self.self_attn = EBTAttention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
        )
        self.mlp = ConvSwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_output = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
        hidden_states = rms_norm(hidden_states + attn_output, variance_epsilon=self.norm_eps)
        mlp_output = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_output, variance_epsilon=self.norm_eps)
        return hidden_states


# ---------------------------------------------------------------------------
# EBT model
# ---------------------------------------------------------------------------

class EBTPuzzle(nn.Module):
    """Energy-Based Transformer for non-autoregressive puzzle tasks.

    Architecture overview:
    1. Embed input tokens and (optionally) prepend puzzle embeddings.
    2. Initialise predicted logits (B, S, V) as zeros or noise.
    3. Run N MCMC steps:
       a. Convert logits to soft embeddings via softmax + embedding weight.
       b. Add input embeddings and predicted embeddings.
       c. Forward through transformer layers.
       d. Compute scalar energy via energy head.
       e. Gradient descent: logits -= alpha * d(energy)/d(logits).
    4. Return final logits as predictions.

    With ``truncate_mcmc=True`` only the last MCMC step retains the
    computation graph (analogous to URM's truncated BPTT).
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = EBTPuzzleConfig(**config_dict)
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Token embeddings
        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )

        # Energy head: hidden_size -> 1  (scalar energy per position)
        self.energy_head = CastedLinear(self.config.hidden_size, 1, bias=False)

        # Puzzle embeddings
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            dim=self.config.hidden_size // self.config.num_heads,
            max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
            base=self.config.rope_theta,
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [EBTBlock(self.config) for _ in range(self.config.num_layers)]
        )

        # Learnable MCMC step size (alpha)
        self.alpha = nn.Parameter(
            torch.tensor(float(self.config.mcmc_step_size)),
            requires_grad=self.config.mcmc_step_size_learnable,
        )

    # -- helpers --

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
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

    # -- interface expected by pretrain.py (via loss head) --

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> EBTCarry:
        B = batch["inputs"].shape[0]
        device = batch["inputs"].device
        return EBTCarry(
            steps=torch.zeros(B, dtype=torch.int32, device=device),
            halted=torch.ones(B, dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: EBTCarry,
        batch: Dict[str, torch.Tensor],
        compute_target_q: bool = False,
    ) -> Tuple[EBTCarry, Dict[str, torch.Tensor]]:
        # --- handle carry (new data for halted examples) ---
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
            for k, v in carry.current_data.items()
        }

        inputs = new_current_data["inputs"]
        B, S = inputs.shape
        V = self.config.vocab_size

        # pretrain.py wraps evaluation in torch.inference_mode() which blocks
        # autograd.  We need gradients for the MCMC loop, so temporarily exit
        # inference mode when it is active.
        needs_inference_escape = not self.training and torch.is_inference_mode_enabled()
        grad_ctx = torch.inference_mode(mode=False) if needs_inference_escape else nullcontext()

        with grad_ctx:
            # --- input embeddings (with puzzle emb prepended) ---
            input_embeddings = self._input_embeddings(inputs, new_current_data["puzzle_identifiers"])
            cos_sin = self.rotary_emb()
            alpha = torch.clamp(self.alpha, min=0.0001)

            # --- initialise predicted logits ---
            if self.config.denoising_initial_condition == "zeros":
                predicted_logits = torch.zeros(B, S, V, device=inputs.device, dtype=torch.float32)
            elif self.config.denoising_initial_condition == "random_noise":
                predicted_logits = torch.randn(B, S, V, device=inputs.device, dtype=torch.float32) * 0.01
            else:
                raise ValueError(
                    f"Unknown denoising_initial_condition: {self.config.denoising_initial_condition}"
                )

            num_steps = self.config.mcmc_num_steps if self.training else self.config.loops

            # --- MCMC energy-minimisation loop ---
            with torch.set_grad_enabled(True):
                for step in range(num_steps):
                    predicted_logits = predicted_logits.detach().requires_grad_()

                    # logits -> soft embeddings
                    pred_probs = F.softmax(predicted_logits, dim=-1)  # (B, S, V) float32
                    pred_embeddings = self.embed_scale * torch.matmul(
                        pred_probs, self.embed_tokens.embedding_weight  # (V, D) float32
                    )  # (B, S, D) float32

                    # pad for puzzle-embedding positions (prepend zeros)
                    if self.puzzle_emb_len > 0:
                        pred_pad = torch.zeros(
                            B, self.puzzle_emb_len, self.config.hidden_size,
                            device=inputs.device, dtype=pred_embeddings.dtype,
                        )
                        pred_embeddings_full = torch.cat([pred_pad, pred_embeddings], dim=1)
                    else:
                        pred_embeddings_full = pred_embeddings

                    # combine input + predicted embeddings
                    hidden = input_embeddings + pred_embeddings_full.to(input_embeddings.dtype)

                    # transformer forward
                    for layer in self.layers:
                        hidden = layer(cos_sin=cos_sin, hidden_states=hidden)

                    # scalar energy (from non-puzzle positions)
                    energy = self.energy_head(hidden[:, self.puzzle_emb_len:])  # (B, S, 1)
                    energy_scalar = energy.to(torch.float32).sum()

                    # gradient of energy w.r.t. predicted logits
                    if self.config.truncate_mcmc:
                        create_graph = (step == num_steps - 1) and self.training
                    else:
                        create_graph = self.training

                    grad = torch.autograd.grad(
                        energy_scalar, predicted_logits, create_graph=create_graph
                    )[0]

                    # optional gradient clamping
                    if self.config.clamp_grad_max > 0:
                        clamp_val = self.config.clamp_grad_max / alpha.detach()
                        grad = torch.clamp(grad, -clamp_val, clamp_val)

                    # MCMC update
                    predicted_logits = predicted_logits - alpha * grad

        # --- outputs ---
        outputs: Dict[str, torch.Tensor] = {"logits": predicted_logits}

        new_carry = EBTCarry(
            steps=torch.ones(B, dtype=torch.int32, device=inputs.device),
            halted=torch.ones(B, dtype=torch.bool, device=inputs.device),
            current_data=new_current_data,
        )
        return new_carry, outputs


# ---------------------------------------------------------------------------
# Loss head (wraps EBTPuzzle for pretrain.py compatibility)
# ---------------------------------------------------------------------------

_LOSS_FNS = {
    "stablemax_cross_entropy": stablemax_cross_entropy,
    "softmax_cross_entropy": softmax_cross_entropy,
}


class EBTLossHead(nn.Module):
    """Computes cross-entropy loss on the final EBT predictions.

    Drop-in replacement for ``ACTLossHead`` – same call signature so that
    ``pretrain.py`` works without modifications.
    """

    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = _LOSS_FNS[loss_type]

    def initial_carry(self, *args: Any, **kwargs: Any) -> EBTCarry:
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Set[str] = set(),
        return_raw_outputs: bool = False,
        **model_kwargs: Any,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # --- mask & divisor (no grad needed) ---
        mask = labels != IGNORE_LABEL_ID
        loss_counts = mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

        # --- reconstruction loss (gradient required) ---
        lm_loss = (
            self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor
        ).sum()

        # --- metrics (no grad) ---
        with torch.no_grad():
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics: Dict[str, torch.Tensor] = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                    torch.tensor(0.0, device=labels.device),
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "lm_loss": lm_loss.detach(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # --- filter return outputs ---
        returned_outputs: Dict[str, torch.Tensor] = {}
        if return_raw_outputs:
            returned_outputs["raw_outputs"] = outputs  # type: ignore
        for k in return_keys:
            if k in outputs:
                returned_outputs[k] = outputs[k].detach()

        return (
            new_carry,
            lm_loss,
            metrics,
            returned_outputs,
            new_carry.halted.all(),
        )
