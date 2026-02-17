"""
TD3-based Reinforcement Learning Training for URM (Universal Reasoning Model).

This script fine-tunes a pretrained URM model using Twin Delayed DDPG (TD3).
Instead of truncated BPTT, each recurrent step is treated as a single-step
MDP transition, allowing gradient-free long-horizon credit assignment via
learned Q-functions.

RL Formulation:
  State:      z_t       = hidden states (carry) before step      [B, L, C]
  Action:     a_t       = residual output of transformer blocks  [B, L, C]
  Next state: z_{t+1}   = z_t + u + a_t  (u = input embeddings)
  Reward:     r_t       = -CE(lm_head(z_{t+1}), labels)         [B]
  Policy:     pi(z_t)   = transformer blocks (pretrained URM)
  Critic:     Q(z_t, a) = transformer-based state-action value   [B]

Usage:
  python rl_train.py \
    --checkpoint checkpoints/URM-sudoku-base \
    --data_path data/sudoku-extreme-1k-aug-1000 \
    --total_steps 50000
"""

import argparse
import copy
import math
import os
import re
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from models.layers import (
    Attention,
    SwiGLU,
    CastedLinear,
    RotaryEmbedding,
    rms_norm,
    CosSin,
)
from models.losses import stablemax_cross_entropy, IGNORE_LABEL_ID
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils import load_model_class

import pydantic


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    arch: ArchConfig
    data_path: str
    evaluators: List[EvaluatorConfig] = []
    global_batch_size: int
    epochs: int
    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    target_q_update_every: int
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float
    grad_accum_steps: int = 1
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    load_checkpoint: Optional[str] = None
    load_strict: bool = True
    load_optimizer_state: bool = True
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []
    loop_deltas: List[str] = []
    ema: bool = False
    ema_rate: float = 0.999
    use_muon: bool = False


# ---------------------------------------------------------------------------
# Checkpoint & Config Loading
# ---------------------------------------------------------------------------

def _resolve_checkpoint_path(path: str) -> Optional[str]:
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        pattern = re.compile(r"step_(\d+)(?:\.pt)?$")
        candidates = []
        for file_name in os.listdir(path):
            match = pattern.match(file_name)
            if match:
                candidates.append((int(match.group(1)), os.path.join(path, file_name)))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[-1][1]
    return None


def load_config_from_checkpoint_path(checkpoint_path: str) -> Optional[PretrainConfig]:
    resolved_path = _resolve_checkpoint_path(checkpoint_path)
    if resolved_path is None:
        return None
    checkpoint_dir = Path(resolved_path).parent
    candidates = [
        checkpoint_dir / "config.json",
        checkpoint_dir / "config.yaml",
        checkpoint_dir / "all_config.yaml",
        checkpoint_dir / ".hydra" / "config.yaml",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            conf = OmegaConf.load(candidate)
            as_dict = OmegaConf.to_container(conf, resolve=True)
            if isinstance(as_dict, dict):
                return PretrainConfig(**as_dict)
        except Exception:
            pass
        try:
            with open(candidate, "r") as f:
                config_dict = (
                    json.load(f) if candidate.suffix == ".json" else yaml.safe_load(f)
                )
            if isinstance(config_dict, dict):
                return PretrainConfig(**config_dict)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Reward Normalizer (Fix #3: running mean/std normalization)
# ---------------------------------------------------------------------------

class RewardNormalizer:
    """Welford online running mean/std for reward normalization."""

    def __init__(self, clip: float = 5.0):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.clip = clip

    def update(self, rewards: torch.Tensor):
        """Update running stats with a batch of rewards (vectorized)."""
        batch = rewards.detach().float().flatten()
        n = batch.numel()
        if n == 0:
            return
        batch_mean = batch.mean().item()
        batch_var = batch.var().item() if n > 1 else 0.0
        # Parallel Welford merge: combine batch stats with running stats
        new_count = self.count + n
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * n / new_count
        m_a = self.var * self.count
        m_b = batch_var * n
        M2 = m_a + m_b + delta ** 2 * self.count * n / new_count
        self.var = M2 / new_count if new_count > 0 else 0.0
        self.count = new_count

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards to zero-mean, unit-variance, clipped."""
        std = max(math.sqrt(self.var), 1e-8)
        return ((rewards - self.mean) / std).clamp(-self.clip, self.clip)


# ---------------------------------------------------------------------------
# Critic Network (Fix #5: richer state-action features + LayerNorm)
# ---------------------------------------------------------------------------

class CriticBlock(nn.Module):
    """Lightweight transformer block for the critic (Attention + SwiGLU)."""

    def __init__(self, hidden_size: int, num_heads: int, rms_norm_eps: float = 1e-5):
        super().__init__()
        self.self_attn = Attention(
            hidden_size=hidden_size,
            head_dim=hidden_size // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(hidden_size=hidden_size, expansion=4.0)
        self.norm_eps = rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_output = self.self_attn(
            cos_sin=cos_sin, hidden_states=hidden_states, window_size=-1
        )
        hidden_states = rms_norm(
            hidden_states + attn_output, variance_epsilon=self.norm_eps
        )
        mlp_output = self.mlp(hidden_states)
        hidden_states = rms_norm(
            hidden_states + mlp_output, variance_epsilon=self.norm_eps
        )
        return hidden_states


class TD3Critic(nn.Module):
    """
    State-action value function Q(s, a) -> scalar.

    Uses three types of features per position:
      - state [B, L, C]
      - action [B, L, C]
      - state * action (interaction) [B, L, C]
    Projects [B, L, 3C] -> [B, L, C], then transformer + pool -> scalar.
    Includes LayerNorm after projection for stable gradient flow.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int = 2,
        max_seq_len: int = 1024,
        rope_theta: float = 10000.0,
        forward_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.forward_dtype = forward_dtype

        # Fix #5: richer input (state, action, state*action)
        self.input_proj = CastedLinear(hidden_size * 3, hidden_size, bias=False)
        self.input_ln = nn.LayerNorm(hidden_size)

        self.layers = nn.ModuleList(
            [CriticBlock(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_size // num_heads,
            max_position_embeddings=max_seq_len,
            base=rope_theta,
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            CastedLinear(hidden_size, hidden_size, bias=False),
            nn.SiLU(),
            CastedLinear(hidden_size, 1, bias=True),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        cos_cached, sin_cached = self.rotary_emb()
        seq_len = state.shape[1]
        cos_sin = (cos_cached[:seq_len], sin_cached[:seq_len])

        s = state.to(self.forward_dtype)
        a = action.to(self.forward_dtype)
        # Fix #5: state, action, and their element-wise interaction
        x = torch.cat([s, a, s * a], dim=-1)  # [B, L, 3C]
        x = self.input_proj(x)  # [B, L, C]
        x = self.input_ln(x.float()).to(self.forward_dtype)

        for layer in self.layers:
            x = layer(cos_sin=cos_sin, hidden_states=x)
        x = x.mean(dim=1)  # [B, C]
        return self.value_head(x.float()).squeeze(-1)  # [B] in float32


# ---------------------------------------------------------------------------
# Replay Buffer (Fix #2: batch-level add, no Python loop)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    GPU-based replay buffer storing transitions in bfloat16.
    All data stays on GPU — no CPU transfers needed.
    next_state is NOT stored; reconstructed as state + input_emb + action.
    """

    def __init__(
        self,
        capacity: int,
        seq_len: int,
        hidden_size: int,
        label_seq_len: int,
        device: str = "cuda",
    ):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        shape = (capacity, seq_len, hidden_size)
        self.states = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        self.actions = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        # next_states omitted: reconstruct as states + input_embeddings + actions
        self.input_embeddings = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.labels = torch.zeros(
            (capacity, label_seq_len), dtype=torch.int32, device=device
        )

    def add_batch(
        self,
        states: torch.Tensor,      # [T, B, L, C] or [1, N, L, C]
        actions: torch.Tensor,
        rewards: torch.Tensor,      # [T, B] or [1, N]
        dones: torch.Tensor,
        input_embs: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Add transitions. All inputs already on GPU — zero-copy."""
        T, B = states.shape[0], states.shape[1]
        N = T * B

        # Flatten [T, B, ...] -> [N, ...]
        flat_s = states.reshape(N, *states.shape[2:]).to(torch.bfloat16)
        flat_a = actions.reshape(N, *actions.shape[2:]).to(torch.bfloat16)
        flat_e = input_embs.reshape(N, *input_embs.shape[2:]).to(torch.bfloat16)
        flat_r = rewards.reshape(N).float()
        flat_d = dones.reshape(N)
        flat_l = labels.reshape(N, labels.shape[-1]).to(torch.int32)

        indices = (torch.arange(N, device=self.device) + self.ptr) % self.capacity

        self.states[indices] = flat_s
        self.actions[indices] = flat_a
        self.input_embeddings[indices] = flat_e
        self.rewards[indices] = flat_r
        self.dones[indices] = flat_d
        self.labels[indices] = flat_l

        self.ptr = (self.ptr + N) % self.capacity
        self.size = min(self.size + N, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        """Sample batch. Returns (s, a, r, d, input_emb, labels). All on GPU."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.dones[indices],
            self.input_embeddings[indices],
            self.labels[indices],
        )


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def actor_step(
    layers: nn.ModuleList,
    cos_sin: CosSin,
    state: torch.Tensor,
    input_emb: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run one recurrent step through the actor (URM transformer blocks).

    Returns:
        next_state: z_{t+1} = state + input_emb + action  [B, L, C]
        action: residual produced by transformer blocks    [B, L, C]
    """
    h = state + input_emb
    for layer in layers:
        h = layer(cos_sin=cos_sin, hidden_states=h)
    action = h - (state + input_emb)
    next_state = h  # = state + input_emb + action
    return next_state, action


def compute_reward(
    lm_head: nn.Module,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    puzzle_emb_len: int,
) -> torch.Tensor:
    """Compute per-sample reward as negative cross-entropy. [B]"""
    logits = lm_head(hidden_states)[:, puzzle_emb_len:]
    min_len = min(logits.shape[1], labels.shape[1])
    logits = logits[:, :min_len]
    labels_trimmed = labels[:, :min_len]

    ce = stablemax_cross_entropy(logits, labels_trimmed)
    mask = labels_trimmed != IGNORE_LABEL_ID
    per_sample = (ce * mask).sum(-1) / mask.sum(-1).clamp(min=1)
    return -per_sample


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Polyak averaging: target = tau * source + (1 - tau) * target."""
    with torch.no_grad():
        t_params = list(target.parameters())
        s_params = list(source.parameters())
        torch._foreach_mul_(t_params, 1.0 - tau)
        torch._foreach_add_(t_params, s_params, alpha=tau)


def compute_exact_accuracy(
    lm_head: nn.Module,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    puzzle_emb_len: int,
) -> float:
    """Compute exact match accuracy (all tokens correct per sample)."""
    logits = lm_head(hidden_states)[:, puzzle_emb_len:]
    min_len = min(logits.shape[1], labels.shape[1])
    preds = logits[:, :min_len].argmax(dim=-1)
    labels_trimmed = labels[:, :min_len]
    mask = labels_trimmed != IGNORE_LABEL_ID
    correct = mask & (preds == labels_trimmed)
    exact = (correct.sum(-1) == mask.sum(-1)).float().mean().item()
    return exact


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_actor(
    checkpoint_path: str,
    data_path: str,
    device: str = "cuda",
    batch_size: Optional[int] = None,
) -> Tuple[nn.Module, dict]:
    """Load a pretrained URM model from checkpoint."""
    config = load_config_from_checkpoint_path(checkpoint_path)
    if config is None:
        raise ValueError(f"Could not load config from {checkpoint_path}")
    if data_path:
        config.data_path = data_path

    ds_config = PuzzleDatasetConfig(
        seed=config.seed, dataset_path=config.data_path,
        rank=0, num_replicas=1,
        global_batch_size=config.global_batch_size,
        test_set_mode=True, epochs_per_iter=1,
    )
    dataset = PuzzleDataset(ds_config, split="train")
    metadata = dataset.metadata

    effective_batch_size = batch_size if batch_size is not None else config.global_batch_size

    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=effective_batch_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
    )
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    model = model_cls(model_cfg)
    model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)
    model = model.to(device)

    resolved_ckpt = _resolve_checkpoint_path(checkpoint_path)
    print(f"Loading weights from {resolved_ckpt}")
    checkpoint = torch.load(resolved_ckpt, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key.replace("_orig_mod.", "")] = value

    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Loaded checkpoint (strict=True).")
    except RuntimeError as e:
        print(f"Strict load failed, retrying with strict=False: {e}")
        model.load_state_dict(new_state_dict, strict=False)

    actor = model.model
    info = {
        "config": config, "metadata": metadata,
        "hidden_size": model_cfg.get("hidden_size", 512),
        "num_heads": model_cfg.get("num_heads", 8),
        "seq_len": metadata.seq_len,
        "puzzle_emb_len": actor.inner.puzzle_emb_len,
        "rope_theta": model_cfg.get("rope_theta", 10000.0),
    }
    return actor, info


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_actor(
    actor: nn.Module,
    dataloader: DataLoader,
    num_steps: int,
    device: str = "cuda",
    max_batches: int = 50,
) -> Dict[str, float]:
    """Run inference without noise and compute accuracy metrics."""
    actor.eval()
    inner = actor.inner
    layers = inner.layers
    cos_sin = inner.rotary_emb()
    puzzle_emb_len = inner.puzzle_emb_len

    total_exact = 0.0
    total_reward = 0.0
    total_count = 0

    for i, (set_name, batch, global_batch_size) in enumerate(dataloader):
        if i >= max_batches:
            break
        batch_gpu = {k: v.to(device) for k, v in batch.items()}
        B = batch_gpu["inputs"].shape[0]

        input_emb = inner._input_embeddings(
            batch_gpu["inputs"], batch_gpu["puzzle_identifiers"]
        )
        hidden = inner.init_hidden.expand(B, input_emb.shape[1], -1).clone()

        for _ in range(num_steps):
            hidden, _ = actor_step(layers, cos_sin, hidden, input_emb)

        reward = compute_reward(inner.lm_head, hidden, batch_gpu["labels"], puzzle_emb_len)
        exact = compute_exact_accuracy(inner.lm_head, hidden, batch_gpu["labels"], puzzle_emb_len)
        total_exact += exact * B
        total_reward += reward.sum().item()
        total_count += B

    actor.train()
    if total_count == 0:
        return {"eval/exact_accuracy": 0.0, "eval/mean_reward": 0.0}
    return {
        "eval/exact_accuracy": total_exact / total_count,
        "eval/mean_reward": total_reward / total_count,
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_rl_checkpoint(
    save_dir: str, step: int,
    actor: nn.Module,
    critic_1: nn.Module, critic_2: nn.Module,
    actor_target_layers: nn.Module,
    critic_1_target: nn.Module, critic_2_target: nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
):
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "step": step,
        "actor_state_dict": actor.state_dict(),
        "critic_1_state_dict": critic_1.state_dict(),
        "critic_2_state_dict": critic_2.state_dict(),
        "actor_target_layers_state_dict": actor_target_layers.state_dict(),
        "critic_1_target_state_dict": critic_1_target.state_dict(),
        "critic_2_target_state_dict": critic_2_target.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
    }, os.path.join(save_dir, f"rl_step_{step}.pt"))
    print(f"Saved RL checkpoint at step {step} to {save_dir}")


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TD3 RL Training for URM")

    # Paths
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="checkpoints/rl_td3")

    # TD3 hyperparameters
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Discount factor. Low values recommended for long-horizon (96-step) tasks")
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy_noise", type=float, default=0.1)
    parser.add_argument("--noise_clip", type=float, default=0.3)
    parser.add_argument("--policy_delay", type=int, default=50,
                        help="Actor update frequency (1 actor update per N critic updates)")
    parser.add_argument("--exploration_noise", type=float, default=0.1)

    # Critic warmup: train critic only for this many steps before actor updates
    parser.add_argument("--critic_warmup_steps", type=int, default=1000,
                        help="Number of critic-only training steps before enabling actor updates")

    # TD3+BC: BC is primary term (weight=1), Q is weighted bonus
    # actor_loss = -lam * Q(s,a).mean() + BC_loss
    # where lam = bc_alpha / (bc_alpha + mean(|Q|))
    parser.add_argument("--bc_alpha", type=float, default=2.5,
                        help="TD3+BC alpha: controls Q-term weight. lam = alpha/(alpha+|Q|)")

    # Subsample transitions per collection step to preserve buffer diversity
    parser.add_argument("--buffer_collect_size", type=int, default=4096,
                        help="Max transitions to store per collection step (subsample from T*B)")

    # Training
    parser.add_argument("--lr_actor", type=float, default=1e-5)
    parser.add_argument("--lr_critic", type=float, default=3e-4)
    # Fix #4: larger buffer, smaller batch by default
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--td3_batch_size", type=int, default=256)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--num_unroll_steps", type=int, default=96,
                        help="Recurrent unroll steps per episode (1 step = 1 layer-stack pass)")
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=10)

    # Critic architecture
    parser.add_argument("--critic_layers", type=int, default=2)

    # Q-value clipping (Fix #5: prevent Q divergence)
    parser.add_argument("--q_clip", type=float, default=10.0,
                        help="Clip target Q-values to [-q_clip, 0] (rewards are non-positive)")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Enable TF32 for faster matmuls on Ampere+ GPUs
    torch.set_float32_matmul_precision("high")

    # -----------------------------------------------------------------------
    # 1. Load pretrained actor
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Loading pretrained URM actor...")
    print("=" * 60)

    actor, info = load_actor(args.checkpoint, args.data_path, device=device, batch_size=None)
    config = info["config"]
    metadata = info["metadata"]
    hidden_size = info["hidden_size"]
    num_heads = info["num_heads"]
    seq_len = info["seq_len"]
    puzzle_emb_len = info["puzzle_emb_len"]
    full_seq_len = seq_len + puzzle_emb_len

    print(f"  hidden_size={hidden_size}, num_heads={num_heads}")
    print(f"  seq_len={seq_len}, puzzle_emb_len={puzzle_emb_len}")
    print(f"  Actor params: {sum(p.numel() for p in actor.parameters()):,}")

    # Freeze embedding layers and lm_head
    for param in actor.inner.embed_tokens.parameters():
        param.requires_grad = False
    for param in actor.inner.lm_head.parameters():
        param.requires_grad = False
    actor.train()

    # -----------------------------------------------------------------------
    # 2. Create target actor layers + frozen pretrained reference for BC
    # -----------------------------------------------------------------------
    actor_target_layers = copy.deepcopy(actor.inner.layers)
    actor_target_layers.requires_grad_(False)
    actor_target_layers.eval()

    # Frozen copy of the pretrained actor for BC regularization
    pretrained_layers = copy.deepcopy(actor.inner.layers)
    pretrained_layers.requires_grad_(False)
    pretrained_layers.eval()

    # -----------------------------------------------------------------------
    # 3. Create twin critics + targets
    # -----------------------------------------------------------------------
    print("Creating twin critics...")
    critic_kwargs = dict(
        hidden_size=hidden_size, num_heads=num_heads,
        num_layers=args.critic_layers, max_seq_len=full_seq_len,
        rope_theta=info["rope_theta"],
    )
    critic_1 = TD3Critic(**critic_kwargs).to(device)
    critic_2 = TD3Critic(**critic_kwargs).to(device)
    critic_1_target = copy.deepcopy(critic_1)
    critic_1_target.requires_grad_(False)
    critic_2_target = copy.deepcopy(critic_2)
    critic_2_target.requires_grad_(False)

    print(f"  Critic params (each): {sum(p.numel() for p in critic_1.parameters()):,}")

    # -----------------------------------------------------------------------
    # 3b. torch.compile for actor and critic (kernel fusion)
    # -----------------------------------------------------------------------
    should_compile = not os.environ.get("DISABLE_COMPILE", "")
    if should_compile and device == "cuda":
        print("Compiling actor_step and critics with torch.compile...")
        compiled_actor_step = torch.compile(actor_step, dynamic=False)
        critic_1 = torch.compile(critic_1, dynamic=False)
        critic_2 = torch.compile(critic_2, dynamic=False)
    else:
        print("Skipping torch.compile (DISABLE_COMPILE set or CPU mode)")
        compiled_actor_step = actor_step

    # -----------------------------------------------------------------------
    # 4. Optimizers
    # -----------------------------------------------------------------------
    actor_params = [p for p in actor.inner.layers.parameters() if p.requires_grad]
    actor_optimizer = torch.optim.Adam(actor_params, lr=args.lr_actor)
    critic_optimizer = torch.optim.Adam(
        list(critic_1.parameters()) + list(critic_2.parameters()),
        lr=args.lr_critic,
    )

    # -----------------------------------------------------------------------
    # 5. Replay buffer & reward normalizer
    # -----------------------------------------------------------------------
    print(f"Creating replay buffer (capacity={args.buffer_size}, device={device})...")
    replay_buffer = ReplayBuffer(
        capacity=args.buffer_size, seq_len=full_seq_len,
        hidden_size=hidden_size, label_seq_len=seq_len,
        device=device,
    )
    # Fix #3: reward normalization
    reward_normalizer = RewardNormalizer(clip=5.0)

    # -----------------------------------------------------------------------
    # 6. Data loaders
    # -----------------------------------------------------------------------
    print("Creating data loaders...")
    ds_config = PuzzleDatasetConfig(
        seed=config.seed,
        dataset_path=config.data_path if args.data_path is None else args.data_path,
        rank=0, num_replicas=1,
        global_batch_size=config.global_batch_size,
        test_set_mode=False,
        epochs_per_iter=max(1, args.total_steps // 100),
    )
    train_dataset = PuzzleDataset(ds_config, split="train")
    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=1, prefetch_factor=4, pin_memory=True,
    )
    train_iter = iter(train_loader)

    eval_ds_config = PuzzleDatasetConfig(
        seed=config.seed,
        dataset_path=config.data_path if args.data_path is None else args.data_path,
        rank=0, num_replicas=1,
        global_batch_size=config.global_batch_size,
        test_set_mode=True, epochs_per_iter=1,
    )
    eval_dataset = PuzzleDataset(eval_ds_config, split="test")
    eval_loader = DataLoader(
        eval_dataset, batch_size=None,
        num_workers=1, prefetch_factor=2, pin_memory=True,
    )

    # -----------------------------------------------------------------------
    # 7. Training loop
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Starting TD3+BC RL Training")
    print(f"  total_steps={args.total_steps}")
    print(f"  num_unroll_steps={args.num_unroll_steps}")
    print(f"  buffer_size={args.buffer_size}, td3_batch_size={args.td3_batch_size}")
    print(f"  buffer_collect_size={args.buffer_collect_size}")
    print(f"  warmup_steps={args.warmup_steps} (buffer fill)")
    print(f"  critic_warmup_steps={args.critic_warmup_steps} (critic-only phase)")
    print(f"  gamma={args.gamma}, tau={args.tau}, q_clip={args.q_clip}")
    print(f"  policy_delay={args.policy_delay}")
    print(f"  bc_alpha={args.bc_alpha} (TD3+BC: lam=alpha/(alpha+|Q|))")
    print(f"  exploration_noise={args.exploration_noise}")
    print(f"  policy_noise={args.policy_noise}")
    print("=" * 60)

    inner = actor.inner
    layers = inner.layers
    lm_head = inner.lm_head
    cos_sin = inner.rotary_emb()
    layers_target = actor_target_layers

    # Use compiled actor_step for training (uncompiled for target networks)
    train_actor_step = compiled_actor_step

    critic_losses = []
    actor_losses = []
    rewards_collected = []

    T = args.num_unroll_steps
    last_actor_loss = None
    last_bc_loss = None
    last_lam = 0.0
    critic_update_count = 0  # Track critic updates for warmup phase

    progress = tqdm(range(1, args.total_steps + 1), desc="RL Training")
    for step in progress:
        # === COLLECT TRANSITIONS (Fix #1: batch on GPU, transfer once) ===
        try:
            set_name, batch, global_batch_size = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            set_name, batch, global_batch_size = next(train_iter)

        batch_gpu = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        B = batch_gpu["inputs"].shape[0]

        # Compute input embeddings (eval mode to bypass sparse emb buffer)
        was_training = inner.training
        inner.eval()
        with torch.no_grad():
            input_emb = inner._input_embeddings(
                batch_gpu["inputs"], batch_gpu["puzzle_identifiers"]
            )
        if was_training:
            inner.train()

        hidden = inner.init_hidden.expand(B, full_seq_len, -1).clone()

        # Pre-allocate GPU buffers for all transitions
        all_states = torch.empty(T, B, full_seq_len, hidden_size,
                                 device=device, dtype=torch.bfloat16)
        all_actions = torch.empty_like(all_states)
        all_rewards = torch.empty(T, B, device=device, dtype=torch.float32)
        all_dones = torch.zeros(T, B, device=device, dtype=torch.bool)

        # Pure GPU unroll loop
        with torch.no_grad():
            for t in range(T):
                state_t = hidden

                # Actor step (compiled)
                next_state, action = train_actor_step(layers, cos_sin, state_t, input_emb)

                # Add exploration noise
                noise = torch.randn_like(action) * args.exploration_noise
                noisy_action = action + noise
                noisy_next_state = state_t + input_emb + noisy_action

                # Reward (stays on GPU)
                reward = compute_reward(
                    lm_head, noisy_next_state, batch_gpu["labels"], puzzle_emb_len
                )

                # Store in GPU buffers (no transfers)
                all_states[t] = state_t.to(torch.bfloat16)
                all_actions[t] = noisy_action.to(torch.bfloat16)
                all_rewards[t] = reward
                if t == T - 1:
                    all_dones[t] = True

                hidden = noisy_next_state

        # Reward stats update (vectorized, on GPU)
        reward_normalizer.update(all_rewards)
        mean_reward_this_step = all_rewards.mean().item()
        rewards_collected.append(mean_reward_this_step)

        # Expand input_emb and labels to [T, B, ...]
        all_input_embs = input_emb.unsqueeze(0).expand(T, -1, -1, -1).to(torch.bfloat16)
        all_labels = batch_gpu["labels"].unsqueeze(0).expand(T, -1, -1)

        # Subsample transitions to preserve buffer diversity
        N_total = T * B
        N_keep = min(N_total, args.buffer_collect_size)
        if N_keep < N_total:
            perm = torch.randperm(N_total, device=device)[:N_keep]
            t_idx = perm // B
            b_idx = perm % B
            sub_states = all_states[t_idx, b_idx].unsqueeze(0)
            sub_actions = all_actions[t_idx, b_idx].unsqueeze(0)
            sub_rewards = all_rewards[t_idx, b_idx].unsqueeze(0)
            sub_dones = all_dones[t_idx, b_idx].unsqueeze(0)
            sub_embs = all_input_embs[t_idx, b_idx].unsqueeze(0)
            sub_labels = all_labels[t_idx, b_idx].unsqueeze(0)
        else:
            sub_states = all_states
            sub_actions = all_actions
            sub_rewards = all_rewards
            sub_dones = all_dones
            sub_embs = all_input_embs
            sub_labels = all_labels

        # Direct GPU write — no CPU transfer needed
        replay_buffer.add_batch(
            sub_states, sub_actions, sub_rewards,
            sub_dones, sub_embs, sub_labels,
        )

        # === TD3+BC UPDATES ===
        if replay_buffer.size < args.warmup_steps:
            progress.set_postfix(buffer=replay_buffer.size)
            continue

        # Sample from GPU buffer — no CPU→GPU transfer
        (s, a, r, d, stored_input_emb, stored_labels
         ) = replay_buffer.sample(args.td3_batch_size)

        # Reconstruct next_state on GPU: s' = s + u + a
        s_next = s + stored_input_emb + a

        # Normalize rewards for Q-learning
        r_norm = reward_normalizer.normalize(r)

        # --- Critic update (always runs) ---
        with torch.no_grad():
            # Target action
            _, target_action = actor_step(layers_target, cos_sin, s_next, stored_input_emb)
            # Target policy smoothing
            noise = (
                torch.randn_like(target_action) * args.policy_noise
            ).clamp(-args.noise_clip, args.noise_clip)
            smoothed_target_action = target_action + noise

            # Target Q values
            target_q1 = critic_1_target(s_next, smoothed_target_action)
            target_q2 = critic_2_target(s_next, smoothed_target_action)
            target_q = r_norm + args.gamma * (~d).float() * torch.min(target_q1, target_q2)
            target_q = target_q.clamp(-args.q_clip, args.q_clip)

        current_q1 = critic_1(s, a)
        current_q2 = critic_2(s, a)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(critic_1.parameters()) + list(critic_2.parameters()), max_norm=1.0
        )
        critic_optimizer.step()
        critic_update_count += 1

        # Soft update critic targets (every step, independent of actor)
        soft_update(critic_1_target, critic_1, args.tau)
        soft_update(critic_2_target, critic_2, args.tau)

        # --- Delayed actor update with critic warmup gate + TD3+BC ---
        actor_update_allowed = critic_update_count > args.critic_warmup_steps
        if actor_update_allowed and step % args.policy_delay == 0:
            _, actor_action = train_actor_step(layers, cos_sin, s, stored_input_emb)

            # Q-value term
            q_val = critic_1(s, actor_action).mean()

            # BC term: penalize deviation from pretrained policy
            with torch.no_grad():
                _, pretrained_action = actor_step(
                    pretrained_layers, cos_sin, s, stored_input_emb
                )
            bc_loss = F.mse_loss(actor_action, pretrained_action)

            # TD3+BC (Fujimoto & Gu, 2021):
            #   actor_loss = -lam * Q(s, pi(s)).mean() + BC_loss
            #   lam = alpha / (alpha + mean(|Q|))
            # BC is the PRIMARY term (weight=1), Q is a small weighted bonus.
            # This prevents the actor from chasing inaccurate Q estimates.
            q_abs_mean = current_q1.detach().abs().mean().clamp(min=1e-6)
            lam = args.bc_alpha / (args.bc_alpha + q_abs_mean)
            actor_loss = -lam * q_val + bc_loss

            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_params, max_norm=1.0)
            actor_optimizer.step()
            last_actor_loss = actor_loss.item()
            last_bc_loss = bc_loss.item()
            last_lam = lam.item() if isinstance(lam, torch.Tensor) else lam

            # Soft update actor target
            soft_update(actor_target_layers, actor.inner.layers, args.tau)

        # === LOGGING ===
        if step % args.log_interval == 0:
            avg_r = (sum(rewards_collected[-args.log_interval:])
                     / max(len(rewards_collected[-args.log_interval:]), 1))
            a_loss_val = last_actor_loss if last_actor_loss is not None else 0.0
            bc_loss_val = last_bc_loss if last_bc_loss is not None else 0.0
            phase = "critic-only" if not actor_update_allowed else "actor+critic"
            progress.set_postfix(
                phase=phase,
                c_loss=f"{critic_loss.item():.4f}",
                a_loss=f"{a_loss_val:.4f}",
                bc=f"{bc_loss_val:.6f}",
                lam=f"{last_lam:.4f}",
                reward=f"{avg_r:.4f}",
                buf=replay_buffer.size,
            )

        # === EVALUATION ===
        if step % args.eval_interval == 0:
            print(f"\n[Step {step}] Running evaluation...")
            eval_metrics = evaluate_actor(
                actor, eval_loader, args.num_unroll_steps, device=device
            )
            print(
                f"  exact_accuracy={eval_metrics['eval/exact_accuracy']:.4f}, "
                f"  mean_reward={eval_metrics['eval/mean_reward']:.4f}"
            )

        # === CHECKPOINTING ===
        if step % args.save_interval == 0:
            save_rl_checkpoint(
                args.save_dir, step, actor, critic_1, critic_2,
                actor_target_layers, critic_1_target, critic_2_target,
                actor_optimizer, critic_optimizer,
            )

    # Final save & eval
    save_rl_checkpoint(
        args.save_dir, args.total_steps, actor, critic_1, critic_2,
        actor_target_layers, critic_1_target, critic_2_target,
        actor_optimizer, critic_optimizer,
    )

    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    eval_metrics = evaluate_actor(
        actor, eval_loader, args.num_unroll_steps, device=device
    )
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
