# Colab-friendly NCA pre-pre-training data generator
# Based on:
# - "Training Language Models via Neural Cellular Automata" (arXiv:2603.10055)
# - the user-provided NCA generation/tokenization codebase
#
# Usage in Colab:
#   1) Paste this whole file into a cell, or save it as nca_prepretraining_data_generator_colab.py
#   2) Run the bottom example block.
#
# What this script does:
#   - samples random discrete NCA rules
#   - rolls out trajectories on a 12x12 grid
#   - filters trajectories by gzip complexity band
#   - tokenizes each frame with 2x2 patches + <grid>, </grid>
#   - saves train/val arrays and metadata
#   - visualizes trajectories, complexity histogram, token frequencies
#
# The implementation is intentionally practical for Colab:
#   - pure PyTorch + NumPy + matplotlib
#   - no JAX/Flax requirement
#   - faithful to the paper's data generation recipe

from __future__ import annotations

import gzip
import io
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colors as mcolors
from matplotlib import animation
from torch import nn


# ============================================================
# 0. Colab helper
# ============================================================

def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. Config
# ============================================================

@dataclass
class NCAConfig:
    # Paper-aligned defaults
    grid_size: int = 12
    num_colors: int = 10
    temperature: float = 1e-3
    identity_bias: float = 0.0

    # Update rule architecture: 3x3 conv -> hidden 16 -> ReLU -> logits(10)
    conv_channels: int = 4
    hidden_dim: int = 16

    # Tokenization: paper uses non-overlapping 2x2 patches
    patch_size: int = 2
    seq_len: int = 1024

    # Sampling
    rollout_steps: int = 32          # raw rollout length before truncation into tokens
    time_subsample: int = 1          # keep every dT-th frame
    start_step: int = 0              # burn-in

    # Complexity filtering; paper default keeps r > 50%
    gzip_threshold_low: float = 0.50
    gzip_threshold_high: Optional[float] = None

    # Dataset size
    train_size: int = 256
    val_size: int = 64

    # Generation loop
    batch_candidate_size: int = 64   # how many candidate trajectories to sample at once
    max_sampling_rounds: int = 200

    # Saving
    out_dir: str = "./nca_dataset"
    save_dtype: str = "int32"

    # Visualization
    num_vis_examples: int = 6

    @property
    def patch_vocab_size(self) -> int:
        return self.num_colors ** (self.patch_size ** 2)

    @property
    def start_token(self) -> int:
        return self.patch_vocab_size

    @property
    def end_token(self) -> int:
        return self.patch_vocab_size + 1

    @property
    def total_vocab_size(self) -> int:
        return self.patch_vocab_size + 2

    @property
    def patches_per_side(self) -> int:
        assert self.grid_size % self.patch_size == 0
        return self.grid_size // self.patch_size

    @property
    def tokens_per_frame(self) -> int:
        return self.patches_per_side ** 2 + 2


# ============================================================
# 2. Random discrete NCA rule
# ============================================================

class RandomDiscreteNCA(nn.Module):
    """
    A single random NCA rule.

    Input state: [H, W] integers in {0, ..., num_colors-1}
    Update:
      one_hot -> circular 3x3 conv (4 ch) -> 1x1 conv (16) -> ReLU -> 1x1 conv (num_colors)
      sample next state from softmax(logits / tau)
    """
    def __init__(self, cfg: NCAConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.num_colors
        self.conv3 = nn.Conv2d(c, cfg.conv_channels, kernel_size=3, padding=0, bias=True)
        self.fc1 = nn.Conv2d(cfg.conv_channels, cfg.hidden_dim, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(cfg.hidden_dim, c, kernel_size=1, bias=True)
        self.reset_random_parameters()
        for p in self.parameters():
            p.requires_grad_(False)

    def reset_random_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=1.0)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.0, std=1.0)

    def logits_from_state(self, state: torch.Tensor) -> torch.Tensor:
        # state: [H, W] int64
        x = F.one_hot(state.long(), num_classes=self.cfg.num_colors).float()   # [H, W, C]
        x = x.permute(2, 0, 1).unsqueeze(0)                                     # [1, C, H, W]
        x = F.pad(x, (1, 1, 1, 1), mode="circular")
        x = self.conv3(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)                                                         # [1, C, H, W]
        return x.squeeze(0).permute(1, 2, 0).contiguous()                       # [H, W, C]

    @torch.no_grad()
    def step(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.logits_from_state(state)
        oh = F.one_hot(state.long(), num_classes=self.cfg.num_colors).float()
        logits = logits + self.cfg.identity_bias * oh
        logits = logits / max(self.cfg.temperature, 1e-8)
        probs = torch.softmax(logits, dim=-1)
        flat = probs.view(-1, self.cfg.num_colors)
        nxt = torch.multinomial(flat, num_samples=1).view(self.cfg.grid_size, self.cfg.grid_size)
        return nxt


# ============================================================
# 3. Trajectory generation
# ============================================================

def sample_initial_state(cfg: NCAConfig, device: torch.device) -> torch.Tensor:
    return torch.randint(
        low=0,
        high=cfg.num_colors,
        size=(cfg.grid_size, cfg.grid_size),
        device=device,
        dtype=torch.long,
    )
    # out = torch.zeros(
    #     size=(cfg.grid_size, cfg.grid_size),
    #     device=device,
    #     dtype=torch.long,
    # )
    # out[cfg.grid_size // 2, cfg.grid_size // 2] = 1
    return out


@torch.no_grad()
def rollout_trajectory(rule: RandomDiscreteNCA, cfg: NCAConfig, device: torch.device) -> np.ndarray:
    """
    Returns trajectory with shape [T, H, W].
    """
    state = sample_initial_state(cfg, device)
    frames: List[np.ndarray] = []
    total_steps = cfg.start_step + cfg.rollout_steps

    for t in range(total_steps):
        if t >= cfg.start_step and ((t - cfg.start_step) % cfg.time_subsample == 0):
            frames.append(state.detach().cpu().numpy().astype(np.int16))
        state = rule.step(state)

    return np.stack(frames, axis=0)


# ============================================================
# 4. Tokenization (paper-style 2x2 patches)
# ============================================================

class NCATokenizer:
    def __init__(self, cfg: NCAConfig):
        self.cfg = cfg
        self.patch = cfg.patch_size
        self.num_colors = cfg.num_colors
        self.start_tk = cfg.start_token
        self.end_tk = cfg.end_token
        self.base = (self.num_colors ** np.arange(self.patch * self.patch)).astype(np.int64)

    def encode_frame(self, frame: np.ndarray) -> np.ndarray:
        """frame: [H, W] -> tokens: [num_patches + 2]"""
        H, W = frame.shape
        p = self.patch
        assert H % p == 0 and W % p == 0

        nh, nw = H // p, W // p
        patches = frame.reshape(nh, p, nw, p).transpose(0, 2, 1, 3).reshape(nh * nw, p * p)
        patch_tokens = (patches * self.base[None, :]).sum(axis=1, dtype=np.int64)
        return np.concatenate([
            np.array([self.start_tk], dtype=np.int64),
            patch_tokens,
            np.array([self.end_tk], dtype=np.int64),
        ])

    def encode_trajectory(self, traj: np.ndarray) -> np.ndarray:
        toks = [self.encode_frame(frame) for frame in traj]
        return np.concatenate(toks, axis=0)

    def decode_frame(self, frame_tokens: np.ndarray) -> np.ndarray:
        # frame_tokens excludes start/end
        p = self.patch
        nh = nw = self.cfg.patches_per_side
        vals = frame_tokens.astype(np.int64)[:, None]
        digits = (vals // self.base[None, :]) % self.num_colors
        frame = digits.reshape(nh, nw, p, p).transpose(0, 2, 1, 3).reshape(nh * p, nw * p)
        return frame.astype(np.int64)


# ============================================================
# 5. Complexity scoring (gzip)
# ============================================================

def gzip_complexity_ratio_from_tokens(tokens: np.ndarray) -> float:
    """
    Returns compressed_bytes / raw_bytes.
    Matches the spirit of the paper and the provided codebase.
    """
    byte_data = np.asarray(tokens, dtype=np.int32).tobytes()
    raw_size = len(byte_data)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9) as f:
        f.write(byte_data)
    compressed_size = len(buf.getvalue())
    return compressed_size / raw_size


def score_trajectory_gzip(traj: np.ndarray, tokenizer: NCATokenizer) -> float:
    # Follow the provided codebase: remove <grid> and </grid> before gzip scoring.
    frame_tokens = [tokenizer.encode_frame(frame)[1:-1] for frame in traj]
    flat = np.concatenate(frame_tokens, axis=0)
    return gzip_complexity_ratio_from_tokens(flat)


# ============================================================
# 6. Dataset generation with gzip-band filtering
# ============================================================

def pad_or_trim(tokens: np.ndarray, seq_len: int, pad_value: int = -100) -> np.ndarray:
    if len(tokens) >= seq_len:
        return tokens[:seq_len].astype(np.int64)
    out = np.full((seq_len,), pad_value, dtype=np.int64)
    out[: len(tokens)] = tokens
    return out


def make_lm_pair(tokens: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    x = pad_or_trim(tokens[:-1], seq_len=seq_len, pad_value=-100)
    y = pad_or_trim(tokens[1:], seq_len=seq_len, pad_value=-100)
    return x, y


def sample_filtered_examples(
    cfg: NCAConfig,
    target_size: int,
    seed: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    seed_everything(seed)
    tokenizer = NCATokenizer(cfg)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    gzip_scores: List[float] = []
    raw_lengths: List[int] = []
    kept_trajs: List[np.ndarray] = []

    rounds = 0
    while len(xs) < target_size and rounds < cfg.max_sampling_rounds:
        rounds += 1
        for _ in range(cfg.batch_candidate_size):
            rule = RandomDiscreteNCA(cfg).to(device)
            traj = rollout_trajectory(rule, cfg, device=device)
            score = score_trajectory_gzip(traj, tokenizer)

            if score < cfg.gzip_threshold_low:
                continue
            if cfg.gzip_threshold_high is not None and score >= cfg.gzip_threshold_high:
                continue

            tokens = tokenizer.encode_trajectory(traj)
            x, y = make_lm_pair(tokens, seq_len=cfg.seq_len)
            xs.append(x)
            ys.append(y)
            gzip_scores.append(score)
            raw_lengths.append(len(tokens))
            kept_trajs.append(traj)

            if len(xs) >= target_size:
                break

    if len(xs) < target_size:
        raise RuntimeError(
            f"Could not collect enough trajectories. Got {len(xs)} / {target_size}. "
            f"Try increasing rollout_steps, batch_candidate_size, or max_sampling_rounds, "
            f"or lower the gzip threshold."
        )

    return {
        "input_ids": np.stack(xs, axis=0),
        "labels": np.stack(ys, axis=0),
        "gzip_ratio": np.asarray(gzip_scores, dtype=np.float32),
        "raw_token_length": np.asarray(raw_lengths, dtype=np.int32),
        "trajectories": np.asarray(kept_trajs, dtype=np.int16),
    }


# ============================================================
# 7. Saving
# ============================================================

def save_dataset_bundle(cfg: NCAConfig, train: Dict[str, np.ndarray], val: Dict[str, np.ndarray]) -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "train_input_ids.npy", train["input_ids"].astype(cfg.save_dtype))
    np.save(out_dir / "train_labels.npy", train["labels"].astype(cfg.save_dtype))
    np.save(out_dir / "train_gzip_ratio.npy", train["gzip_ratio"])
    np.save(out_dir / "train_raw_token_length.npy", train["raw_token_length"])
    np.save(out_dir / "train_trajectories.npy", train["trajectories"])

    np.save(out_dir / "val_input_ids.npy", val["input_ids"].astype(cfg.save_dtype))
    np.save(out_dir / "val_labels.npy", val["labels"].astype(cfg.save_dtype))
    np.save(out_dir / "val_gzip_ratio.npy", val["gzip_ratio"])
    np.save(out_dir / "val_raw_token_length.npy", val["raw_token_length"])
    np.save(out_dir / "val_trajectories.npy", val["trajectories"])

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **asdict(cfg),
                "patch_vocab_size": cfg.patch_vocab_size,
                "total_vocab_size": cfg.total_vocab_size,
                "start_token": cfg.start_token,
                "end_token": cfg.end_token,
                "tokens_per_frame": cfg.tokens_per_frame,
                "notes": {
                    "input_ids": "LM input sequence, shifted right from full token sequence",
                    "labels": "LM targets, shifted left from full token sequence",
                    "padding": -100,
                    "gzip_ratio": "compressed_bytes / raw_bytes measured on patch tokens excluding <grid> and </grid>",
                },
            },
            f,
            indent=2,
        )

    print(f"Saved dataset to: {out_dir.resolve()}")


# ============================================================
# 9. End-to-end runner
# ============================================================

def build_nca_dataset(cfg: NCAConfig, seed: int = 0) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    device = get_device()
    print("Device:", device)
    print("Generating train split...")
    train = sample_filtered_examples(cfg, target_size=cfg.train_size, seed=seed, device=device)
    print("Generating val split...")
    val = sample_filtered_examples(cfg, target_size=cfg.val_size, seed=seed + 1, device=device)
    save_dataset_bundle(cfg, train, val)
    return train, val