from __future__ import annotations

import gzip
import io
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ContinuousNCAConfig:
    grid_size: Optional[int] = None
    grid_height: int = 12
    grid_width: int = 12

    state_channels: int = 8
    visible_channel: int = 0
    clip_value: float = 1.0
    init_noise_std: float = 0.15
    init_blob_scale: float = 1.0
    init_blob_sigma: float = 1.5
    update_scale: float = 0.25
    fire_rate: float = 0.7

    conv_channels: int = 16
    hidden_dim: int = 32

    patch_size: int = 2
    num_bins: int = 16
    visible_min: float = -1.0
    visible_max: float = 1.0
    seq_len: int = 1024

    rollout_steps: int = 32
    time_subsample: int = 1
    start_step: int = 0

    gzip_threshold_low: float = 0.35
    gzip_threshold_high: Optional[float] = 0.95

    train_size: int = 256
    val_size: int = 64

    batch_candidate_size: int = 64
    max_sampling_rounds: int = 200

    out_dir: str = "./continuous_nca_dataset"
    save_dtype: str = "int32"

    num_vis_examples: int = 6

    def __post_init__(self) -> None:
        if self.grid_size is not None:
            self.grid_height = self.grid_size
            self.grid_width = self.grid_size
        if not 0 <= self.visible_channel < self.state_channels:
            raise ValueError("visible_channel must be in [0, state_channels).")
        if self.visible_max <= self.visible_min:
            raise ValueError("visible_max must be larger than visible_min.")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if self.num_bins < 2:
            raise ValueError("num_bins must be at least 2.")

    @property
    def patch_vocab_size(self) -> int:
        return self.num_bins ** (self.patch_size ** 2)

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
    def grid_shape(self) -> Tuple[int, int]:
        return self.grid_height, self.grid_width

    @property
    def patches_per_height(self) -> int:
        assert self.grid_height % self.patch_size == 0
        return self.grid_height // self.patch_size

    @property
    def patches_per_width(self) -> int:
        assert self.grid_width % self.patch_size == 0
        return self.grid_width // self.patch_size

    @property
    def tokens_per_frame(self) -> int:
        return self.patches_per_height * self.patches_per_width + 2


class RandomContinuousNCA(nn.Module):
    """
    A frozen random continuous-state NCA rule.

    State shape: [C, H, W], where the visible frame is one selected channel.
    Update:
      circular 3x3 conv -> 1x1 hidden -> ReLU -> 1x1 delta
      optional stochastic fire mask
      residual update with tanh bounding
    """

    def __init__(self, cfg: ContinuousNCAConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.state_channels
        self.conv3 = nn.Conv2d(c, cfg.conv_channels, kernel_size=3, padding=0, bias=True)
        self.fc1 = nn.Conv2d(cfg.conv_channels, cfg.hidden_dim, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(cfg.hidden_dim, c, kernel_size=1, bias=True)
        self.reset_random_parameters()
        for param in self.parameters():
            param.requires_grad_(False)

    def reset_random_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.4)
                if module.bias is not None:
                    nn.init.normal_(module.bias, mean=0.0, std=0.2)

    def delta_from_state(self, state: torch.Tensor) -> torch.Tensor:
        x = state.unsqueeze(0)
        x = F.pad(x, (1, 1, 1, 1), mode="circular")
        x = torch.tanh(self.conv3(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(0)

    @torch.no_grad()
    def step(self, state: torch.Tensor) -> torch.Tensor:
        delta = self.delta_from_state(state)
        if self.cfg.fire_rate < 1.0:
            mask = (
                torch.rand((1, state.shape[1], state.shape[2]), device=state.device)
                <= self.cfg.fire_rate
            ).float()
            delta = delta * mask
        updated = state + self.cfg.update_scale * delta
        scale = max(self.cfg.clip_value, 1e-6)
        return scale * torch.tanh(updated / scale)


def sample_initial_state(cfg: ContinuousNCAConfig, device: torch.device) -> torch.Tensor:
    state = cfg.init_noise_std * torch.randn(
        cfg.state_channels,
        cfg.grid_height,
        cfg.grid_width,
        device=device,
        dtype=torch.float32,
    )

    yy, xx = torch.meshgrid(
        torch.arange(cfg.grid_height, device=device, dtype=torch.float32),
        torch.arange(cfg.grid_width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cy = (cfg.grid_height - 1) / 2.0
    cx = (cfg.grid_width - 1) / 2.0
    sigma = max(cfg.init_blob_sigma, 1e-3)
    blob = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
    state[cfg.visible_channel] += cfg.init_blob_scale * blob

    if cfg.state_channels > 1:
        aux_channel = (cfg.visible_channel + 1) % cfg.state_channels
        offset_blob = torch.roll(blob, shifts=max(1, cfg.grid_width // 4), dims=1)
        state[aux_channel] -= 0.5 * cfg.init_blob_scale * offset_blob

    scale = max(cfg.clip_value, 1e-6)
    return scale * torch.tanh(state / scale)


@torch.no_grad()
def rollout_trajectory(
    rule: RandomContinuousNCA,
    cfg: ContinuousNCAConfig,
    device: torch.device,
) -> np.ndarray:
    """
    Returns a visible-channel trajectory with shape [T, H, W].
    """

    state = sample_initial_state(cfg, device)
    frames: List[np.ndarray] = []
    total_steps = cfg.start_step + cfg.rollout_steps

    for t in range(total_steps):
        if t >= cfg.start_step and ((t - cfg.start_step) % cfg.time_subsample == 0):
            frame = state[cfg.visible_channel].detach().cpu().numpy().astype(np.float32)
            frames.append(frame)
        state = rule.step(state)

    return np.stack(frames, axis=0)


class ContinuousNCATokenizer:
    def __init__(self, cfg: ContinuousNCAConfig):
        self.cfg = cfg
        self.patch = cfg.patch_size
        self.num_bins = cfg.num_bins
        self.start_tk = cfg.start_token
        self.end_tk = cfg.end_token
        self.base = (self.num_bins ** np.arange(self.patch * self.patch)).astype(np.int64)
        self.value_span = cfg.visible_max - cfg.visible_min

    def quantize_frame(self, frame: np.ndarray) -> np.ndarray:
        clipped = np.clip(frame, self.cfg.visible_min, self.cfg.visible_max)
        scaled = (clipped - self.cfg.visible_min) / self.value_span
        bins = np.floor(scaled * self.num_bins).astype(np.int64)
        return np.clip(bins, 0, self.num_bins - 1)

    def dequantize_frame(self, qframe: np.ndarray) -> np.ndarray:
        centers = (qframe.astype(np.float32) + 0.5) / self.num_bins
        return (self.cfg.visible_min + centers * self.value_span).astype(np.float32)

    def encode_quantized_frame(self, qframe: np.ndarray) -> np.ndarray:
        h, w = qframe.shape
        p = self.patch
        assert h % p == 0 and w % p == 0
        nh, nw = h // p, w // p
        patches = qframe.reshape(nh, p, nw, p).transpose(0, 2, 1, 3).reshape(nh * nw, p * p)
        patch_tokens = (patches * self.base[None, :]).sum(axis=1, dtype=np.int64)
        return np.concatenate(
            [
                np.array([self.start_tk], dtype=np.int64),
                patch_tokens,
                np.array([self.end_tk], dtype=np.int64),
            ]
        )

    def encode_frame(self, frame: np.ndarray) -> np.ndarray:
        return self.encode_quantized_frame(self.quantize_frame(frame))

    def encode_trajectory(self, traj: np.ndarray) -> np.ndarray:
        tokens = [self.encode_frame(frame) for frame in traj]
        return np.concatenate(tokens, axis=0)

    def decode_quantized_frame(self, frame_tokens: np.ndarray) -> np.ndarray:
        p = self.patch
        nh = self.cfg.patches_per_height
        nw = self.cfg.patches_per_width
        vals = frame_tokens.astype(np.int64)[:, None]
        digits = (vals // self.base[None, :]) % self.num_bins
        frame = digits.reshape(nh, nw, p, p).transpose(0, 2, 1, 3).reshape(nh * p, nw * p)
        return frame.astype(np.int64)

    def decode_frame(self, frame_tokens: np.ndarray) -> np.ndarray:
        return self.dequantize_frame(self.decode_quantized_frame(frame_tokens))


def gzip_complexity_ratio_from_tokens(tokens: np.ndarray) -> float:
    byte_data = np.asarray(tokens, dtype=np.int32).tobytes()
    raw_size = len(byte_data)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9) as handle:
        handle.write(byte_data)
    compressed_size = len(buf.getvalue())
    return compressed_size / raw_size


def score_trajectory_gzip(traj: np.ndarray, tokenizer: ContinuousNCATokenizer) -> float:
    frame_tokens = [tokenizer.encode_frame(frame)[1:-1] for frame in traj]
    flat = np.concatenate(frame_tokens, axis=0)
    return gzip_complexity_ratio_from_tokens(flat)


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
    cfg: ContinuousNCAConfig,
    target_size: int,
    seed: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    seed_everything(seed)
    tokenizer = ContinuousNCATokenizer(cfg)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    gzip_scores: List[float] = []
    raw_lengths: List[int] = []
    kept_trajs: List[np.ndarray] = []
    kept_quantized: List[np.ndarray] = []

    rounds = 0
    while len(xs) < target_size and rounds < cfg.max_sampling_rounds:
        rounds += 1
        for _ in range(cfg.batch_candidate_size):
            rule = RandomContinuousNCA(cfg).to(device)
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
            kept_trajs.append(traj.astype(np.float32))
            kept_quantized.append(
                np.stack([tokenizer.quantize_frame(frame) for frame in traj], axis=0).astype(np.int16)
            )

            if len(xs) >= target_size:
                break

    if len(xs) < target_size:
        raise RuntimeError(
            f"Could not collect enough trajectories. Got {len(xs)} / {target_size}. "
            f"Try lowering the gzip threshold or increasing max_sampling_rounds."
        )

    return {
        "input_ids": np.stack(xs, axis=0),
        "labels": np.stack(ys, axis=0),
        "gzip_ratio": np.asarray(gzip_scores, dtype=np.float32),
        "raw_token_length": np.asarray(raw_lengths, dtype=np.int32),
        "trajectories": np.asarray(kept_trajs, dtype=np.float32),
        "quantized_trajectories": np.asarray(kept_quantized, dtype=np.int16),
    }


def save_dataset_bundle(
    cfg: ContinuousNCAConfig,
    train: Dict[str, np.ndarray],
    val: Dict[str, np.ndarray],
) -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "train_input_ids.npy", train["input_ids"].astype(cfg.save_dtype))
    np.save(out_dir / "train_labels.npy", train["labels"].astype(cfg.save_dtype))
    np.save(out_dir / "train_gzip_ratio.npy", train["gzip_ratio"])
    np.save(out_dir / "train_raw_token_length.npy", train["raw_token_length"])
    np.save(out_dir / "train_trajectories.npy", train["trajectories"].astype(np.float32))
    np.save(
        out_dir / "train_quantized_trajectories.npy",
        train["quantized_trajectories"].astype(np.int16),
    )

    np.save(out_dir / "val_input_ids.npy", val["input_ids"].astype(cfg.save_dtype))
    np.save(out_dir / "val_labels.npy", val["labels"].astype(cfg.save_dtype))
    np.save(out_dir / "val_gzip_ratio.npy", val["gzip_ratio"])
    np.save(out_dir / "val_raw_token_length.npy", val["raw_token_length"])
    np.save(out_dir / "val_trajectories.npy", val["trajectories"].astype(np.float32))
    np.save(
        out_dir / "val_quantized_trajectories.npy",
        val["quantized_trajectories"].astype(np.int16),
    )

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                **asdict(cfg),
                "patch_vocab_size": cfg.patch_vocab_size,
                "total_vocab_size": cfg.total_vocab_size,
                "start_token": cfg.start_token,
                "end_token": cfg.end_token,
                "tokens_per_frame": cfg.tokens_per_frame,
                "notes": {
                    "trajectories": "Visible-channel float trajectories before quantization.",
                    "quantized_trajectories": "Visible trajectories mapped to discrete bins before patch tokenization.",
                    "input_ids": "LM input sequence, shifted right from full token sequence.",
                    "labels": "LM targets, shifted left from full token sequence.",
                    "padding": -100,
                    "gzip_ratio": "compressed_bytes / raw_bytes measured on quantized patch tokens excluding <grid> and </grid>.",
                },
            },
            handle,
            indent=2,
        )

    print(f"Saved dataset to: {out_dir.resolve()}")


def build_continuous_nca_dataset(
    cfg: ContinuousNCAConfig,
    seed: int = 0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    device = get_device()
    print("Device:", device)
    print("Generating train split...")
    train = sample_filtered_examples(cfg, target_size=cfg.train_size, seed=seed, device=device)
    print("Generating val split...")
    val = sample_filtered_examples(cfg, target_size=cfg.val_size, seed=seed + 1, device=device)
    save_dataset_bundle(cfg, train, val)
    return train, val
