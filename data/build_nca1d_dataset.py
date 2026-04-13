from __future__ import annotations

import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from argdantic import ArgParser
from pydantic import BaseModel, model_validator
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.common import PuzzleDatasetMetadata
from NCA_datas.NCA_data import (
    NCAConfig,
    NCATokenizer,
    get_device,
    gzip_complexity_ratio_from_tokens,
    seed_everything,
)


cli = ArgParser()


class NCA1DDataConfig(BaseModel):
    output_dir: str = "data/nca1d-data"

    train_size: int = 1000000
    test_size: int = 1000
    seed: int = 0

    state_height: int = 9
    state_height_min: Optional[int] = 9
    state_height_max: Optional[int] = 60
    state_width: int = 1

    num_colors: int = 10
    temperature: float = 1e-5
    identity_bias: float = 0.0
    conv_channels: int = 4
    hidden_dim: int = 16

    # Use patch_size=1 so 1-column NCA states remain tokenizable for gzip filtering.
    patch_size: int = 1

    rollout_steps: int = 9
    rollout_steps_min: Optional[int] = 9
    rollout_steps_max: Optional[int] = 30
    counts: int = 1
    counts_min: Optional[int] = 3
    counts_max: Optional[int] = 12
    time_subsample: int = 1
    start_step: int = 0
    # If set, the first temporal window starts from this raw NCA step instead of start_step.
    time_start: Optional[int] = 50
    # Gap between consecutive temporal windows sampled along the counts axis.
    # Window k covers
    # [time_start + k * (rollout_steps + time_span),
    #  time_start + k * (rollout_steps + time_span) + rollout_steps).
    time_span: int = 10

    gzip_threshold_low: Optional[float] = 0.1
    gzip_threshold_high: Optional[float] = 0.15

    # Number of candidate trajectories to generate in parallel per sampling round.
    batch_candidate_size: int = 65536
    max_sampling_rounds: int = 200
    max_data_seq_len: int = 9000

    # Reserve 0 for PAD and 1 for the default mask token used by PuzzleDataset.
    token_offset: int = 2
    mask_token_id: int = 1
    target_mask_ratio: float = 0.25
    save_dtype: str = "int32"

    @property
    def sampled_state_height_min(self) -> int:
        return self.state_height if self.state_height_min is None else self.state_height_min

    @property
    def sampled_state_height_max(self) -> int:
        return self.state_height if self.state_height_max is None else self.state_height_max

    @property
    def sampled_rollout_steps_min(self) -> int:
        return self.rollout_steps if self.rollout_steps_min is None else self.rollout_steps_min

    @property
    def sampled_rollout_steps_max(self) -> int:
        return self.rollout_steps if self.rollout_steps_max is None else self.rollout_steps_max

    @property
    def sampled_counts_min(self) -> int:
        return self.counts if self.counts_min is None else self.counts_min

    @property
    def sampled_counts_max(self) -> int:
        return self.counts if self.counts_max is None else self.counts_max

    @property
    def resolved_start_step(self) -> int:
        return self.start_step if self.time_start is None else self.time_start

    @property
    def valid_state_heights(self) -> list[int]:
        return [
            height
            for height in range(self.sampled_state_height_min, self.sampled_state_height_max + 1)
            if height % self.patch_size == 0
        ]

    @property
    def max_counts_allowed_by_canvas(self) -> int:
        tokens_per_count = self.sampled_state_height_max * self.max_num_frames
        if tokens_per_count <= 0:
            return 0
        return (self.max_data_seq_len - 1) // tokens_per_count

    @property
    def resolved_counts_max(self) -> int:
        return min(self.sampled_counts_max, self.max_counts_allowed_by_canvas)

    @model_validator(mode="after")
    def _validate(self) -> "NCA1DDataConfig":
        if self.train_size <= 0:
            raise ValueError(f"train_size must be > 0, got {self.train_size}")
        if self.test_size <= 0:
            raise ValueError(f"test_size must be > 0, got {self.test_size}")
        if self.state_height <= 0:
            raise ValueError(f"state_height must be > 0, got {self.state_height}")
        if self.sampled_state_height_min <= 0:
            raise ValueError(
                f"state_height_min/state_height must be > 0, got {self.sampled_state_height_min}"
            )
        if self.sampled_state_height_min > self.sampled_state_height_max:
            raise ValueError(
                "state_height_min must be <= state_height_max, "
                f"got min={self.sampled_state_height_min}, max={self.sampled_state_height_max}"
            )
        if self.state_width != 1:
            raise ValueError(
                f"state_width must be 1 for the requested 1xH -> HxTxC dataset, got {self.state_width}"
            )
        if self.num_colors <= 1:
            raise ValueError(f"num_colors must be > 1, got {self.num_colors}")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {self.patch_size}")
        if self.state_width % self.patch_size != 0:
            raise ValueError(
                "state_width must be divisible by patch_size, "
                f"got state_width={self.state_width} and patch_size={self.patch_size}"
            )
        if not self.valid_state_heights:
            raise ValueError(
                "No valid state heights in the requested range are divisible by patch_size, "
                f"got range=({self.sampled_state_height_min}, {self.sampled_state_height_max}) "
                f"and patch_size={self.patch_size}"
            )
        if self.rollout_steps <= 0:
            raise ValueError(f"rollout_steps must be > 0, got {self.rollout_steps}")
        if self.sampled_rollout_steps_min <= 0:
            raise ValueError(
                "rollout_steps_min/rollout_steps must be > 0, "
                f"got {self.sampled_rollout_steps_min}"
            )
        if self.sampled_rollout_steps_min > self.sampled_rollout_steps_max:
            raise ValueError(
                "rollout_steps_min must be <= rollout_steps_max, "
                f"got min={self.sampled_rollout_steps_min}, max={self.sampled_rollout_steps_max}"
            )
        if self.counts <= 0:
            raise ValueError(f"counts must be > 0, got {self.counts}")
        if self.sampled_counts_min <= 0:
            raise ValueError(f"counts_min/counts must be > 0, got {self.sampled_counts_min}")
        if self.sampled_counts_min > self.sampled_counts_max:
            raise ValueError(
                "counts_min must be <= counts_max, "
                f"got min={self.sampled_counts_min}, max={self.sampled_counts_max}"
            )
        if self.time_subsample <= 0:
            raise ValueError(f"time_subsample must be > 0, got {self.time_subsample}")
        if self.start_step < 0:
            raise ValueError(f"start_step must be >= 0, got {self.start_step}")
        if self.time_start is not None and self.time_start < 0:
            raise ValueError(f"time_start must be >= 0, got {self.time_start}")
        if self.time_span < 0:
            raise ValueError(f"time_span must be >= 0, got {self.time_span}")
        if self.batch_candidate_size <= 0:
            raise ValueError(f"batch_candidate_size must be > 0, got {self.batch_candidate_size}")
        if self.max_sampling_rounds <= 0:
            raise ValueError(f"max_sampling_rounds must be > 0, got {self.max_sampling_rounds}")
        if self.max_data_seq_len <= 0:
            raise ValueError(f"max_data_seq_len must be > 0, got {self.max_data_seq_len}")
        if self.token_offset < 2:
            raise ValueError(f"token_offset must be >= 2, got {self.token_offset}")
        if not (0 < self.mask_token_id < self.token_offset):
            raise ValueError(
                "mask_token_id must be in the reserved token range [1, token_offset), "
                f"got mask_token_id={self.mask_token_id}, token_offset={self.token_offset}"
            )
        if not (0.0 <= self.target_mask_ratio <= 1.0):
            raise ValueError(
                f"target_mask_ratio must be in [0, 1], got {self.target_mask_ratio}"
            )
        if self.gzip_threshold_low is not None and self.gzip_threshold_low < 0.0:
            raise ValueError(
                f"gzip_threshold_low must be >= 0.0 or None, got {self.gzip_threshold_low}"
            )
        if self.gzip_threshold_high is not None and self.gzip_threshold_high < 0.0:
            raise ValueError(
                f"gzip_threshold_high must be >= 0.0 or None, got {self.gzip_threshold_high}"
            )
        if (
            self.gzip_threshold_low is not None
            and self.gzip_threshold_high is not None
            and self.gzip_threshold_low >= self.gzip_threshold_high
        ):
            raise ValueError(
                "gzip_threshold_low must be < gzip_threshold_high when both are set, "
                f"got low={self.gzip_threshold_low}, high={self.gzip_threshold_high}"
            )
        if self.max_counts_allowed_by_canvas < 1:
            raise ValueError(
                "The maximum state_height x num_frames already reaches the max_data_seq_len cap "
                f"with a single count: state_height_max={self.sampled_state_height_max}, "
                f"max_num_frames={self.max_num_frames}, max_data_seq_len={self.max_data_seq_len}"
            )
        return self

    @property
    def max_num_frames(self) -> int:
        return math.ceil(self.sampled_rollout_steps_max / self.time_subsample)

    @property
    def num_frames(self) -> int:
        return self.max_num_frames

    @property
    def final_image_shape(self) -> tuple[int, int, int]:
        return self.sampled_state_height_max, self.max_num_frames, self.resolved_counts_max

    @property
    def seq_len(self) -> int:
        height, width, counts = self.final_image_shape
        return height * width * counts

    @property
    def position_id_shape(self) -> tuple[int, int, int]:
        height, width, counts = self.final_image_shape
        return counts, height, width

    @property
    def vocab_size(self) -> int:
        return self.num_colors + self.token_offset


def make_nca_config(config: NCA1DDataConfig) -> NCAConfig:
    return NCAConfig(
        grid_height=config.state_height,
        grid_width=config.state_width,
        num_colors=config.num_colors,
        temperature=config.temperature,
        identity_bias=config.identity_bias,
        conv_channels=config.conv_channels,
        hidden_dim=config.hidden_dim,
        patch_size=config.patch_size,
        seq_len=config.seq_len,
        rollout_steps=config.sampled_rollout_steps_max,
        time_subsample=config.time_subsample,
        start_step=config.resolved_start_step,
        gzip_threshold_low=config.gzip_threshold_low or 0.0,
        gzip_threshold_high=config.gzip_threshold_high,
        batch_candidate_size=config.batch_candidate_size,
        max_sampling_rounds=config.max_sampling_rounds,
        train_size=config.train_size,
        val_size=config.test_size,
        out_dir=config.output_dir,
        save_dtype=config.save_dtype,
    )


def make_nca_config_for_sample(
    config: NCA1DDataConfig,
    state_height: int,
    rollout_steps: int,
) -> NCAConfig:
    return NCAConfig(
        grid_height=state_height,
        grid_width=config.state_width,
        num_colors=config.num_colors,
        temperature=config.temperature,
        identity_bias=config.identity_bias,
        conv_channels=config.conv_channels,
        hidden_dim=config.hidden_dim,
        patch_size=config.patch_size,
        seq_len=config.seq_len,
        rollout_steps=rollout_steps,
        time_subsample=config.time_subsample,
        start_step=config.resolved_start_step,
        gzip_threshold_low=config.gzip_threshold_low or 0.0,
        gzip_threshold_high=config.gzip_threshold_high,
        batch_candidate_size=config.batch_candidate_size,
        max_sampling_rounds=config.max_sampling_rounds,
        train_size=config.train_size,
        val_size=config.test_size,
        out_dir=config.output_dir,
        save_dtype=config.save_dtype,
    )


def num_frames_for_rollout_steps(rollout_steps: int, time_subsample: int) -> int:
    return math.ceil(rollout_steps / time_subsample)


def time_window_stride(rollout_steps: int, time_span: int) -> int:
    if rollout_steps <= 0:
        raise ValueError(f"rollout_steps must be > 0, got {rollout_steps}")
    if time_span < 0:
        raise ValueError(f"time_span must be >= 0, got {time_span}")
    return rollout_steps + time_span


def counts_limit_for_shape(
    state_height: int,
    num_frames: int,
    max_data_seq_len: int,
) -> int:
    tokens_per_count = state_height * num_frames
    if tokens_per_count <= 0:
        raise ValueError(
            f"state_height and num_frames must be > 0, got {state_height=} and {num_frames=}"
        )
    return (max_data_seq_len - 1) // tokens_per_count


def _configure_generation_backend(device: torch.device) -> None:
    if device.type != "cuda":
        return

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except RuntimeError:
        pass


def _sample_batched_rule_parameters(
    cfg: NCAConfig,
    batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        "conv3_weight": torch.randn(
            batch_size,
            cfg.conv_channels,
            cfg.num_colors,
            3,
            3,
            device=device,
        ),
        "conv3_bias": torch.randn(batch_size, cfg.conv_channels, device=device),
        "fc1_weight": torch.randn(
            batch_size,
            cfg.hidden_dim,
            cfg.conv_channels,
            1,
            1,
            device=device,
        ),
        "fc1_bias": torch.randn(batch_size, cfg.hidden_dim, device=device),
        "fc2_weight": torch.randn(
            batch_size,
            cfg.num_colors,
            cfg.hidden_dim,
            1,
            1,
            device=device,
        ),
        "fc2_bias": torch.randn(batch_size, cfg.num_colors, device=device),
    }


def _batched_step(
    state: torch.Tensor,
    rule_parameters: Dict[str, torch.Tensor],
    cfg: NCAConfig,
) -> torch.Tensor:
    batch_size, height, width = state.shape
    num_colors = cfg.num_colors

    x = F.one_hot(state.long(), num_classes=num_colors).float().permute(0, 3, 1, 2).contiguous()
    x = F.pad(x, (1, 1, 1, 1), mode="circular")
    x = F.conv2d(
        x.reshape(1, batch_size * num_colors, height + 2, width + 2),
        rule_parameters["conv3_weight"].reshape(batch_size * cfg.conv_channels, num_colors, 3, 3),
        bias=rule_parameters["conv3_bias"].reshape(batch_size * cfg.conv_channels),
        groups=batch_size,
    )
    x = x.reshape(batch_size, cfg.conv_channels, height, width)
    x = F.conv2d(
        x.reshape(1, batch_size * cfg.conv_channels, height, width),
        rule_parameters["fc1_weight"].reshape(batch_size * cfg.hidden_dim, cfg.conv_channels, 1, 1),
        bias=rule_parameters["fc1_bias"].reshape(batch_size * cfg.hidden_dim),
        groups=batch_size,
    )
    x = F.relu(x.reshape(batch_size, cfg.hidden_dim, height, width))
    x = F.conv2d(
        x.reshape(1, batch_size * cfg.hidden_dim, height, width),
        rule_parameters["fc2_weight"].reshape(batch_size * num_colors, cfg.hidden_dim, 1, 1),
        bias=rule_parameters["fc2_bias"].reshape(batch_size * num_colors),
        groups=batch_size,
    )
    logits = x.reshape(batch_size, num_colors, height, width).permute(0, 2, 3, 1).contiguous()
    one_hot_state = F.one_hot(state.long(), num_classes=num_colors).float()
    logits = logits + cfg.identity_bias * one_hot_state
    logits = logits / max(cfg.temperature, 1e-8)
    probs = torch.softmax(logits, dim=-1)
    next_state = torch.multinomial(probs.reshape(-1, num_colors), num_samples=1)
    return next_state.reshape(batch_size, height, width)


@torch.no_grad()
def rollout_trajectories_batched(
    cfg: NCAConfig,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    state = torch.randint(
        low=0,
        high=cfg.num_colors,
        size=(batch_size, cfg.grid_height, cfg.grid_width),
        device=device,
        dtype=torch.long,
    )
    rule_parameters = _sample_batched_rule_parameters(cfg, batch_size=batch_size, device=device)
    frames: list[torch.Tensor] = []
    total_steps = cfg.start_step + cfg.rollout_steps

    for t in range(total_steps):
        if t >= cfg.start_step and ((t - cfg.start_step) % cfg.time_subsample == 0):
            frames.append(state)
        state = _batched_step(state, rule_parameters, cfg)

    trajectories = torch.stack(frames, dim=1)
    return trajectories.detach().cpu().numpy().astype(np.int16, copy=False)


@torch.no_grad()
def rollout_trajectories_shared_rule(
    cfg: NCAConfig,
    count: int,
    device: torch.device,
    time_span: int = 0,
) -> np.ndarray:
    return rollout_trajectory_sets_batched(
        cfg=cfg,
        sample_batch_size=1,
        count=count,
        time_span=time_span,
        device=device,
    )[0].detach().cpu().numpy().astype(np.int16, copy=False)


@torch.no_grad()
def rollout_trajectory_sets_batched(
    cfg: NCAConfig,
    sample_batch_size: int,
    count: int,
    device: torch.device,
    time_span: int = 0,
) -> torch.Tensor:
    if sample_batch_size <= 0:
        raise ValueError(f"sample_batch_size must be > 0, got {sample_batch_size}")
    if count <= 0:
        raise ValueError(f"count must be > 0, got {count}")
    if time_span < 0:
        raise ValueError(f"time_span must be >= 0, got {time_span}")

    state = torch.randint(
        low=0,
        high=cfg.num_colors,
        size=(sample_batch_size, cfg.grid_height, cfg.grid_width),
        device=device,
        dtype=torch.long,
    )
    rule_parameters = _sample_batched_rule_parameters(
        cfg,
        batch_size=sample_batch_size,
        device=device,
    )
    frames: list[torch.Tensor] = []
    num_frames = num_frames_for_rollout_steps(cfg.rollout_steps, cfg.time_subsample)
    window_stride = time_window_stride(cfg.rollout_steps, time_span)
    total_steps = cfg.start_step + (count - 1) * window_stride + cfg.rollout_steps

    for t in range(total_steps):
        relative_t = t - cfg.start_step
        if relative_t >= 0:
            window_idx = relative_t // window_stride
            offset_within_window = relative_t % window_stride
            if (
                window_idx < count
                # Each count keeps the same-length window [start, start + rollout_steps).
                and offset_within_window < cfg.rollout_steps
                and (offset_within_window % cfg.time_subsample == 0)
            ):
                frames.append(state)
        state = _batched_step(state, rule_parameters, cfg)

    expected_frame_count = count * num_frames
    if len(frames) != expected_frame_count:
        raise RuntimeError(
            "Collected an unexpected number of temporal-window frames, "
            f"got {len(frames)} but expected {expected_frame_count} "
            f"for count={count}, rollout_steps={cfg.rollout_steps}, "
            f"time_subsample={cfg.time_subsample}, time_span={time_span}"
        )

    return torch.stack(frames, dim=1).reshape(
        sample_batch_size,
        count,
        num_frames,
        cfg.grid_height,
        cfg.grid_width,
    )


def trajectory_to_time_image(trajectory: np.ndarray) -> np.ndarray:
    if trajectory.ndim != 3:
        raise ValueError(f"trajectory must have shape [T, H, W], got ndim={trajectory.ndim}")
    if trajectory.shape[2] != 1:
        raise ValueError(
            f"trajectory width must be 1 for 1xH -> HxT conversion, got shape={trajectory.shape}"
        )
    return np.squeeze(trajectory, axis=2).T.astype(np.int16, copy=False)


def trajectories_to_time_volume(trajectories: np.ndarray) -> np.ndarray:
    if trajectories.ndim != 4:
        raise ValueError(
            f"trajectories must have shape [counts, T, H, W], got ndim={trajectories.ndim}"
        )
    if trajectories.shape[3] != 1:
        raise ValueError(
            "trajectory width must be 1 for 1xH -> HxTxC conversion, "
            f"got shape={trajectories.shape}"
        )
    return np.transpose(np.squeeze(trajectories, axis=3), (2, 1, 0)).astype(np.int16, copy=False)


def trajectory_sets_to_time_volumes_batched(trajectory_sets: torch.Tensor) -> torch.Tensor:
    if trajectory_sets.ndim != 5:
        raise ValueError(
            "trajectory_sets must have shape [B, C, T, H, W], "
            f"got ndim={trajectory_sets.ndim}"
        )
    if trajectory_sets.shape[-1] != 1:
        raise ValueError(
            "trajectory width must be 1 for 1xH -> HxTxC conversion, "
            f"got shape={tuple(trajectory_sets.shape)}"
        )
    return trajectory_sets.squeeze(-1).permute(0, 3, 2, 1).contiguous()


def flatten_time_image(
    time_image: np.ndarray,
    token_offset: int,
    padded_height: Optional[int] = None,
    padded_num_frames: Optional[int] = None,
) -> np.ndarray:
    return flatten_time_volume(
        time_image[:, :, None],
        token_offset=token_offset,
        padded_height=padded_height,
        padded_num_frames=padded_num_frames,
        padded_counts=1,
    )


def flatten_time_volume(
    time_volume: np.ndarray,
    token_offset: int,
    padded_height: Optional[int] = None,
    padded_num_frames: Optional[int] = None,
    padded_counts: Optional[int] = None,
) -> np.ndarray:
    if time_volume.ndim != 3:
        raise ValueError(f"time_volume must have shape [H, T, C], got ndim={time_volume.ndim}")
    image_height, num_frames, counts = time_volume.shape
    padded_height = image_height if padded_height is None else padded_height
    padded_num_frames = num_frames if padded_num_frames is None else padded_num_frames
    padded_counts = counts if padded_counts is None else padded_counts

    if image_height > padded_height or num_frames > padded_num_frames or counts > padded_counts:
        raise ValueError(
            "time_volume must fit inside the padded canvas, "
            f"got image_shape={(image_height, num_frames, counts)} and "
            f"padded_shape={(padded_height, padded_num_frames, padded_counts)}"
        )

    canvas = np.zeros((padded_height, padded_num_frames, padded_counts), dtype=np.int32)
    canvas[:image_height, :num_frames, :counts] = (
        time_volume.astype(np.int32, copy=False) + token_offset
    )
    return canvas.reshape(-1)


def unflatten_time_image(
    flat_tokens: np.ndarray,
    image_height: int,
    num_frames: int,
    token_offset: int,
    padded_height: Optional[int] = None,
    padded_num_frames: Optional[int] = None,
) -> np.ndarray:
    return np.squeeze(
        unflatten_time_volume(
            flat_tokens,
            image_height=image_height,
            num_frames=num_frames,
            counts=1,
            token_offset=token_offset,
            padded_height=padded_height,
            padded_num_frames=padded_num_frames,
            padded_counts=1,
        ),
        axis=2,
    )


def unflatten_time_volume(
    flat_tokens: np.ndarray,
    image_height: int,
    num_frames: int,
    counts: int,
    token_offset: int,
    padded_height: Optional[int] = None,
    padded_num_frames: Optional[int] = None,
    padded_counts: Optional[int] = None,
) -> np.ndarray:
    padded_height = image_height if padded_height is None else padded_height
    padded_num_frames = num_frames if padded_num_frames is None else padded_num_frames
    padded_counts = counts if padded_counts is None else padded_counts
    canvas = flat_tokens.reshape(padded_height, padded_num_frames, padded_counts)
    return canvas[:image_height, :num_frames, :counts] - token_offset


def score_trajectory_set_gzip(trajectories: np.ndarray, tokenizer: NCATokenizer) -> float:
    if trajectories.ndim != 4:
        raise ValueError(
            f"trajectories must have shape [counts, T, H, W], got ndim={trajectories.ndim}"
        )
    frame_tokens = []
    for trajectory in trajectories:
        frame_tokens.extend(tokenizer.encode_frame(frame)[1:-1] for frame in trajectory)
    flat = np.concatenate(frame_tokens, axis=0)
    return gzip_complexity_ratio_from_tokens(flat)


def _trajectory_set_tokens_for_gzip_patch1(
    trajectory_sets: np.ndarray,
    num_frames: np.ndarray,
    counts: np.ndarray,
) -> list[np.ndarray]:
    if trajectory_sets.ndim != 5:
        raise ValueError(
            "trajectory_sets must have shape [B, C, T, H, W], "
            f"got ndim={trajectory_sets.ndim}"
        )
    if trajectory_sets.shape[-1] != 1:
        raise ValueError(
            "trajectory width must be 1 for 1xH gzip scoring, "
            f"got shape={trajectory_sets.shape}"
        )

    token_sets: list[np.ndarray] = []
    for batch_idx in range(trajectory_sets.shape[0]):
        token_sets.append(
            trajectory_sets[
                batch_idx,
                : int(counts[batch_idx]),
                : int(num_frames[batch_idx]),
                :,
                0,
            ].reshape(-1).astype(np.int32, copy=False)
        )
    return token_sets


def score_trajectory_sets_gzip_batched(
    trajectory_sets: np.ndarray,
    num_frames: np.ndarray,
    counts: np.ndarray,
    tokenizer: NCATokenizer,
) -> np.ndarray:
    if tokenizer.patch != 1:
        scores = [
            score_trajectory_set_gzip(
                trajectory_sets[idx, : int(counts[idx]), : int(num_frames[idx])],
                tokenizer,
            )
            for idx in range(trajectory_sets.shape[0])
        ]
        return np.asarray(scores, dtype=np.float32)

    token_sets = _trajectory_set_tokens_for_gzip_patch1(trajectory_sets, num_frames, counts)
    max_workers = min(len(token_sets), max(os.cpu_count() or 1, 1))
    if max_workers <= 1:
        scores = [gzip_complexity_ratio_from_tokens(tokens) for tokens in token_sets]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            scores = list(executor.map(gzip_complexity_ratio_from_tokens, token_sets))
    return np.asarray(scores, dtype=np.float32)


def _make_target_mask(
    height: int,
    num_frames: int,
    rng: np.random.Generator,
    mask_ratio: float,
) -> np.ndarray:
    mask = rng.random((height, num_frames)) < mask_ratio
    if mask_ratio > 0.0 and not mask.any() and height > 0 and num_frames > 0:
        flat_index = int(rng.integers(height * num_frames))
        mask.reshape(-1)[flat_index] = True
    return mask


def build_sample_arrays(
    time_volume: np.ndarray,
    config: NCA1DDataConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if time_volume.ndim != 3:
        raise ValueError(f"time_volume must have shape [H, T, C], got ndim={time_volume.ndim}")

    image_height, num_frames, counts = time_volume.shape
    padded_height, padded_num_frames, padded_counts = config.final_image_shape

    if (
        image_height > padded_height
        or num_frames > padded_num_frames
        or counts > padded_counts
    ):
        raise ValueError(
            "time_volume must fit inside the fixed dataset canvas, "
            f"got image_shape={(image_height, num_frames, counts)} and "
            f"padded_shape={config.final_image_shape}"
        )

    encoded_volume = time_volume.astype(np.int32, copy=False) + config.token_offset

    input_canvas = np.zeros((padded_height, padded_num_frames, padded_counts), dtype=np.int32)
    input_canvas[:image_height, :num_frames, :counts] = encoded_volume

    label_canvas = np.zeros_like(input_canvas)
    label_canvas[:image_height, :num_frames, counts - 1] = encoded_volume[:, :, counts - 1]

    target_mask = _make_target_mask(
        image_height,
        num_frames,
        rng=rng,
        mask_ratio=config.target_mask_ratio,
    )
    input_canvas[:image_height, :num_frames, counts - 1][target_mask] = config.mask_token_id

    position_ids = np.zeros((padded_height, padded_num_frames, padded_counts, 3), dtype=np.int32)
    position_ids[:image_height, :num_frames, :counts, 0] = np.arange(
        counts,
        dtype=np.int32,
    )[None, None, :]
    position_ids[:image_height, :num_frames, :counts, 1] = np.arange(
        image_height,
        dtype=np.int32,
    )[:, None, None]
    position_ids[:image_height, :num_frames, :counts, 2] = np.arange(
        num_frames,
        dtype=np.int32,
    )[None, :, None]

    return (
        input_canvas.reshape(-1),
        label_canvas.reshape(-1),
        position_ids.reshape(-1, 3),
    )


def build_sample_arrays_batched(
    time_volumes: torch.Tensor,
    *,
    num_frames: np.ndarray,
    counts: np.ndarray,
    config: NCA1DDataConfig,
    rng: np.random.Generator,
    mask_generator: Optional[torch.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if time_volumes.ndim != 4:
        raise ValueError(
            f"time_volumes must have shape [B, H, T, C], got ndim={time_volumes.ndim}"
        )

    batch_size, image_height, max_num_frames, max_counts = time_volumes.shape
    padded_height, padded_num_frames, padded_counts = config.final_image_shape
    if (
        image_height > padded_height
        or max_num_frames > padded_num_frames
        or max_counts > padded_counts
    ):
        raise ValueError(
            "time_volumes must fit inside the fixed dataset canvas, "
            f"got image_shape={(image_height, max_num_frames, max_counts)} and "
            f"padded_shape={config.final_image_shape}"
        )

    device = time_volumes.device
    dtype = torch.int32
    num_frames_t = torch.as_tensor(num_frames, device=device, dtype=torch.int64)
    counts_t = torch.as_tensor(counts, device=device, dtype=torch.int64)

    encoded = time_volumes.to(dtype=dtype) + config.token_offset

    input_canvas = torch.zeros(
        (batch_size, padded_height, padded_num_frames, padded_counts),
        dtype=dtype,
        device=device,
    )
    input_canvas[:, :image_height, :max_num_frames, :max_counts] = encoded

    time_ids = torch.arange(max_num_frames, device=device, dtype=torch.int64).view(1, 1, max_num_frames, 1)
    count_ids = torch.arange(max_counts, device=device, dtype=torch.int64).view(1, 1, 1, max_counts)
    valid_mask = (time_ids < num_frames_t.view(-1, 1, 1, 1)) & (
        count_ids < counts_t.view(-1, 1, 1, 1)
    )
    input_canvas[:, :image_height, :max_num_frames, :max_counts] *= valid_mask.to(dtype)

    label_canvas = torch.zeros_like(input_canvas)
    label_region = torch.zeros(
        (batch_size, image_height, max_num_frames, max_counts),
        dtype=dtype,
        device=device,
    )
    last_count_idx = (counts_t - 1).clamp_min(0)
    last_count_idx_expanded = last_count_idx.view(-1, 1, 1, 1).expand(-1, image_height, max_num_frames, 1)
    last_slices = encoded.gather(dim=3, index=last_count_idx_expanded)
    valid_last = (time_ids[:, :, :, :1] < num_frames_t.view(-1, 1, 1, 1)).to(dtype)
    label_region.scatter_(3, last_count_idx_expanded, last_slices * valid_last)
    label_canvas[:, :image_height, :max_num_frames, :max_counts] = label_region

    valid_target = (
        torch.arange(max_num_frames, device=device, dtype=torch.int64).view(1, 1, max_num_frames)
        < num_frames_t.view(-1, 1, 1)
    ).expand(-1, image_height, -1)
    target_mask = torch.rand(
        (batch_size, image_height, max_num_frames),
        device=device,
        generator=mask_generator,
    ) < config.target_mask_ratio
    target_mask &= valid_target

    if config.target_mask_ratio > 0.0:
        needs_one = ~target_mask.reshape(batch_size, -1).any(dim=1) & valid_target.reshape(batch_size, -1).any(dim=1)
        if torch.any(needs_one):
            missing_indices = torch.nonzero(needs_one, as_tuple=False).squeeze(1)
            random_rows = torch.randint(
                image_height,
                (missing_indices.numel(),),
                device=device,
                generator=mask_generator,
            )
            random_cols = (
                torch.rand(
                    missing_indices.numel(),
                    device=device,
                    generator=mask_generator,
                ) * num_frames_t[missing_indices].to(torch.float32)
            ).to(torch.int64)
            target_mask[missing_indices, random_rows, random_cols] = True

    input_region = input_canvas[:, :image_height, :max_num_frames, :max_counts]
    last_inputs = input_region.gather(dim=3, index=last_count_idx_expanded).squeeze(-1)
    last_inputs = torch.where(
        target_mask,
        torch.full_like(last_inputs, config.mask_token_id),
        last_inputs,
    )
    input_region.scatter_(3, last_count_idx_expanded, last_inputs.unsqueeze(-1))

    position_ids = torch.zeros(
        (batch_size, padded_height, padded_num_frames, padded_counts, 3),
        dtype=dtype,
        device=device,
    )
    depth_grid = torch.arange(max_counts, dtype=dtype, device=device).view(1, 1, 1, max_counts)
    row_grid = torch.arange(image_height, dtype=dtype, device=device).view(1, image_height, 1, 1)
    col_grid = torch.arange(max_num_frames, dtype=dtype, device=device).view(1, 1, max_num_frames, 1)
    position_region = torch.stack(
        [
            depth_grid.expand(batch_size, image_height, max_num_frames, max_counts),
            row_grid.expand(batch_size, image_height, max_num_frames, max_counts),
            col_grid.expand(batch_size, image_height, max_num_frames, max_counts),
        ],
        dim=-1,
    )
    position_ids[:, :image_height, :max_num_frames, :max_counts] = position_region * valid_mask.unsqueeze(-1).to(dtype)

    return (
        input_canvas.reshape(batch_size, -1).cpu().numpy(),
        label_canvas.reshape(batch_size, -1).cpu().numpy(),
        position_ids.reshape(batch_size, -1, 3).cpu().numpy(),
    )


def _passes_gzip_filter(score: float, config: NCA1DDataConfig) -> bool:
    if config.gzip_threshold_low is not None and score < config.gzip_threshold_low:
        return False
    if config.gzip_threshold_high is not None and score >= config.gzip_threshold_high:
        return False
    return True


def _shape_group_specs(config: NCA1DDataConfig) -> list[tuple[int, int]]:
    return [
        (state_height, rollout_steps)
        for state_height in config.valid_state_heights
        for rollout_steps in range(
            config.sampled_rollout_steps_min,
            config.sampled_rollout_steps_max + 1,
        )
    ]


def _allocate_examples_per_shape_group(
    size: int,
    shape_groups: list[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}")
    if not shape_groups:
        raise ValueError("shape_groups must not be empty")

    base_examples = size // len(shape_groups)
    remainder = size % len(shape_groups)
    allocated = np.full(len(shape_groups), base_examples, dtype=np.int32)
    if remainder > 0:
        allocated[rng.permutation(len(shape_groups))[:remainder]] += 1
    return allocated


def _shuffle_split_arrays(
    split_arrays: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    if not split_arrays:
        return split_arrays

    num_examples = int(split_arrays["inputs"].shape[0])
    if num_examples <= 1:
        return split_arrays

    permutation = rng.permutation(num_examples)
    shuffled: Dict[str, np.ndarray] = {}
    for key, value in split_arrays.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == num_examples:
            shuffled[key] = value[permutation]
        else:
            shuffled[key] = value
    return shuffled


def _make_split_arrays(
    flat_inputs: list[np.ndarray],
    flat_labels: list[np.ndarray],
    position_ids: list[np.ndarray],
    gzip_scores: list[float],
    state_heights: list[int],
    num_frames: list[int],
    rollout_steps: list[int],
    counts: list[int],
    config: NCA1DDataConfig,
) -> Dict[str, np.ndarray]:
    num_examples = len(flat_inputs)

    inputs = np.stack(flat_inputs, axis=0).astype(config.save_dtype, copy=False)
    labels = np.stack(flat_labels, axis=0).astype(config.save_dtype, copy=False)

    results: Dict[str, np.ndarray] = {
        "inputs": inputs,
        "labels": labels,
        "position_ids": np.stack(position_ids, axis=0).astype(np.int32, copy=False),
        "puzzle_identifiers": np.zeros((num_examples,), dtype=np.int32),
        "puzzle_indices": np.arange(num_examples + 1, dtype=np.int32),
        "group_indices": np.arange(num_examples + 1, dtype=np.int32),
        "gzip_ratio": np.asarray(gzip_scores, dtype=np.float32),
        "state_heights": np.asarray(state_heights, dtype=np.int32),
        "num_frames": np.asarray(num_frames, dtype=np.int32),
        "rollout_steps": np.asarray(rollout_steps, dtype=np.int32),
        "counts": np.asarray(counts, dtype=np.int32),
    }
    return results


def generate_split(split_name: str, size: int, seed: int, config: NCA1DDataConfig) -> Dict[str, np.ndarray]:
    seed_everything(seed)
    device = get_device()
    _configure_generation_backend(device)
    rng = np.random.default_rng(seed)
    mask_generator = torch.Generator(device=device)
    mask_generator.manual_seed(seed + 1_000_000)

    flat_inputs: list[np.ndarray] = []
    flat_labels: list[np.ndarray] = []
    split_position_ids: list[np.ndarray] = []
    gzip_scores: list[float] = []
    state_heights: list[int] = []
    num_frames: list[int] = []
    rollout_steps: list[int] = []
    counts: list[int] = []

    shape_groups = _shape_group_specs(config)
    allocated_examples = _allocate_examples_per_shape_group(size, shape_groups, rng)
    progress = tqdm(total=size, desc=f"Generating {split_name}", leave=False)

    for (sampled_state_height, sampled_group_rollout_steps), target_group_size in zip(
        shape_groups,
        allocated_examples.tolist(),
    ):
        if target_group_size <= 0:
            continue

        target_group_size = int(target_group_size)
        group_num_frames_value = num_frames_for_rollout_steps(
            sampled_group_rollout_steps,
            config.time_subsample,
        )
        group_count_limit = counts_limit_for_shape(
            sampled_state_height,
            group_num_frames_value,
            config.max_data_seq_len,
        )
        candidate_batch_size = max(
            1,
            min(
                config.batch_candidate_size,
                max(target_group_size, 64),
            ),
        )
        min_rounds_for_group = math.ceil(target_group_size / candidate_batch_size)
        effective_max_sampling_rounds = max(config.max_sampling_rounds, min_rounds_for_group)

        nca_config = make_nca_config_for_sample(
            config,
            state_height=int(sampled_state_height),
            rollout_steps=int(sampled_group_rollout_steps),
        )
        tokenizer = NCATokenizer(nca_config)
        collected_group_size = 0
        rounds = 0

        while collected_group_size < target_group_size and rounds < effective_max_sampling_rounds:
            rounds += 1

            sampled_counts = rng.integers(
                config.sampled_counts_min,
                config.sampled_counts_max + 1,
                size=candidate_batch_size,
            )
            sampled_counts = np.minimum(sampled_counts, config.resolved_counts_max).astype(
                np.int32,
                copy=False,
            )
            sampled_counts = np.minimum(sampled_counts, group_count_limit).astype(
                np.int32,
                copy=False,
            )

            valid_counts = sampled_counts >= 1
            if not np.any(valid_counts):
                continue

            group_counts = sampled_counts[valid_counts]
            group_num_frames = np.full(
                group_counts.shape[0],
                group_num_frames_value,
                dtype=np.int32,
            )
            group_rollout_steps = np.full(
                group_counts.shape[0],
                sampled_group_rollout_steps,
                dtype=np.int32,
            )

            batched_trajectory_sets = rollout_trajectory_sets_batched(
                nca_config,
                sample_batch_size=int(group_counts.shape[0]),
                count=int(group_counts.max()),
                time_span=config.time_span,
                device=device,
            )
            batched_trajectory_sets_cpu = batched_trajectory_sets.cpu().numpy()
            group_gzip_scores = score_trajectory_sets_gzip_batched(
                batched_trajectory_sets_cpu,
                num_frames=group_num_frames,
                counts=group_counts,
                tokenizer=tokenizer,
            )
            accepted_mask = np.array(
                [_passes_gzip_filter(float(score), config) for score in group_gzip_scores],
                dtype=bool,
            )
            if not np.any(accepted_mask):
                continue

            remaining = target_group_size - collected_group_size
            accepted_indices = np.flatnonzero(accepted_mask)[:remaining]
            accepted_num_frames = group_num_frames[accepted_indices]
            accepted_rollout_steps = group_rollout_steps[accepted_indices]
            accepted_counts = group_counts[accepted_indices]
            accepted_scores = group_gzip_scores[accepted_indices]

            accepted_trajectory_sets = batched_trajectory_sets[
                torch.as_tensor(
                    accepted_indices,
                    device=batched_trajectory_sets.device,
                    dtype=torch.long,
                )
            ]
            accepted_time_volumes = trajectory_sets_to_time_volumes_batched(
                accepted_trajectory_sets
            )
            batch_inputs, batch_labels, batch_position_ids = build_sample_arrays_batched(
                accepted_time_volumes,
                num_frames=accepted_num_frames,
                counts=accepted_counts,
                config=config,
                rng=rng,
                mask_generator=mask_generator,
            )

            flat_inputs.extend(batch_inputs)
            flat_labels.extend(batch_labels)
            split_position_ids.extend(batch_position_ids)
            gzip_scores.extend(accepted_scores.tolist())
            state_heights.extend([int(sampled_state_height)] * accepted_indices.size)
            num_frames.extend(accepted_num_frames.astype(np.int32).tolist())
            rollout_steps.extend(accepted_rollout_steps.astype(np.int32).tolist())
            counts.extend(accepted_counts.astype(np.int32).tolist())
            collected_group_size += int(accepted_indices.size)
            progress.update(int(accepted_indices.size))

        if collected_group_size < target_group_size:
            progress.close()
            raise RuntimeError(
                f"Could not collect enough {split_name} examples for "
                f"state_height={sampled_state_height}, rollout_steps={sampled_group_rollout_steps}. "
                f"Collected {collected_group_size} / {target_group_size} after {rounds} rounds. "
                "Try increasing max_sampling_rounds or batch_candidate_size, or relax the gzip thresholds."
            )

    progress.close()

    if len(flat_inputs) != size:
        raise RuntimeError(
            f"Expected to collect exactly {size} {split_name} examples, "
            f"but got {len(flat_inputs)}."
        )

    split_arrays = _make_split_arrays(
        flat_inputs,
        flat_labels,
        split_position_ids,
        gzip_scores,
        state_heights,
        num_frames,
        rollout_steps,
        counts,
        config,
    )
    return _shuffle_split_arrays(split_arrays, rng)


def save_split(split_name: str, split_arrays: Dict[str, np.ndarray], config: NCA1DDataConfig) -> None:
    split_dir = Path(config.output_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    split_position_id_shape = (
        split_arrays["position_ids"].max(axis=(0, 1)).astype(np.int32) + 1
        if split_arrays["position_ids"].size > 0
        else None
    )

    metadata = PuzzleDatasetMetadata(
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=split_arrays["group_indices"].size - 1,
        mean_puzzle_examples=1.0,
        sets=["all"],
        position_id_shape=split_position_id_shape.tolist() if split_position_id_shape is not None else None,
    )

    with open(split_dir / "dataset.json", "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    np.save(split_dir / "all__inputs.npy", split_arrays["inputs"])
    np.save(split_dir / "all__labels.npy", split_arrays["labels"])
    np.save(split_dir / "all__position_ids.npy", split_arrays["position_ids"])
    np.save(split_dir / "all__puzzle_identifiers.npy", split_arrays["puzzle_identifiers"])
    np.save(split_dir / "all__puzzle_indices.npy", split_arrays["puzzle_indices"])
    np.save(split_dir / "all__group_indices.npy", split_arrays["group_indices"])
    np.save(split_dir / "all__gzip_ratio.npy", split_arrays["gzip_ratio"])
    np.save(split_dir / "all__state_heights.npy", split_arrays["state_heights"])
    np.save(split_dir / "all__num_frames.npy", split_arrays["num_frames"])
    np.save(split_dir / "all__rollout_steps.npy", split_arrays["rollout_steps"])
    np.save(split_dir / "all__counts.npy", split_arrays["counts"])

    with open(split_dir / "summary.json", "w") as f:
        json.dump(
            {
                "split": split_name,
                "num_examples": int(split_arrays["inputs"].shape[0]),
                "seq_len": int(config.seq_len),
                "final_image_shape": list(config.final_image_shape),
                "gzip_ratio_mean": float(split_arrays["gzip_ratio"].mean()),
                "gzip_ratio_std": float(split_arrays["gzip_ratio"].std()),
                "gzip_ratio_min": float(split_arrays["gzip_ratio"].min()),
                "gzip_ratio_max": float(split_arrays["gzip_ratio"].max()),
                "state_height_min": int(split_arrays["state_heights"].min()),
                "state_height_max": int(split_arrays["state_heights"].max()),
                "num_frames_min": int(split_arrays["num_frames"].min()),
                "num_frames_max": int(split_arrays["num_frames"].max()),
                "rollout_steps_min": int(split_arrays["rollout_steps"].min()),
                "rollout_steps_max": int(split_arrays["rollout_steps"].max()),
                "counts_min": int(split_arrays["counts"].min()),
                "counts_max": int(split_arrays["counts"].max()),
                "position_id_shape": metadata.position_id_shape,
            },
            f,
            indent=2,
        )


def save_dataset_config(config: NCA1DDataConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(
            {
                **config.model_dump(),
                "resolved_state_height_min": config.sampled_state_height_min,
                "resolved_state_height_max": config.sampled_state_height_max,
                "resolved_start_step": config.resolved_start_step,
                "resolved_rollout_steps_min": config.sampled_rollout_steps_min,
                "resolved_rollout_steps_max": config.sampled_rollout_steps_max,
                "resolved_time_start": config.resolved_start_step,
                "num_frames": config.max_num_frames,
                "max_num_frames": config.max_num_frames,
                "seq_len": config.seq_len,
                "vocab_size": config.vocab_size,
                "final_image_shape": list(config.final_image_shape),
                "position_id_shape": list(config.position_id_shape),
                "requested_counts_min": config.sampled_counts_min,
                "requested_counts_max": config.sampled_counts_max,
                "resolved_counts_max": config.resolved_counts_max,
                "sample_random_state_height": config.sampled_state_height_min != config.sampled_state_height_max,
                "sample_random_rollout_steps": config.sampled_rollout_steps_min != config.sampled_rollout_steps_max,
                "sample_random_counts": config.sampled_counts_min != config.sampled_counts_max,
                "time_axis": "width",
                "count_axis": "depth",
                "max_data_seq_len": config.max_data_seq_len,
                "value_encoding": {
                    "pad": 0,
                    "mask": config.mask_token_id,
                    "nca_color_range": [config.token_offset, config.token_offset + config.num_colors - 1],
                },
                "target_mask_ratio": config.target_mask_ratio,
            },
            f,
            indent=2,
        )

    with open(output_dir / "identifiers.json", "w") as f:
        json.dump(["<blank>"], f, indent=2)


def build_dataset(config: NCA1DDataConfig) -> None:
    save_dataset_config(config)

    train_arrays = generate_split("train", config.train_size, config.seed, config)
    save_split("train", train_arrays, config)

    test_arrays = generate_split("test", config.test_size, config.seed + 1, config)
    save_split("test", test_arrays, config)


@cli.command(singleton=True)
def preprocess_data(config: NCA1DDataConfig) -> None:
    build_dataset(config)


if __name__ == "__main__":
    cli()
