from __future__ import annotations

import json
import math
import sys
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
    score_trajectory_gzip,
    seed_everything,
)


cli = ArgParser()


class NCA1DDataConfig(BaseModel):
    output_dir: str = "data/nca1d-default"

    train_size: int = 1024
    test_size: int = 256
    seed: int = 0

    state_height: int = 9
    state_height_min: Optional[int] = None
    state_height_max: Optional[int] = None
    state_width: int = 1

    num_colors: int = 8
    temperature: float = 1e-3
    identity_bias: float = 0.0
    conv_channels: int = 4
    hidden_dim: int = 16

    # Use patch_size=1 so 1-column NCA states remain tokenizable for gzip filtering.
    patch_size: int = 1

    rollout_steps: int = 9
    rollout_steps_min: Optional[int] = None
    rollout_steps_max: Optional[int] = None
    time_subsample: int = 1
    start_step: int = 0
    # When both are set, these directly specify the raw trajectory window [time_start, time_end].
    time_start: Optional[int] = 30
    time_end: Optional[int] = 38

    gzip_threshold_low: Optional[float] = None
    gzip_threshold_high: Optional[float] = None

    # Number of candidate trajectories to generate in parallel per sampling round.
    batch_candidate_size: int = 64
    max_sampling_rounds: int = 200

    # Reserve 0 for PAD and 1 for the default mask token used by PuzzleDataset.
    token_offset: int = 2
    save_dtype: str = "int32"

    @property
    def sampled_state_height_min(self) -> int:
        return self.state_height if self.state_height_min is None else self.state_height_min

    @property
    def sampled_state_height_max(self) -> int:
        return self.state_height if self.state_height_max is None else self.state_height_max

    @property
    def sampled_rollout_steps_min(self) -> int:
        if self.uses_explicit_time_range:
            return self.selected_time_span
        return self.rollout_steps if self.rollout_steps_min is None else self.rollout_steps_min

    @property
    def sampled_rollout_steps_max(self) -> int:
        if self.uses_explicit_time_range:
            return self.selected_time_span
        return self.rollout_steps if self.rollout_steps_max is None else self.rollout_steps_max

    @property
    def uses_explicit_time_range(self) -> bool:
        return self.time_start is not None or self.time_end is not None

    @property
    def resolved_start_step(self) -> int:
        return self.start_step if self.time_start is None else self.time_start

    @property
    def selected_time_span(self) -> int:
        if self.time_start is None or self.time_end is None:
            raise ValueError("time_start and time_end must both be set to use an explicit time range.")
        return self.time_end - self.time_start + 1

    @property
    def resolved_time_end(self) -> int:
        return self.resolved_start_step + self.sampled_rollout_steps_max - 1

    @property
    def valid_state_heights(self) -> list[int]:
        return [
            height
            for height in range(self.sampled_state_height_min, self.sampled_state_height_max + 1)
            if height % self.patch_size == 0
        ]

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
                f"state_width must be 1 for the requested 1xH -> HxT dataset, got {self.state_width}"
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
        if not self.uses_explicit_time_range and self.rollout_steps <= 0:
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
        if self.time_subsample <= 0:
            raise ValueError(f"time_subsample must be > 0, got {self.time_subsample}")
        if self.start_step < 0:
            raise ValueError(f"start_step must be >= 0, got {self.start_step}")
        if self.uses_explicit_time_range:
            if self.time_start is None or self.time_end is None:
                raise ValueError("time_start and time_end must both be set when using an explicit time range.")
            if self.time_start < 0:
                raise ValueError(f"time_start must be >= 0, got {self.time_start}")
            if self.time_end < self.time_start:
                raise ValueError(
                    f"time_end must be >= time_start, got time_start={self.time_start}, time_end={self.time_end}"
                )
        if self.batch_candidate_size <= 0:
            raise ValueError(f"batch_candidate_size must be > 0, got {self.batch_candidate_size}")
        if self.max_sampling_rounds <= 0:
            raise ValueError(f"max_sampling_rounds must be > 0, got {self.max_sampling_rounds}")
        if self.token_offset < 2:
            raise ValueError(f"token_offset must be >= 2, got {self.token_offset}")
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
        return self

    @property
    def max_num_frames(self) -> int:
        return math.ceil(self.sampled_rollout_steps_max / self.time_subsample)

    @property
    def num_frames(self) -> int:
        return self.max_num_frames

    @property
    def final_image_shape(self) -> tuple[int, int]:
        return self.sampled_state_height_max, self.max_num_frames

    @property
    def seq_len(self) -> int:
        height, width = self.final_image_shape
        return height * width

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


def trajectory_to_time_image(trajectory: np.ndarray) -> np.ndarray:
    if trajectory.ndim != 3:
        raise ValueError(f"trajectory must have shape [T, H, W], got ndim={trajectory.ndim}")
    if trajectory.shape[2] != 1:
        raise ValueError(
            f"trajectory width must be 1 for 1xH -> HxT conversion, got shape={trajectory.shape}"
        )
    return np.squeeze(trajectory, axis=2).T.astype(np.int16, copy=False)


def flatten_time_image(
    time_image: np.ndarray,
    token_offset: int,
    padded_height: Optional[int] = None,
    padded_num_frames: Optional[int] = None,
) -> np.ndarray:
    if time_image.ndim != 2:
        raise ValueError(f"time_image must have shape [H, T], got ndim={time_image.ndim}")
    image_height, num_frames = time_image.shape
    padded_height = image_height if padded_height is None else padded_height
    padded_num_frames = num_frames if padded_num_frames is None else padded_num_frames

    if image_height > padded_height or num_frames > padded_num_frames:
        raise ValueError(
            "time_image must fit inside the padded canvas, "
            f"got image_shape={(image_height, num_frames)} and "
            f"padded_shape={(padded_height, padded_num_frames)}"
        )

    canvas = np.zeros((padded_height, padded_num_frames), dtype=np.int32)
    canvas[:image_height, :num_frames] = time_image.astype(np.int32, copy=False) + token_offset
    return canvas.reshape(-1)


def unflatten_time_image(
    flat_tokens: np.ndarray,
    image_height: int,
    num_frames: int,
    token_offset: int,
    padded_height: Optional[int] = None,
    padded_num_frames: Optional[int] = None,
) -> np.ndarray:
    padded_height = image_height if padded_height is None else padded_height
    padded_num_frames = num_frames if padded_num_frames is None else padded_num_frames
    canvas = flat_tokens.reshape(padded_height, padded_num_frames)
    return canvas[:image_height, :num_frames] - token_offset


def _passes_gzip_filter(score: float, config: NCA1DDataConfig) -> bool:
    if config.gzip_threshold_low is not None and score < config.gzip_threshold_low:
        return False
    if config.gzip_threshold_high is not None and score >= config.gzip_threshold_high:
        return False
    return True


def _make_split_arrays(
    flat_images: list[np.ndarray],
    gzip_scores: list[float],
    state_heights: list[int],
    num_frames: list[int],
    rollout_steps: list[int],
    config: NCA1DDataConfig,
) -> Dict[str, np.ndarray]:
    num_examples = len(flat_images)

    inputs = np.stack(flat_images, axis=0).astype(config.save_dtype, copy=False)
    labels = inputs.copy()

    results: Dict[str, np.ndarray] = {
        "inputs": inputs,
        "labels": labels,
        "puzzle_identifiers": np.zeros((num_examples,), dtype=np.int32),
        "puzzle_indices": np.arange(num_examples + 1, dtype=np.int32),
        "group_indices": np.arange(num_examples + 1, dtype=np.int32),
        "gzip_ratio": np.asarray(gzip_scores, dtype=np.float32),
        "state_heights": np.asarray(state_heights, dtype=np.int32),
        "num_frames": np.asarray(num_frames, dtype=np.int32),
        "rollout_steps": np.asarray(rollout_steps, dtype=np.int32),
    }
    return results


def generate_split(split_name: str, size: int, seed: int, config: NCA1DDataConfig) -> Dict[str, np.ndarray]:
    seed_everything(seed)
    device = get_device()
    rng = np.random.default_rng(seed)

    flat_images: list[np.ndarray] = []
    gzip_scores: list[float] = []
    state_heights: list[int] = []
    num_frames: list[int] = []
    rollout_steps: list[int] = []

    min_rounds_for_size = math.ceil(size / config.batch_candidate_size)
    effective_max_sampling_rounds = max(config.max_sampling_rounds, min_rounds_for_size)
    total_candidates = config.batch_candidate_size * effective_max_sampling_rounds
    progress = tqdm(total=size, desc=f"Generating {split_name}", leave=False)

    rounds = 0
    while len(flat_images) < size and rounds < effective_max_sampling_rounds:
        rounds += 1
        sampled_state_heights = rng.choice(config.valid_state_heights, size=config.batch_candidate_size)
        sampled_rollout_steps = rng.integers(
            config.sampled_rollout_steps_min,
            config.sampled_rollout_steps_max + 1,
            size=config.batch_candidate_size,
        )

        candidate_trajectories: list[Optional[np.ndarray]] = [None] * config.batch_candidate_size
        candidate_tokenizers: list[Optional[NCATokenizer]] = [None] * config.batch_candidate_size

        for sampled_state_height in np.unique(sampled_state_heights):
            group_indices = np.flatnonzero(sampled_state_heights == sampled_state_height)
            max_rollout_steps = int(sampled_rollout_steps[group_indices].max())
            nca_config = make_nca_config_for_sample(
                config,
                state_height=int(sampled_state_height),
                rollout_steps=max_rollout_steps,
            )
            tokenizer = NCATokenizer(nca_config)
            trajectories = rollout_trajectories_batched(
                nca_config,
                batch_size=int(group_indices.size),
                device=device,
            )

            for local_idx, candidate_idx in enumerate(group_indices):
                candidate_trajectories[int(candidate_idx)] = trajectories[local_idx]
                candidate_tokenizers[int(candidate_idx)] = tokenizer

        for candidate_idx in range(config.batch_candidate_size):
            sampled_state_height = int(sampled_state_heights[candidate_idx])
            sampled_rollout_step_count = int(sampled_rollout_steps[candidate_idx])
            sampled_num_frames = num_frames_for_rollout_steps(
                sampled_rollout_step_count,
                config.time_subsample,
            )
            trajectory = candidate_trajectories[candidate_idx]
            tokenizer = candidate_tokenizers[candidate_idx]

            if trajectory is None or tokenizer is None:
                raise RuntimeError("Batched candidate generation failed to produce a trajectory.")

            trajectory = trajectory[:sampled_num_frames]
            gzip_score = score_trajectory_gzip(trajectory, tokenizer)

            if not _passes_gzip_filter(gzip_score, config):
                continue

            time_image = trajectory_to_time_image(trajectory)
            flat_image = flatten_time_image(
                time_image,
                config.token_offset,
                padded_height=config.final_image_shape[0],
                padded_num_frames=config.final_image_shape[1],
            )

            flat_images.append(flat_image)
            gzip_scores.append(gzip_score)
            state_heights.append(sampled_state_height)
            num_frames.append(int(time_image.shape[1]))
            rollout_steps.append(sampled_rollout_step_count)
            progress.update(1)

            if len(flat_images) >= size:
                break

    progress.close()

    if len(flat_images) < size:
        raise RuntimeError(
            f"Could not collect enough {split_name} examples. "
            f"Collected {len(flat_images)} / {size} after {rounds} rounds ({total_candidates} candidates budget). "
            "Try increasing max_sampling_rounds or batch_candidate_size, or relax the gzip thresholds."
        )

    return _make_split_arrays(
        flat_images,
        gzip_scores,
        state_heights,
        num_frames,
        rollout_steps,
        config,
    )


def save_split(split_name: str, split_arrays: Dict[str, np.ndarray], config: NCA1DDataConfig) -> None:
    split_dir = Path(config.output_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

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
    )

    with open(split_dir / "dataset.json", "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    np.save(split_dir / "all__inputs.npy", split_arrays["inputs"])
    np.save(split_dir / "all__labels.npy", split_arrays["labels"])
    np.save(split_dir / "all__puzzle_identifiers.npy", split_arrays["puzzle_identifiers"])
    np.save(split_dir / "all__puzzle_indices.npy", split_arrays["puzzle_indices"])
    np.save(split_dir / "all__group_indices.npy", split_arrays["group_indices"])
    np.save(split_dir / "all__gzip_ratio.npy", split_arrays["gzip_ratio"])
    np.save(split_dir / "all__state_heights.npy", split_arrays["state_heights"])
    np.save(split_dir / "all__num_frames.npy", split_arrays["num_frames"])
    np.save(split_dir / "all__rollout_steps.npy", split_arrays["rollout_steps"])

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
                "resolved_time_end": config.resolved_time_end,
                "num_frames": config.max_num_frames,
                "max_num_frames": config.max_num_frames,
                "seq_len": config.seq_len,
                "vocab_size": config.vocab_size,
                "final_image_shape": list(config.final_image_shape),
                "sample_random_state_height": config.sampled_state_height_min != config.sampled_state_height_max,
                "sample_random_rollout_steps": config.sampled_rollout_steps_min != config.sampled_rollout_steps_max,
                "time_axis": "width",
                "value_encoding": {
                    "pad": 0,
                    "mask": 1,
                    "nca_color_range": [config.token_offset, config.token_offset + config.num_colors - 1],
                },
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
