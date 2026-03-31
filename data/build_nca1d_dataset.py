from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
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
    RandomDiscreteNCA,
    get_device,
    rollout_trajectory,
    score_trajectory_gzip,
    seed_everything,
)


cli = ArgParser()


class NCA1DDataConfig(BaseModel):
    output_dir: str = "data/nca1d-default"

    train_size: int = 1024
    test_size: int = 256
    seed: int = 0

    state_height: int = 16
    state_width: int = 1

    num_colors: int = 8
    temperature: float = 1e-3
    identity_bias: float = 0.0
    conv_channels: int = 4
    hidden_dim: int = 16

    # Use patch_size=1 so 1-column NCA states remain tokenizable for gzip filtering.
    patch_size: int = 1

    rollout_steps: int = 64
    time_subsample: int = 1
    start_step: int = 0

    gzip_threshold_low: Optional[float] = None
    gzip_threshold_high: Optional[float] = None

    batch_candidate_size: int = 64
    max_sampling_rounds: int = 200

    # Reserve 0 for PAD and 1 for the default mask token used by PuzzleDataset.
    token_offset: int = 2
    save_dtype: str = "int32"

    @model_validator(mode="after")
    def _validate(self) -> "NCA1DDataConfig":
        if self.train_size <= 0:
            raise ValueError(f"train_size must be > 0, got {self.train_size}")
        if self.test_size <= 0:
            raise ValueError(f"test_size must be > 0, got {self.test_size}")
        if self.state_height <= 0:
            raise ValueError(f"state_height must be > 0, got {self.state_height}")
        if self.state_width != 1:
            raise ValueError(
                f"state_width must be 1 for the requested 1xH -> HxT dataset, got {self.state_width}"
            )
        if self.num_colors <= 1:
            raise ValueError(f"num_colors must be > 1, got {self.num_colors}")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {self.patch_size}")
        if self.state_height % self.patch_size != 0 or self.state_width % self.patch_size != 0:
            raise ValueError(
                "state_height and state_width must both be divisible by patch_size, "
                f"got {(self.state_height, self.state_width)} and patch_size={self.patch_size}"
            )
        if self.rollout_steps <= 0:
            raise ValueError(f"rollout_steps must be > 0, got {self.rollout_steps}")
        if self.time_subsample <= 0:
            raise ValueError(f"time_subsample must be > 0, got {self.time_subsample}")
        if self.start_step < 0:
            raise ValueError(f"start_step must be >= 0, got {self.start_step}")
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
    def num_frames(self) -> int:
        return math.ceil(self.rollout_steps / self.time_subsample)

    @property
    def final_image_shape(self) -> tuple[int, int]:
        return self.state_height, self.num_frames

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
        rollout_steps=config.rollout_steps,
        time_subsample=config.time_subsample,
        start_step=config.start_step,
        gzip_threshold_low=config.gzip_threshold_low or 0.0,
        gzip_threshold_high=config.gzip_threshold_high,
        batch_candidate_size=config.batch_candidate_size,
        max_sampling_rounds=config.max_sampling_rounds,
        train_size=config.train_size,
        val_size=config.test_size,
        out_dir=config.output_dir,
        save_dtype=config.save_dtype,
    )


def trajectory_to_time_image(trajectory: np.ndarray) -> np.ndarray:
    if trajectory.ndim != 3:
        raise ValueError(f"trajectory must have shape [T, H, W], got ndim={trajectory.ndim}")
    if trajectory.shape[2] != 1:
        raise ValueError(
            f"trajectory width must be 1 for 1xH -> HxT conversion, got shape={trajectory.shape}"
        )
    return np.squeeze(trajectory, axis=2).T.astype(np.int16, copy=False)


def flatten_time_image(time_image: np.ndarray, token_offset: int) -> np.ndarray:
    if time_image.ndim != 2:
        raise ValueError(f"time_image must have shape [H, T], got ndim={time_image.ndim}")
    return time_image.astype(np.int32, copy=False).reshape(-1) + token_offset


def unflatten_time_image(
    flat_tokens: np.ndarray,
    image_height: int,
    num_frames: int,
    token_offset: int,
) -> np.ndarray:
    return flat_tokens.reshape(image_height, num_frames) - token_offset


def _passes_gzip_filter(score: float, config: NCA1DDataConfig) -> bool:
    if config.gzip_threshold_low is not None and score < config.gzip_threshold_low:
        return False
    if config.gzip_threshold_high is not None and score >= config.gzip_threshold_high:
        return False
    return True


def _make_split_arrays(
    flat_images: list[np.ndarray],
    gzip_scores: list[float],
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
    }
    return results


def generate_split(split_name: str, size: int, seed: int, config: NCA1DDataConfig) -> Dict[str, np.ndarray]:
    seed_everything(seed)
    device = get_device()
    nca_config = make_nca_config(config)
    tokenizer = NCATokenizer(nca_config)

    flat_images: list[np.ndarray] = []
    gzip_scores: list[float] = []

    total_candidates = config.batch_candidate_size * config.max_sampling_rounds
    progress = tqdm(total=size, desc=f"Generating {split_name}", leave=False)

    rounds = 0
    while len(flat_images) < size and rounds < config.max_sampling_rounds:
        rounds += 1
        for _candidate_idx in range(config.batch_candidate_size):
            rule = RandomDiscreteNCA(nca_config).to(device)
            trajectory = rollout_trajectory(rule, nca_config, device=device)
            gzip_score = score_trajectory_gzip(trajectory, tokenizer)

            if not _passes_gzip_filter(gzip_score, config):
                continue

            time_image = trajectory_to_time_image(trajectory)
            flat_image = flatten_time_image(time_image, config.token_offset)

            flat_images.append(flat_image)
            gzip_scores.append(gzip_score)
            progress.update(1)

            if len(flat_images) >= size:
                break

    progress.close()

    if len(flat_images) < size:
        raise RuntimeError(
            f"Could not collect enough {split_name} examples. "
            f"Collected {len(flat_images)} / {size} after {total_candidates} candidates. "
            "Try increasing max_sampling_rounds or batch_candidate_size, or relax the gzip thresholds."
        )

    return _make_split_arrays(flat_images, gzip_scores, config)


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
                "num_frames": config.num_frames,
                "seq_len": config.seq_len,
                "vocab_size": config.vocab_size,
                "final_image_shape": list(config.final_image_shape),
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
