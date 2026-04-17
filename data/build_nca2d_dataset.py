from __future__ import annotations

import json
import math
import os
import sys
import time
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


class NCA2DDataConfig(BaseModel):
    output_dir: str = "data/nca2d-data"

    train_size: int = 10000
    test_size: int = 1000
    seed: int = 0

    state_height: int = 16
    state_height_min: Optional[int] = 8
    state_height_max: Optional[int] = 24
    state_width: int = 16
    state_width_min: Optional[int] = 8
    state_width_max: Optional[int] = 24

    num_colors: int = 10
    out_colors: Optional[int] = 10
    temperature: float = 1e-12
    identity_bias: float = 0.0
    conv_channels: int = 4
    hidden_dim: int = 16

    patch_size: int = 1

    # The answer is the NCA state after `answer_steps` raw updates.
    answer_steps: int = 1
    answer_steps_min: Optional[int] = None
    answer_steps_max: Optional[int] = None

    # Number of in-context example pairs. The input channel count is 2 * counts + 1.
    counts: int = 2
    counts_min: Optional[int] = 1
    counts_max: Optional[int] = 4

    start_step: int = 0
    time_start: Optional[int] = 32
    # Number of skipped raw NCA steps after a question/answer pair before the next question.
    time_span: int = 0

    gzip_threshold_low: Optional[float] = None
    gzip_threshold_high: Optional[float] = None

    batch_candidate_size: int = 2048
    max_sampling_rounds: int = 2000
    # Approximate upper bound used to adapt the per-group batch size:
    # batch_size * H * W * (2 * (counts + 1)).
    max_cells_per_candidate_batch: int = 8_000_000
    max_data_seq_len: int = 65536

    token_offset: int = 2
    mask_token_id: int = 1
    save_dtype: str = "int32"

    @property
    def sampled_state_height_min(self) -> int:
        return self.state_height if self.state_height_min is None else self.state_height_min

    @property
    def sampled_state_height_max(self) -> int:
        return self.state_height if self.state_height_max is None else self.state_height_max

    @property
    def sampled_state_width_min(self) -> int:
        return self.state_width if self.state_width_min is None else self.state_width_min

    @property
    def sampled_state_width_max(self) -> int:
        return self.state_width if self.state_width_max is None else self.state_width_max

    @property
    def sampled_answer_steps_min(self) -> int:
        return self.answer_steps if self.answer_steps_min is None else self.answer_steps_min

    @property
    def sampled_answer_steps_max(self) -> int:
        return self.answer_steps if self.answer_steps_max is None else self.answer_steps_max

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
    def valid_state_widths(self) -> list[int]:
        return [
            width
            for width in range(self.sampled_state_width_min, self.sampled_state_width_max + 1)
            if width % self.patch_size == 0
        ]

    @property
    def max_counts_allowed_by_canvas(self) -> int:
        max_tokens_per_plane = self.sampled_state_height_max * self.sampled_state_width_max
        if max_tokens_per_plane <= 0:
            return 0
        return max((self.max_data_seq_len // max_tokens_per_plane - 1) // 2, 0)

    @property
    def resolved_counts_max(self) -> int:
        return min(self.sampled_counts_max, self.max_counts_allowed_by_canvas)

    @property
    def max_input_channels(self) -> int:
        return 2 * self.resolved_counts_max + 1

    @property
    def final_image_shape(self) -> tuple[int, int, int]:
        return (
            self.sampled_state_height_max,
            self.sampled_state_width_max,
            self.max_input_channels,
        )

    @property
    def seq_len(self) -> int:
        height, width, channels = self.final_image_shape
        return height * width * channels

    @property
    def position_id_shape(self) -> tuple[int, int, int]:
        height, width, channels = self.final_image_shape
        return channels, height, width

    @property
    def vocab_size(self) -> int:
        return self.resolved_out_colors + self.token_offset

    @property
    def resolved_out_colors(self) -> int:
        return self.num_colors if self.out_colors is None else self.out_colors

    @property
    def gzip_enabled(self) -> bool:
        return self.gzip_threshold_low is not None or self.gzip_threshold_high is not None

    @staticmethod
    def input_channels_for_counts(counts: int) -> int:
        if counts <= 0:
            raise ValueError(f"counts must be > 0, got {counts}")
        return 2 * counts + 1

    @staticmethod
    def query_channel_index_for_counts(counts: int) -> int:
        return NCA2DDataConfig.input_channels_for_counts(counts) - 1

    @model_validator(mode="after")
    def _validate(self) -> "NCA2DDataConfig":
        if self.train_size <= 0:
            raise ValueError(f"train_size must be > 0, got {self.train_size}")
        if self.test_size <= 0:
            raise ValueError(f"test_size must be > 0, got {self.test_size}")
        if self.state_height <= 0:
            raise ValueError(f"state_height must be > 0, got {self.state_height}")
        if self.state_width <= 0:
            raise ValueError(f"state_width must be > 0, got {self.state_width}")
        if self.sampled_state_height_min <= 0:
            raise ValueError(
                f"state_height_min/state_height must be > 0, got {self.sampled_state_height_min}"
            )
        if self.sampled_state_width_min <= 0:
            raise ValueError(
                f"state_width_min/state_width must be > 0, got {self.sampled_state_width_min}"
            )
        if self.sampled_state_height_min > self.sampled_state_height_max:
            raise ValueError(
                "state_height_min must be <= state_height_max, "
                f"got min={self.sampled_state_height_min}, max={self.sampled_state_height_max}"
            )
        if self.sampled_state_width_min > self.sampled_state_width_max:
            raise ValueError(
                "state_width_min must be <= state_width_max, "
                f"got min={self.sampled_state_width_min}, max={self.sampled_state_width_max}"
            )
        if self.num_colors <= 1:
            raise ValueError(f"num_colors must be > 1, got {self.num_colors}")
        if self.resolved_out_colors <= 1:
            raise ValueError(f"out_colors/num_colors must be > 1, got {self.resolved_out_colors}")
        if self.resolved_out_colors < self.num_colors:
            raise ValueError(
                "out_colors must be >= num_colors so colors can be remapped without duplication, "
                f"got num_colors={self.num_colors}, out_colors={self.resolved_out_colors}"
            )
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {self.patch_size}")
        if not self.valid_state_heights:
            raise ValueError(
                "No valid state heights in the requested range are divisible by patch_size, "
                f"got range=({self.sampled_state_height_min}, {self.sampled_state_height_max}) "
                f"and patch_size={self.patch_size}"
            )
        if not self.valid_state_widths:
            raise ValueError(
                "No valid state widths in the requested range are divisible by patch_size, "
                f"got range=({self.sampled_state_width_min}, {self.sampled_state_width_max}) "
                f"and patch_size={self.patch_size}"
            )
        if self.answer_steps <= 0:
            raise ValueError(f"answer_steps must be > 0, got {self.answer_steps}")
        if self.sampled_answer_steps_min <= 0:
            raise ValueError(
                "answer_steps_min/answer_steps must be > 0, "
                f"got {self.sampled_answer_steps_min}"
            )
        if self.sampled_answer_steps_min > self.sampled_answer_steps_max:
            raise ValueError(
                "answer_steps_min must be <= answer_steps_max, "
                f"got min={self.sampled_answer_steps_min}, max={self.sampled_answer_steps_max}"
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
        if self.start_step < 0:
            raise ValueError(f"start_step must be >= 0, got {self.start_step}")
        if self.time_start is not None and self.time_start < 0:
            raise ValueError(f"time_start must be >= 0, got {self.time_start}")
        if self.time_span < 0:
            raise ValueError(f"time_span must be >= 0, got {self.time_span}")
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
        if self.batch_candidate_size <= 0:
            raise ValueError(f"batch_candidate_size must be > 0, got {self.batch_candidate_size}")
        if self.max_sampling_rounds <= 0:
            raise ValueError(f"max_sampling_rounds must be > 0, got {self.max_sampling_rounds}")
        if self.max_cells_per_candidate_batch <= 0:
            raise ValueError(
                "max_cells_per_candidate_batch must be > 0, "
                f"got {self.max_cells_per_candidate_batch}"
            )
        if self.max_data_seq_len <= 0:
            raise ValueError(f"max_data_seq_len must be > 0, got {self.max_data_seq_len}")
        if self.token_offset < 2:
            raise ValueError(f"token_offset must be >= 2, got {self.token_offset}")
        if not (0 < self.mask_token_id < self.token_offset):
            raise ValueError(
                "mask_token_id must be in the reserved token range [1, token_offset), "
                f"got mask_token_id={self.mask_token_id}, token_offset={self.token_offset}"
            )
        if self.max_counts_allowed_by_canvas < 1:
            raise ValueError(
                "The maximum HxW canvas already exceeds the max_data_seq_len cap for a single "
                f"example count: state_height_max={self.sampled_state_height_max}, "
                f"state_width_max={self.sampled_state_width_max}, max_data_seq_len={self.max_data_seq_len}"
            )
        return self


def make_nca_config_for_sample(
    config: NCA2DDataConfig,
    state_height: int,
    state_width: int,
    answer_steps: int,
) -> NCAConfig:
    return NCAConfig(
        grid_height=state_height,
        grid_width=state_width,
        num_colors=config.num_colors,
        temperature=config.temperature,
        identity_bias=config.identity_bias,
        conv_channels=config.conv_channels,
        hidden_dim=config.hidden_dim,
        patch_size=config.patch_size,
        seq_len=config.seq_len,
        rollout_steps=answer_steps + 1,
        time_subsample=1,
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
        if t >= cfg.start_step:
            frames.append(state)
        state = _batched_step(state, rule_parameters, cfg)

    trajectories = torch.stack(frames, dim=1)
    return trajectories.detach().cpu().numpy().astype(np.int16, copy=False)


def pair_window_stride(answer_steps: int, time_span: int) -> int:
    if answer_steps <= 0:
        raise ValueError(f"answer_steps must be > 0, got {answer_steps}")
    if time_span < 0:
        raise ValueError(f"time_span must be >= 0, got {time_span}")
    return answer_steps + 1 + time_span


@torch.no_grad()
def rollout_pair_windows_batched(
    cfg: NCAConfig,
    sample_batch_size: int,
    example_count: int,
    answer_steps: int,
    device: torch.device,
    time_span: int = 0,
) -> torch.Tensor:
    if sample_batch_size <= 0:
        raise ValueError(f"sample_batch_size must be > 0, got {sample_batch_size}")
    if example_count <= 0:
        raise ValueError(f"example_count must be > 0, got {example_count}")
    if answer_steps <= 0:
        raise ValueError(f"answer_steps must be > 0, got {answer_steps}")
    if time_span < 0:
        raise ValueError(f"time_span must be >= 0, got {time_span}")

    total_windows = example_count + 1
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
    window_stride = pair_window_stride(answer_steps, time_span)
    total_steps = cfg.start_step + (total_windows - 1) * window_stride + answer_steps + 1

    for t in range(total_steps):
        relative_t = t - cfg.start_step
        if relative_t >= 0:
            window_idx = relative_t // window_stride
            offset_within_window = relative_t % window_stride
            if window_idx < total_windows and (
                offset_within_window == 0 or offset_within_window == answer_steps
            ):
                frames.append(state)
        state = _batched_step(state, rule_parameters, cfg)

    expected_frame_count = total_windows * 2
    if len(frames) != expected_frame_count:
        raise RuntimeError(
            "Collected an unexpected number of pair-window frames, "
            f"got {len(frames)} but expected {expected_frame_count} "
            f"for example_count={example_count}, answer_steps={answer_steps}, "
            f"time_span={time_span}"
        )

    return torch.stack(frames, dim=1).reshape(
        sample_batch_size,
        total_windows,
        2,
        cfg.grid_height,
        cfg.grid_width,
    )


def pair_windows_to_input_label_grids(pair_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pair_windows.ndim != 4:
        raise ValueError(
            "pair_windows must have shape [Wn, 2, H, W], "
            f"got ndim={pair_windows.ndim}"
        )
    if pair_windows.shape[1] != 2:
        raise ValueError(
            "pair_windows second axis must have size 2 for (question, answer), "
            f"got shape={pair_windows.shape}"
        )

    example_count = pair_windows.shape[0] - 1
    if example_count <= 0:
        raise ValueError(
            "pair_windows must contain at least one example pair and one query pair, "
            f"got shape={pair_windows.shape}"
        )

    example_channels = pair_windows[:example_count].transpose(2, 3, 0, 1).reshape(
        pair_windows.shape[2],
        pair_windows.shape[3],
        2 * example_count,
    )
    query_input = pair_windows[example_count, 0][:, :, None]
    target_label = pair_windows[example_count, 1][:, :, None]
    return (
        np.concatenate([example_channels, query_input], axis=2).astype(np.int16, copy=False),
        target_label.astype(np.int16, copy=False),
    )


def pair_windows_to_input_label_grids_batched(
    pair_windows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pair_windows.ndim != 5:
        raise ValueError(
            "pair_windows must have shape [B, Wn, 2, H, W], "
            f"got ndim={pair_windows.ndim}"
        )
    if pair_windows.shape[2] != 2:
        raise ValueError(
            "pair_windows third axis must have size 2 for (question, answer), "
            f"got shape={tuple(pair_windows.shape)}"
        )

    example_count = pair_windows.shape[1] - 1
    if example_count <= 0:
        raise ValueError(
            "pair_windows must contain at least one example pair and one query pair, "
            f"got shape={tuple(pair_windows.shape)}"
        )

    example_channels = (
        pair_windows[:, :example_count]
        .permute(0, 3, 4, 1, 2)
        .contiguous()
        .reshape(pair_windows.shape[0], pair_windows.shape[3], pair_windows.shape[4], 2 * example_count)
    )
    query_input = pair_windows[:, example_count, 0].unsqueeze(-1)
    target_label = pair_windows[:, example_count, 1].unsqueeze(-1)
    return torch.cat([example_channels, query_input], dim=-1), target_label


def sample_output_color_maps_batched(
    *,
    batch_size: int,
    num_colors: int,
    out_colors: int,
    device: torch.device,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if num_colors <= 0:
        raise ValueError(f"num_colors must be > 0, got {num_colors}")
    if out_colors < num_colors:
        raise ValueError(
            f"out_colors must be >= num_colors, got num_colors={num_colors}, out_colors={out_colors}"
        )

    if out_colors == num_colors:
        return torch.arange(num_colors, device=device, dtype=torch.long).expand(batch_size, -1)

    weights = torch.ones((batch_size, out_colors), dtype=torch.float32, device=device)
    return torch.multinomial(weights, num_samples=num_colors, replacement=False)


def remap_colors_per_sample_batched(
    input_grids: torch.Tensor,
    label_grids: torch.Tensor,
    *,
    config: NCA2DDataConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_grids.ndim != 4:
        raise ValueError(
            f"input_grids must have shape [B, H, W, K], got ndim={input_grids.ndim}"
        )
    if label_grids.ndim != 4:
        raise ValueError(
            f"label_grids must have shape [B, H, W, 1], got ndim={label_grids.ndim}"
        )

    if config.resolved_out_colors == config.num_colors:
        return input_grids, label_grids

    batch_size = input_grids.shape[0]
    device = input_grids.device
    color_maps = sample_output_color_maps_batched(
        batch_size=batch_size,
        num_colors=config.num_colors,
        out_colors=config.resolved_out_colors,
        device=device,
    )

    flat_input = input_grids.to(torch.long).reshape(batch_size, -1)
    flat_label = label_grids.to(torch.long).reshape(batch_size, -1)
    remapped_input = torch.gather(color_maps, 1, flat_input).reshape_as(input_grids)
    remapped_label = torch.gather(color_maps, 1, flat_label).reshape_as(label_grids)
    return remapped_input.to(input_grids.dtype), remapped_label.to(label_grids.dtype)


def flatten_input_grid(
    input_grid: np.ndarray,
    token_offset: int,
    padded_height: Optional[int] = None,
    padded_width: Optional[int] = None,
    padded_channels: Optional[int] = None,
) -> np.ndarray:
    if input_grid.ndim != 3:
        raise ValueError(f"input_grid must have shape [H, W, K], got ndim={input_grid.ndim}")
    image_height, image_width, input_channels = input_grid.shape
    padded_height = image_height if padded_height is None else padded_height
    padded_width = image_width if padded_width is None else padded_width
    padded_channels = input_channels if padded_channels is None else padded_channels

    if (
        image_height > padded_height
        or image_width > padded_width
        or input_channels > padded_channels
    ):
        raise ValueError(
            "input_grid must fit inside the padded canvas, "
            f"got image_shape={(image_height, image_width, input_channels)} and "
            f"padded_shape={(padded_height, padded_width, padded_channels)}"
        )

    canvas = np.zeros((padded_height, padded_width, padded_channels), dtype=np.int32)
    canvas[:image_height, :image_width, :input_channels] = input_grid.astype(np.int32, copy=False) + token_offset
    return canvas.reshape(-1)


def unflatten_input_grid(
    flat_tokens: np.ndarray,
    image_height: int,
    image_width: int,
    counts: int,
    token_offset: int,
    padded_height: Optional[int] = None,
    padded_width: Optional[int] = None,
    padded_channels: Optional[int] = None,
) -> np.ndarray:
    input_channels = NCA2DDataConfig.input_channels_for_counts(counts)
    padded_height = image_height if padded_height is None else padded_height
    padded_width = image_width if padded_width is None else padded_width
    padded_channels = input_channels if padded_channels is None else padded_channels
    canvas = flat_tokens.reshape(padded_height, padded_width, padded_channels)
    return canvas[:image_height, :image_width, :input_channels] - token_offset


def extract_label_grid(
    flat_labels: np.ndarray,
    image_height: int,
    image_width: int,
    counts: int,
    token_offset: int,
    padded_height: Optional[int] = None,
    padded_width: Optional[int] = None,
    padded_channels: Optional[int] = None,
) -> np.ndarray:
    query_channel_idx = NCA2DDataConfig.query_channel_index_for_counts(counts)
    padded_height = image_height if padded_height is None else padded_height
    padded_width = image_width if padded_width is None else padded_width
    padded_channels = (
        NCA2DDataConfig.input_channels_for_counts(counts)
        if padded_channels is None
        else padded_channels
    )
    canvas = flat_labels.reshape(padded_height, padded_width, padded_channels)
    return canvas[:image_height, :image_width, query_channel_idx : query_channel_idx + 1] - token_offset


def score_pair_window_set_gzip(pair_windows: np.ndarray, tokenizer: NCATokenizer) -> float:
    if pair_windows.ndim != 4:
        raise ValueError(
            f"pair_windows must have shape [Wn, 2, H, W], got ndim={pair_windows.ndim}"
        )

    frame_tokens = []
    for window in pair_windows:
        frame_tokens.extend(tokenizer.encode_frame(frame)[1:-1] for frame in window)
    flat = np.concatenate(frame_tokens, axis=0)
    return gzip_complexity_ratio_from_tokens(flat)


def _pair_window_tokens_for_gzip_patch1(pair_window_sets: np.ndarray) -> list[np.ndarray]:
    if pair_window_sets.ndim != 5:
        raise ValueError(
            "pair_window_sets must have shape [B, Wn, 2, H, W], "
            f"got ndim={pair_window_sets.ndim}"
        )
    return [
        pair_window_sets[batch_idx].reshape(-1).astype(np.int32, copy=False)
        for batch_idx in range(pair_window_sets.shape[0])
    ]


def score_pair_window_sets_gzip_batched(
    pair_window_sets: np.ndarray,
    tokenizer: NCATokenizer,
) -> np.ndarray:
    if tokenizer.patch != 1:
        scores = [
            score_pair_window_set_gzip(pair_window_sets[idx], tokenizer)
            for idx in range(pair_window_sets.shape[0])
        ]
        return np.asarray(scores, dtype=np.float32)

    token_sets = _pair_window_tokens_for_gzip_patch1(pair_window_sets)
    max_workers = min(len(token_sets), max(os.cpu_count() or 1, 1))
    if max_workers <= 1:
        scores = [gzip_complexity_ratio_from_tokens(tokens) for tokens in token_sets]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            scores = list(executor.map(gzip_complexity_ratio_from_tokens, token_sets))
    return np.asarray(scores, dtype=np.float32)


def build_sample_arrays_batched(
    input_grids: torch.Tensor,
    label_grids: torch.Tensor,
    *,
    counts: int,
    config: NCA2DDataConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if input_grids.ndim != 4:
        raise ValueError(
            f"input_grids must have shape [B, H, W, K], got ndim={input_grids.ndim}"
        )
    if label_grids.ndim != 4:
        raise ValueError(
            f"label_grids must have shape [B, H, W, 1], got ndim={label_grids.ndim}"
        )
    if label_grids.shape[-1] != 1:
        raise ValueError(
            f"label_grids last axis must be 1, got shape={tuple(label_grids.shape)}"
        )

    batch_size, image_height, image_width, input_channels = input_grids.shape
    padded_height, padded_width, padded_channels = config.final_image_shape
    if (
        image_height > padded_height
        or image_width > padded_width
        or input_channels > padded_channels
    ):
        raise ValueError(
            "input_grids must fit inside the fixed dataset canvas, "
            f"got image_shape={(image_height, image_width, input_channels)} and "
            f"padded_shape={config.final_image_shape}"
        )

    device = input_grids.device
    dtype = torch.int32
    query_channel_idx = config.query_channel_index_for_counts(counts)

    encoded_inputs = input_grids.to(dtype=dtype) + config.token_offset
    encoded_labels = label_grids.to(dtype=dtype) + config.token_offset

    input_canvas = torch.zeros(
        (batch_size, padded_height, padded_width, padded_channels),
        dtype=dtype,
        device=device,
    )
    input_canvas[:, :image_height, :image_width, :input_channels] = encoded_inputs

    label_canvas = torch.zeros_like(input_canvas)
    label_canvas[:, :image_height, :image_width, query_channel_idx : query_channel_idx + 1] = encoded_labels

    position_ids = torch.zeros(
        (batch_size, padded_height, padded_width, padded_channels, 3),
        dtype=dtype,
        device=device,
    )
    channel_grid = torch.arange(input_channels, dtype=dtype, device=device).view(1, 1, 1, input_channels)
    row_grid = torch.arange(image_height, dtype=dtype, device=device).view(1, image_height, 1, 1)
    col_grid = torch.arange(image_width, dtype=dtype, device=device).view(1, 1, image_width, 1)
    position_region = torch.stack(
        [
            channel_grid.expand(batch_size, image_height, image_width, input_channels),
            row_grid.expand(batch_size, image_height, image_width, input_channels),
            col_grid.expand(batch_size, image_height, image_width, input_channels),
        ],
        dim=-1,
    )
    position_ids[:, :image_height, :image_width, :input_channels] = position_region

    return (
        input_canvas.reshape(batch_size, -1).cpu().numpy(),
        label_canvas.reshape(batch_size, -1).cpu().numpy(),
        position_ids.reshape(batch_size, -1, 3).cpu().numpy(),
    )


def _passes_gzip_filter(score: float, config: NCA2DDataConfig) -> bool:
    if config.gzip_threshold_low is not None and score < config.gzip_threshold_low:
        return False
    if config.gzip_threshold_high is not None and score >= config.gzip_threshold_high:
        return False
    return True


def _shape_group_specs(config: NCA2DDataConfig) -> list[tuple[int, int, int, int]]:
    return [
        (state_height, state_width, answer_steps, counts)
        for state_height in config.valid_state_heights
        for state_width in config.valid_state_widths
        for answer_steps in range(
            config.sampled_answer_steps_min,
            config.sampled_answer_steps_max + 1,
        )
        for counts in range(
            config.sampled_counts_min,
            config.resolved_counts_max + 1,
        )
    ]


def _allocate_examples_per_shape_group(
    size: int,
    shape_groups: list[tuple[int, int, int, int]],
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
    state_widths: list[int],
    answer_steps: list[int],
    counts: list[int],
    config: NCA2DDataConfig,
) -> Dict[str, np.ndarray]:
    num_examples = len(flat_inputs)

    inputs = np.stack(flat_inputs, axis=0).astype(config.save_dtype, copy=False)
    labels = np.stack(flat_labels, axis=0).astype(config.save_dtype, copy=False)

    return {
        "inputs": inputs,
        "labels": labels,
        "position_ids": np.stack(position_ids, axis=0).astype(np.int32, copy=False),
        "puzzle_identifiers": np.zeros((num_examples,), dtype=np.int32),
        "puzzle_indices": np.arange(num_examples + 1, dtype=np.int32),
        "group_indices": np.arange(num_examples + 1, dtype=np.int32),
        "gzip_ratio": np.asarray(gzip_scores, dtype=np.float32),
        "state_heights": np.asarray(state_heights, dtype=np.int32),
        "state_widths": np.asarray(state_widths, dtype=np.int32),
        "answer_steps": np.asarray(answer_steps, dtype=np.int32),
        "counts": np.asarray(counts, dtype=np.int32),
        "input_channels": np.asarray(
            [config.input_channels_for_counts(value) for value in counts],
            dtype=np.int32,
        ),
        "query_channel_indices": np.asarray(
            [config.query_channel_index_for_counts(value) for value in counts],
            dtype=np.int32,
        ),
    }


def _candidate_batch_size_for_group(
    config: NCA2DDataConfig,
    *,
    state_height: int,
    state_width: int,
    counts: int,
    target_group_size: int,
) -> int:
    per_sample_cells = state_height * state_width * max(2 * (counts + 1), 1)
    adaptive_cap = max(1, config.max_cells_per_candidate_batch // max(per_sample_cells, 1))
    return max(
        1,
        min(
            config.batch_candidate_size,
            max(target_group_size, 64),
            adaptive_cap,
        ),
    )


def _summarize_gzip(values: np.ndarray) -> dict[str, Optional[float]]:
    valid = values[values >= 0.0]
    if valid.size == 0:
        return {
            "gzip_ratio_mean": None,
            "gzip_ratio_std": None,
            "gzip_ratio_min": None,
            "gzip_ratio_max": None,
        }
    return {
        "gzip_ratio_mean": float(valid.mean()),
        "gzip_ratio_std": float(valid.std()),
        "gzip_ratio_min": float(valid.min()),
        "gzip_ratio_max": float(valid.max()),
    }


def generate_split(split_name: str, size: int, seed: int, config: NCA2DDataConfig) -> Dict[str, np.ndarray]:
    seed_everything(seed)
    device = get_device()
    _configure_generation_backend(device)
    rng = np.random.default_rng(seed)

    flat_inputs: list[np.ndarray] = []
    flat_labels: list[np.ndarray] = []
    split_position_ids: list[np.ndarray] = []
    gzip_scores: list[float] = []
    state_heights: list[int] = []
    state_widths: list[int] = []
    sampled_answer_steps: list[int] = []
    counts: list[int] = []

    shape_groups = _shape_group_specs(config)
    allocated_examples = _allocate_examples_per_shape_group(size, shape_groups, rng)
    progress = tqdm(total=size, desc=f"Generating {split_name}", leave=False)

    for (
        sampled_state_height,
        sampled_state_width,
        sampled_group_answer_steps,
        sampled_group_counts,
    ), target_group_size in zip(shape_groups, allocated_examples.tolist()):
        if target_group_size <= 0:
            continue

        target_group_size = int(target_group_size)
        candidate_batch_size = _candidate_batch_size_for_group(
            config,
            state_height=int(sampled_state_height),
            state_width=int(sampled_state_width),
            counts=int(sampled_group_counts),
            target_group_size=target_group_size,
        )
        min_rounds_for_group = math.ceil(target_group_size / candidate_batch_size)
        effective_max_sampling_rounds = max(config.max_sampling_rounds, min_rounds_for_group)

        nca_config = make_nca_config_for_sample(
            config,
            state_height=int(sampled_state_height),
            state_width=int(sampled_state_width),
            answer_steps=int(sampled_group_answer_steps),
        )
        tokenizer = NCATokenizer(nca_config)
        collected_group_size = 0
        rounds = 0

        while collected_group_size < target_group_size and rounds < effective_max_sampling_rounds:
            rounds += 1

            pair_window_sets = rollout_pair_windows_batched(
                nca_config,
                sample_batch_size=candidate_batch_size,
                example_count=int(sampled_group_counts),
                answer_steps=int(sampled_group_answer_steps),
                time_span=config.time_span,
                device=device,
            )

            if config.gzip_enabled:
                pair_window_sets_cpu = pair_window_sets.cpu().numpy()
                group_gzip_scores = score_pair_window_sets_gzip_batched(
                    pair_window_sets_cpu,
                    tokenizer=tokenizer,
                )
                accepted_mask = np.array(
                    [_passes_gzip_filter(float(score), config) for score in group_gzip_scores],
                    dtype=bool,
                )
                if not np.any(accepted_mask):
                    continue
            else:
                group_gzip_scores = np.full((candidate_batch_size,), -1.0, dtype=np.float32)
                accepted_mask = np.ones((candidate_batch_size,), dtype=bool)

            remaining = target_group_size - collected_group_size
            accepted_indices = np.flatnonzero(accepted_mask)[:remaining]
            if accepted_indices.size == 0:
                continue

            accepted_scores = group_gzip_scores[accepted_indices]
            accepted_pair_window_sets = pair_window_sets[
                torch.as_tensor(
                    accepted_indices,
                    device=pair_window_sets.device,
                    dtype=torch.long,
                )
            ]
            accepted_input_grids, accepted_label_grids = pair_windows_to_input_label_grids_batched(
                accepted_pair_window_sets
            )
            accepted_input_grids, accepted_label_grids = remap_colors_per_sample_batched(
                accepted_input_grids,
                accepted_label_grids,
                config=config,
            )
            batch_inputs, batch_labels, batch_position_ids = build_sample_arrays_batched(
                accepted_input_grids,
                accepted_label_grids,
                counts=int(sampled_group_counts),
                config=config,
            )

            flat_inputs.extend(batch_inputs)
            flat_labels.extend(batch_labels)
            split_position_ids.extend(batch_position_ids)
            gzip_scores.extend(accepted_scores.tolist())
            state_heights.extend([int(sampled_state_height)] * accepted_indices.size)
            state_widths.extend([int(sampled_state_width)] * accepted_indices.size)
            sampled_answer_steps.extend([int(sampled_group_answer_steps)] * accepted_indices.size)
            counts.extend([int(sampled_group_counts)] * accepted_indices.size)
            collected_group_size += int(accepted_indices.size)
            progress.update(int(accepted_indices.size))

        if collected_group_size < target_group_size:
            progress.close()
            raise RuntimeError(
                f"Could not collect enough {split_name} examples for "
                f"state_height={sampled_state_height}, state_width={sampled_state_width}, "
                f"answer_steps={sampled_group_answer_steps}, counts={sampled_group_counts}. "
                f"Collected {collected_group_size} / {target_group_size} after {rounds} rounds. "
                "Try increasing max_sampling_rounds or batch_candidate_size, or relax the gzip thresholds."
            )

    progress.close()

    if len(flat_inputs) != size:
        raise RuntimeError(
            f"Expected to collect exactly {size} {split_name} examples, but got {len(flat_inputs)}."
        )

    split_arrays = _make_split_arrays(
        flat_inputs,
        flat_labels,
        split_position_ids,
        gzip_scores,
        state_heights,
        state_widths,
        sampled_answer_steps,
        counts,
        config,
    )
    return _shuffle_split_arrays(split_arrays, rng)


def save_split(split_name: str, split_arrays: Dict[str, np.ndarray], config: NCA2DDataConfig) -> None:
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
    np.save(split_dir / "all__state_widths.npy", split_arrays["state_widths"])
    np.save(split_dir / "all__answer_steps.npy", split_arrays["answer_steps"])
    np.save(split_dir / "all__counts.npy", split_arrays["counts"])
    np.save(split_dir / "all__input_channels.npy", split_arrays["input_channels"])
    np.save(split_dir / "all__query_channel_indices.npy", split_arrays["query_channel_indices"])

    gzip_summary = _summarize_gzip(split_arrays["gzip_ratio"])
    with open(split_dir / "summary.json", "w") as f:
        json.dump(
            {
                "split": split_name,
                "num_examples": int(split_arrays["inputs"].shape[0]),
                "seq_len": int(config.seq_len),
                "final_image_shape": list(config.final_image_shape),
                **gzip_summary,
                "state_height_min": int(split_arrays["state_heights"].min()),
                "state_height_max": int(split_arrays["state_heights"].max()),
                "state_width_min": int(split_arrays["state_widths"].min()),
                "state_width_max": int(split_arrays["state_widths"].max()),
                "answer_steps_min": int(split_arrays["answer_steps"].min()),
                "answer_steps_max": int(split_arrays["answer_steps"].max()),
                "counts_min": int(split_arrays["counts"].min()),
                "counts_max": int(split_arrays["counts"].max()),
                "input_channels_min": int(split_arrays["input_channels"].min()),
                "input_channels_max": int(split_arrays["input_channels"].max()),
                "position_id_shape": metadata.position_id_shape,
            },
            f,
            indent=2,
        )


def save_dataset_config(config: NCA2DDataConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(
            {
                **config.model_dump(),
                "resolved_state_height_min": config.sampled_state_height_min,
                "resolved_state_height_max": config.sampled_state_height_max,
                "resolved_state_width_min": config.sampled_state_width_min,
                "resolved_state_width_max": config.sampled_state_width_max,
                "resolved_start_step": config.resolved_start_step,
                "resolved_answer_steps_min": config.sampled_answer_steps_min,
                "resolved_answer_steps_max": config.sampled_answer_steps_max,
                "resolved_counts_max": config.resolved_counts_max,
                "resolved_out_colors": config.resolved_out_colors,
                "max_input_channels": config.max_input_channels,
                "seq_len": config.seq_len,
                "vocab_size": config.vocab_size,
                "final_image_shape": list(config.final_image_shape),
                "position_id_shape": list(config.position_id_shape),
                "sample_random_state_height": config.sampled_state_height_min != config.sampled_state_height_max,
                "sample_random_state_width": config.sampled_state_width_min != config.sampled_state_width_max,
                "sample_random_answer_steps": config.sampled_answer_steps_min != config.sampled_answer_steps_max,
                "sample_random_counts": config.sampled_counts_min != config.resolved_counts_max,
                "pair_layout": [
                    "example_1_input",
                    "example_1_label",
                    "...",
                    "example_C_input",
                    "example_C_label",
                    "query_input",
                ],
                "label_layout": {
                    "query_output": "stored on the same channel index as query_input inside all__labels.npy",
                    "exposed_shape": ["H", "W", 1],
                },
                "value_encoding": {
                    "pad": 0,
                    "mask": config.mask_token_id,
                    "internal_nca_colors": config.num_colors,
                    "output_color_range": [
                        config.token_offset,
                        config.token_offset + config.resolved_out_colors - 1,
                    ],
                },
                "time_axis_recipe": {
                    "question_to_answer_steps": "answer_steps",
                    "gap_after_each_pair": "time_span",
                    "window_stride": "answer_steps + 1 + time_span",
                },
                "max_data_seq_len": config.max_data_seq_len,
                "gzip_enabled": config.gzip_enabled,
            },
            f,
            indent=2,
        )

    with open(output_dir / "identifiers.json", "w") as f:
        json.dump(["<blank>"], f, indent=2)


def benchmark_generation(
    config: NCA2DDataConfig,
    *,
    warmup_rounds: int = 2,
    measure_rounds: int = 5,
) -> Dict[str, float]:
    if warmup_rounds < 0:
        raise ValueError(f"warmup_rounds must be >= 0, got {warmup_rounds}")
    if measure_rounds <= 0:
        raise ValueError(f"measure_rounds must be > 0, got {measure_rounds}")

    seed_everything(config.seed)
    device = get_device()
    _configure_generation_backend(device)

    state_height = config.sampled_state_height_max
    state_width = config.sampled_state_width_max
    answer_steps = config.sampled_answer_steps_max
    counts = config.resolved_counts_max
    batch_size = _candidate_batch_size_for_group(
        config,
        state_height=state_height,
        state_width=state_width,
        counts=counts,
        target_group_size=max(config.batch_candidate_size, 64),
    )
    nca_config = make_nca_config_for_sample(
        config,
        state_height=state_height,
        state_width=state_width,
        answer_steps=answer_steps,
    )
    tokenizer = NCATokenizer(nca_config)

    for _ in range(warmup_rounds):
        pair_windows = rollout_pair_windows_batched(
            nca_config,
            sample_batch_size=batch_size,
            example_count=counts,
            answer_steps=answer_steps,
            device=device,
            time_span=config.time_span,
        )
        if config.gzip_enabled:
            pair_windows_cpu = pair_windows.cpu().numpy()
            score_pair_window_sets_gzip_batched(pair_windows_cpu, tokenizer=tokenizer)
        input_grids, label_grids = pair_windows_to_input_label_grids_batched(pair_windows)
        input_grids, label_grids = remap_colors_per_sample_batched(
            input_grids,
            label_grids,
            config=config,
        )
        build_sample_arrays_batched(
            input_grids,
            label_grids,
            counts=counts,
            config=config,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    timings: list[float] = []
    for _ in range(measure_rounds):
        start = time.perf_counter()
        pair_windows = rollout_pair_windows_batched(
            nca_config,
            sample_batch_size=batch_size,
            example_count=counts,
            answer_steps=answer_steps,
            device=device,
            time_span=config.time_span,
        )
        if config.gzip_enabled:
            pair_windows_cpu = pair_windows.cpu().numpy()
            score_pair_window_sets_gzip_batched(pair_windows_cpu, tokenizer=tokenizer)
        input_grids, label_grids = pair_windows_to_input_label_grids_batched(pair_windows)
        input_grids, label_grids = remap_colors_per_sample_batched(
            input_grids,
            label_grids,
            config=config,
        )
        build_sample_arrays_batched(
            input_grids,
            label_grids,
            counts=counts,
            config=config,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)

    mean_seconds = float(np.mean(timings))
    std_seconds = float(np.std(timings))
    samples_per_second = batch_size / max(mean_seconds, 1e-12)
    total_examples = config.train_size + config.test_size
    estimated_total_seconds = total_examples / max(samples_per_second, 1e-12)

    return {
        "batch_size": float(batch_size),
        "state_height": float(state_height),
        "state_width": float(state_width),
        "answer_steps": float(answer_steps),
        "counts": float(counts),
        "mean_seconds_per_batch": mean_seconds,
        "std_seconds_per_batch": std_seconds,
        "samples_per_second": float(samples_per_second),
        "estimated_total_seconds": float(estimated_total_seconds),
    }


def build_dataset(config: NCA2DDataConfig) -> None:
    save_dataset_config(config)

    train_arrays = generate_split("train", config.train_size, config.seed, config)
    save_split("train", train_arrays, config)

    test_arrays = generate_split("test", config.test_size, config.seed + 1, config)
    save_split("test", test_arrays, config)


@cli.command(singleton=True)
def preprocess_data(config: NCA2DDataConfig) -> None:
    build_dataset(config)


if __name__ == "__main__":
    cli()
