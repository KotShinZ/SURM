from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

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

    # Number of in-context example pairs. The saved sample contains
    # [counts examples + 1 query, 2(question/answer), H, W].
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
    # batch_size * (counts + 1) * 2 * H * W.
    max_cells_per_candidate_batch: int = 8_000_000
    max_data_seq_len: int = 65536

    token_offset: int = 2
    mask_token_id: int = 1
    save_dtype: str = "int32"
    no_padding: bool = True
    no_padding_mode: Literal["sample", "pair_eos", "pair_no_eos"] = "pair_eos"

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
        max_tokens_per_sample = 2 * self.max_pair_canvas_height * self.max_pair_canvas_width
        if max_tokens_per_sample <= 0:
            return 0
        return max(self.max_data_seq_len // max_tokens_per_sample - 1, 0)

    @property
    def resolved_counts_max(self) -> int:
        return min(self.sampled_counts_max, self.max_counts_allowed_by_canvas)

    @property
    def max_input_channels(self) -> int:
        return 2 * self.resolved_counts_max + 1

    @property
    def max_pair_slots(self) -> int:
        return self.resolved_counts_max + 1

    @property
    def final_image_shape(self) -> tuple[int, int, int, int]:
        return (
            self.max_pair_slots,
            2,
            self.max_pair_canvas_height,
            self.max_pair_canvas_width,
        )

    @property
    def seq_len(self) -> int:
        num_pairs, io_slots, height, width = self.final_image_shape
        return num_pairs * io_slots * height * width

    @property
    def position_id_shape(self) -> tuple[int, int, int, int]:
        return self.final_image_shape

    @property
    def max_pair_canvas_height(self) -> int:
        return self.sampled_state_height_max + (
            1 if self.no_padding and self.no_padding_mode == "pair_eos" else 0
        )

    @property
    def max_pair_canvas_width(self) -> int:
        return self.sampled_state_width_max + (
            1 if self.no_padding and self.no_padding_mode == "pair_eos" else 0
        )

    @property
    def sequence_layout(self) -> str:
        return self.no_padding_mode if self.no_padding else "fixed"

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

    @staticmethod
    def pair_slots_for_counts(counts: int) -> int:
        if counts <= 0:
            raise ValueError(f"counts must be > 0, got {counts}")
        return counts + 1

    @staticmethod
    def query_pair_index_for_counts(counts: int) -> int:
        return NCA2DDataConfig.pair_slots_for_counts(counts) - 1

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
        if not self.no_padding and self.no_padding_mode != "sample":
            raise ValueError("no_padding_mode is only used when no_padding=True.")
        if self.max_counts_allowed_by_canvas < 1:
            raise ValueError(
                "The maximum HxW canvas already exceeds the max_data_seq_len cap for a single "
                f"example count: pair_canvas_height_max={self.max_pair_canvas_height}, "
                f"pair_canvas_width_max={self.max_pair_canvas_width}, "
                f"max_data_seq_len={self.max_data_seq_len}"
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


def _make_sample_position_ids(
    num_pair_slots: int,
    canvas_shape: Tuple[int, int],
    *,
    dtype: np.dtype = np.int32,
) -> np.ndarray:
    canvas_h, canvas_w = canvas_shape
    return np.moveaxis(
        np.indices((num_pair_slots, 2, canvas_h, canvas_w), dtype=dtype),
        0,
        -1,
    )


def _token_to_debug_symbol(token: int, token_offset: int) -> str:
    token = int(token)
    if token == 0:
        return "."
    if token == 1:
        return "#"
    return str(token - token_offset)


def _format_debug_grid(grid: np.ndarray, token_offset: int) -> str:
    return "\n".join(
        " ".join(_token_to_debug_symbol(token, token_offset) for token in row)
        for row in grid
    )


def _print_terminal_friendly_arc_sample(
    flat_tokens: np.ndarray,
    position_ids: np.ndarray,
    sample_name: str,
    *,
    token_offset: int = 2,
    seq_shape: Optional[Tuple[int, ...]] = None,
) -> None:
    print(f"{sample_name}:")

    if flat_tokens.size == 0:
        print("  empty sample")
        return

    if flat_tokens.ndim == 4:
        flat_tokens = flat_tokens.reshape(-1)
    if position_ids.ndim == 5 and position_ids.shape[-1] == 4:
        position_ids = position_ids.reshape(-1, 4)

    if flat_tokens.ndim != 1:
        print(f"  unexpected token shape={flat_tokens.shape}; falling back to raw print")
        print(flat_tokens)
        return

    if position_ids.ndim != 2 or position_ids.shape != (flat_tokens.shape[0], 4):
        print(
            f"  unexpected position_ids shape={position_ids.shape}; "
            f"expected ({flat_tokens.shape[0]}, 4), falling back to raw print"
        )
        print(flat_tokens)
        return

    pair_ids = np.unique(position_ids[:, 0]).astype(np.int32, copy=False)
    seq_shape_repr = seq_shape if seq_shape is not None else "unknown"
    print(f"  packed_len={flat_tokens.shape[0]}, seq_shape={seq_shape_repr}, pair_slots={len(pair_ids)}")
    print("  legend: .=PAD  #=EOS  other values=encoded NCA colors")

    for pair_pos in pair_ids:
        print(f"  pair {int(pair_pos)}:")
        for io_idx, io_name in enumerate(("input", "output")):
            mask = (position_ids[:, 0] == pair_pos) & (position_ids[:, 1] == io_idx)
            if not np.any(mask):
                continue

            coords = position_ids[mask][:, 2:].astype(np.int32, copy=False)
            grid_h = int(coords[:, 0].max()) + 1
            grid_w = int(coords[:, 1].max()) + 1
            grid = np.zeros((grid_h, grid_w), dtype=flat_tokens.dtype)
            grid[coords[:, 0], coords[:, 1]] = flat_tokens[mask]

            print(f"    {io_name} shape=({grid_h}, {grid_w})")
            for line in _format_debug_grid(grid, token_offset).splitlines():
                print(f"      {line}")


def _build_pair_window_canvas_batched(
    pair_window_sets: torch.Tensor,
    *,
    token_offset: int,
    include_eos: bool,
) -> tuple[torch.Tensor, int, int]:
    if pair_window_sets.ndim != 5:
        raise ValueError(
            f"pair_window_sets must have shape [B, N, 2, H, W], got ndim={pair_window_sets.ndim}"
        )

    batch_size, pair_slots, io_slots, image_height, image_width = pair_window_sets.shape
    if io_slots != 2:
        raise ValueError(
            f"pair_window_sets third axis must be 2, got shape={tuple(pair_window_sets.shape)}"
        )

    canvas_height = image_height + (1 if include_eos else 0)
    canvas_width = image_width + (1 if include_eos else 0)
    device = pair_window_sets.device
    dtype = torch.int32

    canvas = torch.zeros(
        (batch_size, pair_slots, io_slots, canvas_height, canvas_width),
        dtype=dtype,
        device=device,
    )
    canvas[:, :, :, :image_height, :image_width] = pair_window_sets.to(dtype=dtype) + token_offset

    if include_eos:
        canvas[:, :, :, image_height, :image_width] = 1
        canvas[:, :, :, :image_height, image_width] = 1

    return canvas, canvas_height, canvas_width


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
    # Legacy helper kept for notebooks/tests that still work with the old
    # [H, W, 2 * counts + 1] channel-interleaved view.
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


def unflatten_pair_window_inputs(
    flat_tokens: np.ndarray,
    image_height: int,
    image_width: int,
    counts: int,
    token_offset: int,
    padded_pairs: Optional[int] = None,
    padded_io_slots: int = 2,
    padded_height: Optional[int] = None,
    padded_width: Optional[int] = None,
) -> np.ndarray:
    pair_slots = NCA2DDataConfig.pair_slots_for_counts(counts)
    padded_pairs = pair_slots if padded_pairs is None else padded_pairs
    padded_height = image_height if padded_height is None else padded_height
    padded_width = image_width if padded_width is None else padded_width
    if flat_tokens.ndim == 4:
        canvas = flat_tokens
    else:
        canvas = flat_tokens.reshape(padded_pairs, padded_io_slots, padded_height, padded_width)
    return canvas[:pair_slots, :, :image_height, :image_width] - token_offset


def unflatten_input_grid(
    flat_tokens: np.ndarray,
    image_height: int,
    image_width: int,
    counts: int,
    token_offset: int,
    padded_pairs: Optional[int] = None,
    padded_io_slots: int = 2,
    padded_height: Optional[int] = None,
    padded_width: Optional[int] = None,
) -> np.ndarray:
    # Legacy decoder that reconstructs the old [H, W, 2 * counts + 1] view from
    # the new saved [pair, io, H, W] tensor.
    pair_inputs = unflatten_pair_window_inputs(
        flat_tokens,
        image_height=image_height,
        image_width=image_width,
        counts=counts,
        token_offset=token_offset,
        padded_pairs=padded_pairs,
        padded_io_slots=padded_io_slots,
        padded_height=padded_height,
        padded_width=padded_width,
    )
    input_channels = NCA2DDataConfig.input_channels_for_counts(counts)
    query_pair_idx = NCA2DDataConfig.query_pair_index_for_counts(counts)

    legacy_grid = np.zeros((image_height, image_width, input_channels), dtype=pair_inputs.dtype)
    for example_idx in range(counts):
        legacy_grid[:, :, 2 * example_idx] = pair_inputs[example_idx, 0]
        legacy_grid[:, :, 2 * example_idx + 1] = pair_inputs[example_idx, 1]
    legacy_grid[:, :, 2 * counts] = pair_inputs[query_pair_idx, 0]
    return legacy_grid


def extract_label_grid(
    flat_labels: np.ndarray,
    image_height: int,
    image_width: int,
    counts: int,
    token_offset: int,
    padded_pairs: Optional[int] = None,
    padded_io_slots: int = 2,
    padded_height: Optional[int] = None,
    padded_width: Optional[int] = None,
) -> np.ndarray:
    query_pair_idx = NCA2DDataConfig.query_pair_index_for_counts(counts)
    padded_pairs = NCA2DDataConfig.pair_slots_for_counts(counts) if padded_pairs is None else padded_pairs
    padded_height = image_height if padded_height is None else padded_height
    padded_width = image_width if padded_width is None else padded_width
    if flat_labels.ndim == 4:
        canvas = flat_labels
    else:
        canvas = flat_labels.reshape(padded_pairs, padded_io_slots, padded_height, padded_width)
    return canvas[query_pair_idx, 1, :image_height, :image_width, None] - token_offset


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


def remap_pair_windows_per_sample_batched(
    pair_windows: torch.Tensor,
    *,
    config: NCA2DDataConfig,
) -> torch.Tensor:
    if pair_windows.ndim != 5:
        raise ValueError(
            f"pair_windows must have shape [B, N, 2, H, W], got ndim={pair_windows.ndim}"
        )

    if config.resolved_out_colors == config.num_colors:
        return pair_windows

    batch_size = pair_windows.shape[0]
    device = pair_windows.device
    color_maps = sample_output_color_maps_batched(
        batch_size=batch_size,
        num_colors=config.num_colors,
        out_colors=config.resolved_out_colors,
        device=device,
    )

    flat_pair_windows = pair_windows.to(torch.long).reshape(batch_size, -1)
    remapped = torch.gather(color_maps, 1, flat_pair_windows).reshape_as(pair_windows)
    return remapped.to(pair_windows.dtype)


def build_sample_arrays_batched(
    pair_window_sets: torch.Tensor,
    *,
    config: NCA2DDataConfig,
    store_all_pairs: bool,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    if pair_window_sets.ndim != 5:
        raise ValueError(
            f"pair_window_sets must have shape [B, N, 2, H, W], got ndim={pair_window_sets.ndim}"
        )
    if pair_window_sets.shape[2] != 2:
        raise ValueError(
            f"pair_window_sets third axis must be 2, got shape={tuple(pair_window_sets.shape)}"
        )

    batch_size, pair_slots, io_slots, image_height, image_width = pair_window_sets.shape
    example_count = pair_slots - 1
    if example_count <= 0:
        raise ValueError(
            "pair_window_sets must contain at least one example pair and one query pair, "
            f"got shape={tuple(pair_window_sets.shape)}"
        )

    include_eos = config.no_padding and config.no_padding_mode == "pair_eos"
    encoded_pairs, canvas_height, canvas_width = _build_pair_window_canvas_batched(
        pair_window_sets,
        token_offset=config.token_offset,
        include_eos=include_eos,
    )

    device = pair_window_sets.device
    dtype = torch.int32
    pair_axis = pair_slots if config.no_padding else config.max_pair_slots
    io_axis = 2
    height_axis = canvas_height if config.no_padding else config.max_pair_canvas_height
    width_axis = canvas_width if config.no_padding else config.max_pair_canvas_width

    if (
        pair_slots > pair_axis
        or io_slots > io_axis
        or canvas_height > height_axis
        or canvas_width > width_axis
    ):
        raise ValueError(
            "pair_window_sets must fit inside the requested dataset canvas, "
            f"got image_shape={(pair_slots, io_slots, canvas_height, canvas_width)} and "
            f"padded_shape={(pair_axis, io_axis, height_axis, width_axis)}"
        )

    input_canvas = torch.zeros(
        (batch_size, pair_axis, io_axis, height_axis, width_axis),
        dtype=dtype,
        device=device,
    )
    label_canvas = None

    if store_all_pairs:
        input_canvas[:, :pair_slots, :, :canvas_height, :canvas_width] = encoded_pairs
    else:
        label_canvas = torch.zeros_like(input_canvas)
        if example_count > 0:
            input_canvas[:, :example_count, :, :canvas_height, :canvas_width] = encoded_pairs[:, :example_count]

        query_inputs = encoded_pairs[:, example_count, 0]
        query_outputs = encoded_pairs[:, example_count, 1]
        input_canvas[:, example_count, 0, :canvas_height, :canvas_width] = query_inputs
        input_canvas[:, example_count, 1, :canvas_height, :canvas_width] = query_inputs
        label_canvas[:, example_count, 1, :canvas_height, :canvas_width] = query_outputs

    position_ids = torch.stack(
        torch.meshgrid(
            torch.arange(pair_axis, dtype=dtype, device=device),
            torch.arange(io_axis, dtype=dtype, device=device),
            torch.arange(height_axis, dtype=dtype, device=device),
            torch.arange(width_axis, dtype=dtype, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).unsqueeze(0).expand(batch_size, -1, -1, -1, -1, -1)

    seq_shape = np.tile(
        np.array(
            [
                pair_slots if config.no_padding else pair_axis,
                io_axis,
                canvas_height if config.no_padding else height_axis,
                canvas_width if config.no_padding else width_axis,
            ],
            dtype=np.int32,
        ),
        (batch_size, 1),
    )

    if config.no_padding:
        flat_inputs = input_canvas[:, :pair_slots, :, :canvas_height, :canvas_width].reshape(batch_size, -1)
        flat_position_ids = position_ids[:, :pair_slots, :, :canvas_height, :canvas_width].reshape(
            batch_size,
            -1,
            4,
        )
        flat_labels = None
        if label_canvas is not None:
            flat_labels = label_canvas[:, :pair_slots, :, :canvas_height, :canvas_width].reshape(batch_size, -1)
        return (
            flat_inputs.cpu().numpy(),
            None if flat_labels is None else flat_labels.cpu().numpy(),
            flat_position_ids.cpu().numpy(),
            seq_shape,
        )

    return (
        input_canvas.cpu().numpy(),
        None if label_canvas is None else label_canvas.cpu().numpy(),
        position_ids.cpu().numpy(),
        seq_shape,
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


def _make_split_arrays(
    flat_inputs: list[np.ndarray],
    flat_labels: Optional[list[np.ndarray]],
    position_ids: list[np.ndarray],
    seq_shapes: Optional[list[np.ndarray]],
    gzip_scores: list[float],
    state_heights: list[int],
    state_widths: list[int],
    answer_steps: list[int],
    counts: list[int],
    config: NCA2DDataConfig,
) -> tuple[Dict[str, np.ndarray], dict[str, Optional[list[int]] | int | bool]]:
    num_examples = len(flat_inputs)
    save_labels = flat_labels is not None

    if config.no_padding:
        seq_lengths = np.asarray([sample.shape[0] for sample in flat_inputs], dtype=np.int64)
        seq_offsets = np.concatenate(
            [np.array([0], dtype=np.int64), np.cumsum(seq_lengths, dtype=np.int64)]
        )
        inputs = (
            np.concatenate(flat_inputs, axis=0).astype(config.save_dtype, copy=False)
            if flat_inputs
            else np.empty((0,), dtype=config.save_dtype)
        )
        position_id_array = (
            np.concatenate(position_ids, axis=0).astype(np.int32, copy=False)
            if position_ids
            else np.empty((0, 4), dtype=np.int32)
        )
        labels = None
        if flat_labels is not None:
            labels = (
                np.concatenate(flat_labels, axis=0).astype(config.save_dtype, copy=False)
                if flat_labels
                else np.empty((0,), dtype=config.save_dtype)
            )
        split_max_seq_len = int(seq_lengths.max()) if seq_lengths.size > 0 else 0
        split_position_id_shape = (
            (position_id_array.max(axis=0).astype(np.int32) + 1).tolist()
            if position_id_array.size > 0
            else None
        )
    else:
        inputs = (
            np.stack(flat_inputs, axis=0).astype(config.save_dtype, copy=False)
            if flat_inputs
            else np.empty((0, *config.final_image_shape), dtype=config.save_dtype)
        )
        position_id_array = (
            np.stack(position_ids, axis=0).astype(np.int32, copy=False)
            if position_ids
            else np.empty((0, *config.final_image_shape, 4), dtype=np.int32)
        )
        labels = None
        if flat_labels is not None:
            labels = (
                np.stack(flat_labels, axis=0).astype(config.save_dtype, copy=False)
                if flat_labels
                else np.empty((0, *config.final_image_shape), dtype=config.save_dtype)
            )
        split_max_seq_len = int(config.seq_len)
        split_position_id_shape = list(config.position_id_shape) if num_examples > 0 else None

    arrays = {
        "inputs": inputs,
        "position_ids": position_id_array,
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
    if labels is not None:
        arrays["labels"] = labels
    if config.no_padding:
        arrays["seq_offsets"] = seq_offsets
        arrays["seq_shapes"] = (
            np.asarray(seq_shapes, dtype=np.int32)
            if seq_shapes is not None
            else np.empty((0, 4), dtype=np.int32)
        )

    stats: dict[str, Optional[list[int]] | int | bool] = {
        "num_examples": num_examples,
        "save_labels": save_labels,
        "seq_len": split_max_seq_len,
        "position_id_shape": split_position_id_shape,
    }
    return arrays, stats


def _candidate_batch_size_for_group(
    config: NCA2DDataConfig,
    *,
    state_height: int,
    state_width: int,
    counts: int,
    target_group_size: int,
) -> int:
    canvas_height = state_height + (1 if config.no_padding and config.no_padding_mode == "pair_eos" else 0)
    canvas_width = state_width + (1 if config.no_padding and config.no_padding_mode == "pair_eos" else 0)
    per_sample_cells = canvas_height * canvas_width * max(2 * (counts + 1), 1)
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


def generate_split(
    split_name: str,
    size: int,
    seed: int,
    config: NCA2DDataConfig,
) -> tuple[Dict[str, np.ndarray], dict[str, Optional[list[int]] | int | bool]]:
    seed_everything(seed)
    device = get_device()
    _configure_generation_backend(device)
    rng = np.random.default_rng(seed)
    store_all_pairs = split_name == "train"

    flat_inputs: list[np.ndarray] = []
    flat_labels: Optional[list[np.ndarray]] = [] if not store_all_pairs else None
    split_position_ids: list[np.ndarray] = []
    split_seq_shapes: Optional[list[np.ndarray]] = [] if config.no_padding else None
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
            accepted_pair_window_sets = remap_pair_windows_per_sample_batched(
                accepted_pair_window_sets,
                config=config,
            )
            batch_inputs, batch_labels, batch_position_ids, batch_seq_shapes = build_sample_arrays_batched(
                accepted_pair_window_sets,
                config=config,
                store_all_pairs=store_all_pairs,
            )

            flat_inputs.extend(batch_inputs)
            if flat_labels is not None:
                if batch_labels is None:
                    raise RuntimeError("Expected labels for non-train split, but build_sample_arrays_batched returned None.")
                flat_labels.extend(batch_labels)
            split_position_ids.extend(batch_position_ids)
            if split_seq_shapes is not None:
                split_seq_shapes.extend(batch_seq_shapes)
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

    if size > 1:
        permutation = rng.permutation(size).tolist()
        flat_inputs = [flat_inputs[idx] for idx in permutation]
        split_position_ids = [split_position_ids[idx] for idx in permutation]
        gzip_scores = [gzip_scores[idx] for idx in permutation]
        state_heights = [state_heights[idx] for idx in permutation]
        state_widths = [state_widths[idx] for idx in permutation]
        sampled_answer_steps = [sampled_answer_steps[idx] for idx in permutation]
        counts = [counts[idx] for idx in permutation]
        if flat_labels is not None:
            flat_labels = [flat_labels[idx] for idx in permutation]
        if split_seq_shapes is not None:
            split_seq_shapes = [split_seq_shapes[idx] for idx in permutation]

    if flat_inputs:
        seq_shape = tuple(int(x) for x in (split_seq_shapes[0] if split_seq_shapes is not None else flat_inputs[0].shape))
        _print_terminal_friendly_arc_sample(
            flat_inputs[0],
            split_position_ids[0],
            f"{split_name}/inputs[0]",
            token_offset=config.token_offset,
            seq_shape=seq_shape,
        )
        if flat_labels is not None:
            _print_terminal_friendly_arc_sample(
                flat_labels[0],
                split_position_ids[0],
                f"{split_name}/labels[0]",
                token_offset=config.token_offset,
                seq_shape=seq_shape,
            )

    split_arrays, split_stats = _make_split_arrays(
        flat_inputs,
        flat_labels,
        split_position_ids,
        split_seq_shapes,
        gzip_scores,
        state_heights,
        state_widths,
        sampled_answer_steps,
        counts,
        config,
    )
    return split_arrays, split_stats


def save_split(
    split_name: str,
    split_arrays: Dict[str, np.ndarray],
    split_stats: dict[str, Optional[list[int]] | int | bool],
    config: NCA2DDataConfig,
) -> None:
    split_dir = Path(config.output_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    save_labels = bool(split_stats["save_labels"])

    metadata = PuzzleDatasetMetadata(
        seq_len=int(split_stats["seq_len"]),
        vocab_size=config.vocab_size,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=split_arrays["group_indices"].size - 1,
        mean_puzzle_examples=1.0,
        sets=["all"],
        variable_seq_lengths=config.no_padding,
        position_id_shape=split_stats["position_id_shape"],
        sequence_layout=config.sequence_layout,
        train_target_mode="random_output_pair" if split_name == "train" else None,
    )

    with open(split_dir / "dataset.json", "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    np.save(split_dir / "all__inputs.npy", split_arrays["inputs"])
    if save_labels:
        np.save(split_dir / "all__labels.npy", split_arrays["labels"])
    np.save(split_dir / "all__position_ids.npy", split_arrays["position_ids"])
    if config.no_padding:
        np.save(split_dir / "all__seq_offsets.npy", split_arrays["seq_offsets"])
        np.save(split_dir / "all__seq_shapes.npy", split_arrays["seq_shapes"])
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
                "num_examples": int(split_stats["num_examples"]),
                "seq_len": int(split_stats["seq_len"]),
                "final_image_shape": list(config.final_image_shape),
                "variable_seq_lengths": config.no_padding,
                "sequence_layout": config.sequence_layout,
                "save_labels": save_labels,
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
                "max_pair_canvas_height": config.max_pair_canvas_height,
                "max_pair_canvas_width": config.max_pair_canvas_width,
                "max_pair_slots": config.max_pair_slots,
                "max_input_channels": config.max_input_channels,
                "seq_len": config.seq_len,
                "vocab_size": config.vocab_size,
                "final_image_shape": list(config.final_image_shape),
                "position_id_shape": list(config.position_id_shape),
                "variable_seq_lengths": config.no_padding,
                "sequence_layout": config.sequence_layout,
                "sample_random_state_height": config.sampled_state_height_min != config.sampled_state_height_max,
                "sample_random_state_width": config.sampled_state_width_min != config.sampled_state_width_max,
                "sample_random_answer_steps": config.sampled_answer_steps_min != config.sampled_answer_steps_max,
                "sample_random_counts": config.sampled_counts_min != config.resolved_counts_max,
                "pair_layout": {
                    "stored_shape": ["num_pairs", "question_answer", "H", "W"],
                    "train_split": "stores all counts + 1 pairs without labels; one output pair is selected dynamically during training",
                    "eval_split": "stores counts context pairs plus one labeled query pair",
                    "num_pairs": "counts + 1 sampled temporal pairs",
                    "question_answer": ["question", "answer"],
                    "train_pair_content": "all pairs are saved as-is",
                    "target_pair_input": ["query_input", "query_input_copy"],
                    "target_pair_label": "stored only for eval samples at [query_pair, answer]",
                },
                "legacy_channel_view": {
                    "input_shape": ["H", "W", "2 * counts + 1"],
                    "label_shape": ["H", "W", 1],
                    "helpers": ["unflatten_input_grid", "extract_label_grid"],
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
        pair_windows = remap_pair_windows_per_sample_batched(
            pair_windows,
            config=config,
        )
        build_sample_arrays_batched(
            pair_windows,
            config=config,
            store_all_pairs=True,
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
        pair_windows = remap_pair_windows_per_sample_batched(
            pair_windows,
            config=config,
        )
        build_sample_arrays_batched(
            pair_windows,
            config=config,
            store_all_pairs=True,
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

    train_arrays, train_stats = generate_split("train", config.train_size, config.seed, config)
    save_split("train", train_arrays, train_stats, config)

    test_arrays, test_stats = generate_split("test", config.test_size, config.seed + 1, config)
    save_split("test", test_arrays, test_stats, config)


@cli.command(singleton=True)
def preprocess_data(config: NCA2DDataConfig) -> None:
    build_dataset(config)


if __name__ == "__main__":
    cli()
