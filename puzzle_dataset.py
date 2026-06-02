import os
import json
import math
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from models.losses import IGNORE_LABEL_ID
from data.common import PuzzleDatasetMetadata
from data.online_aug import OnlineAugConfig, apply_online_aug


ARC_MAX_GRID_SIZE = 30
ARC_EOS_TOKEN_ID = 1
ForwardMode = Literal["standard", "answer_only", "prefix_lm", "casual", "causal"]


class MaskedInputConfig(pydantic.BaseModel):
    enabled: bool = False
    apply_probability: float = 1.0
    min_mask_ratio: float = 0.0
    max_mask_ratio: float = 0.0
    mask_token_id: int = 1
    preserve_source_inputs: bool = True

    @pydantic.model_validator(mode="after")
    def _validate_ranges(self):
        if not (0.0 <= self.apply_probability <= 1.0):
            raise ValueError(f"apply_probability must be in [0, 1], got {self.apply_probability}")
        if not (0.0 <= self.min_mask_ratio <= 1.0):
            raise ValueError(f"min_mask_ratio must be in [0, 1], got {self.min_mask_ratio}")
        if not (0.0 <= self.max_mask_ratio <= 1.0):
            raise ValueError(f"max_mask_ratio must be in [0, 1], got {self.max_mask_ratio}")
        if self.min_mask_ratio > self.max_mask_ratio:
            raise ValueError(
                f"min_mask_ratio ({self.min_mask_ratio}) must be <= max_mask_ratio ({self.max_mask_ratio})"
            )
        if self.mask_token_id < 0:
            raise ValueError(f"mask_token_id must be >= 0, got {self.mask_token_id}")
        return self


class ARCOutputMaskConfig(pydantic.BaseModel):
    enabled: bool = True
    fill_mode: Literal["zero", "random_color"] = "zero"
    preserve_source_inputs: bool = True
    answer_slot_max_grid_size: Optional[int] = None
    min_context_pairs: Optional[int] = None

    @pydantic.model_validator(mode="after")
    def _validate_answer_slot_size(self):
        if self.answer_slot_max_grid_size is not None and self.answer_slot_max_grid_size <= 0:
            raise ValueError(
                "answer_slot_max_grid_size must be a positive integer when provided, "
                f"got {self.answer_slot_max_grid_size}"
            )
        if self.min_context_pairs is not None and self.min_context_pairs < 0:
            raise ValueError(
                "min_context_pairs must be >= 0 when provided, "
                f"got {self.min_context_pairs}"
            )
        return self


def _debug_print_arc_variable_batch(
    inputs: np.ndarray,
    labels: np.ndarray,
    position_ids: np.ndarray,
    seq_offsets: np.ndarray,
) -> None:
    # Reuse the dataset builder's ARC pretty-printer so debug output stays consistent.
    from data.build_arc_dataset_full_2 import print_data

    num_samples = max(int(seq_offsets.shape[0]) - 1, 0)
    for sample_idx in range(num_samples):
        start = int(seq_offsets[sample_idx])
        end = int(seq_offsets[sample_idx + 1])
        sample_position_ids = position_ids[start:end]

        print_data(inputs[start:end], sample_position_ids, title=f"batch[{sample_idx}] inputs")
        print_data(labels[start:end], sample_position_ids, title=f"batch[{sample_idx}] labels")


def _sample_batch(
    rng: np.random.Generator,
    group_order: np.ndarray,
    puzzle_indices: np.ndarray,
    group_indices: np.ndarray,
    start_index: int,
    global_batch_size: int,
    data_fraction: float = 1.0,
    examples_per_puzzle: Optional[int] = 1,
):
    if examples_per_puzzle is not None and examples_per_puzzle <= 0:
        raise ValueError(
            "examples_per_puzzle must be a positive integer, or None for full-puzzle sampling; "
            f"got {examples_per_puzzle}"
        )

    # Pack examples into a full batch
    batch = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        # Pick a group and a puzzle from that group
        group_id = group_order[start_index]
        group_start = group_indices[group_id]
        group_end = group_indices[group_id + 1]
        # Limit puzzles per group based on data_fraction
        group_end_limited = group_start + max(1, round((group_end - group_start) * data_fraction))
        puzzle_id = rng.integers(group_start, group_end_limited)
        start_index += 1

        # Get range of the puzzle
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        puzzle_sample_size = puzzle_size
        if examples_per_puzzle is not None:
            puzzle_sample_size = min(puzzle_sample_size, examples_per_puzzle)

        append_size = min(puzzle_sample_size, global_batch_size - current_size)

        # Put into batch
        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(puzzle_start + rng.choice(puzzle_size, append_size, replace=False))

        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)


class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_path: str
    global_batch_size: int
    test_set_mode: bool
    padding: bool = False

    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead.
    grad_accum_steps: int = 1

    rank: int
    num_replicas: int

    data_fraction: float = 1.0  # Fraction of training groups to use per epoch (1.0 = all)

    # Number of randomly selected examples to draw from each puzzle during training.
    # Set to None to use the previous full-puzzle packing behavior.
    examples_per_puzzle: Optional[int] = 1

    # Online augmentation applied at training time (None = disabled)
    online_aug: Optional[OnlineAugConfig] = None

    # Replace model inputs with randomly masked labels.
    masked_input: Optional[MaskedInputConfig] = None

    # ARC full-context training: mask one output pair on the fly and generate labels from it.
    arc_output_mask: Optional[ARCOutputMaskConfig] = None

    # ARC full-context training built from one-pair examples.
    full_min_pairs: int = 2
    full_max_pairs: int = 8
    full_answer_initial_mode: Literal["black", "noised_label"] = "black"
    full_answer_initial_black_token_id: int = 2
    full_answer_initial_gamma_min: float = 0.0
    full_answer_initial_gamma_max: float = 1.0
    full_answer_initial_noise_token_min: int = 2
    full_answer_initial_noise_token_max: int = 11

    # Forward/data mode. Legacy aliases below are kept for older configs.
    forward_mode: ForwardMode = "standard"

    # Emit labels as only the answer tokens while keeping inputs full-length.
    answer_only_labels: bool = False
    casual: bool = False
    causal_lm_start_token_id: int = 1

    # Separate problem and answer tokens in the model input.
    label_separate: bool = False
    SeparateMode: str = "D"
    separate_mode: Optional[str] = None
    label_separate_noise_token_min: int = 2
    label_separate_noise_token_max: Optional[int] = None

    @pydantic.model_validator(mode="before")
    @classmethod
    def _translate_legacy_nopadding(cls, values):
        if isinstance(values, dict) and "padding" not in values and "nopadding" in values:
            values = dict(values)
            values["padding"] = not pydantic.TypeAdapter(bool).validate_python(values.pop("nopadding"))
        return values

    @pydantic.model_validator(mode="after")
    def _validate_training_sampling(self):
        if self.forward_mode == "causal":
            self.forward_mode = "casual"
        if self.forward_mode == "standard" and self.casual:
            self.forward_mode = "prefix_lm"
        if self.forward_mode in {"answer_only", "prefix_lm"}:
            self.answer_only_labels = True
        if self.forward_mode in {"prefix_lm", "casual"}:
            self.casual = True

        if self.examples_per_puzzle is not None and self.examples_per_puzzle <= 0:
            raise ValueError(
                "examples_per_puzzle must be a positive integer, or None for full-puzzle sampling; "
                f"got {self.examples_per_puzzle}"
            )
        if self.full_min_pairs <= 0:
            raise ValueError(f"full_min_pairs must be positive, got {self.full_min_pairs}")
        if self.full_max_pairs < self.full_min_pairs:
            raise ValueError(
                f"full_max_pairs ({self.full_max_pairs}) must be >= full_min_pairs ({self.full_min_pairs})"
            )
        if self.full_answer_initial_black_token_id < 0:
            raise ValueError(
                "full_answer_initial_black_token_id must be >= 0, "
                f"got {self.full_answer_initial_black_token_id}"
            )
        if not (0.0 <= self.full_answer_initial_gamma_min <= 1.0):
            raise ValueError(
                "full_answer_initial_gamma_min must be in [0, 1], "
                f"got {self.full_answer_initial_gamma_min}"
            )
        if not (0.0 <= self.full_answer_initial_gamma_max <= 1.0):
            raise ValueError(
                "full_answer_initial_gamma_max must be in [0, 1], "
                f"got {self.full_answer_initial_gamma_max}"
            )
        if self.full_answer_initial_gamma_min > self.full_answer_initial_gamma_max:
            raise ValueError(
                "full_answer_initial_gamma_min must be <= full_answer_initial_gamma_max, "
                f"got {self.full_answer_initial_gamma_min} > {self.full_answer_initial_gamma_max}"
            )
        if self.full_answer_initial_noise_token_min < 0:
            raise ValueError(
                "full_answer_initial_noise_token_min must be >= 0, "
                f"got {self.full_answer_initial_noise_token_min}"
            )
        if self.full_answer_initial_noise_token_max < self.full_answer_initial_noise_token_min:
            raise ValueError(
                "full_answer_initial_noise_token_max must be >= full_answer_initial_noise_token_min, "
                f"got {self.full_answer_initial_noise_token_max} < {self.full_answer_initial_noise_token_min}"
            )
        if self.causal_lm_start_token_id < 0:
            raise ValueError(
                "causal_lm_start_token_id must be >= 0, "
                f"got {self.causal_lm_start_token_id}"
            )
        if self.label_separate_noise_token_min < 0:
            raise ValueError(
                "label_separate_noise_token_min must be >= 0, "
                f"got {self.label_separate_noise_token_min}"
            )
        if (
            self.label_separate_noise_token_max is not None
            and self.label_separate_noise_token_max < self.label_separate_noise_token_min
        ):
            raise ValueError(
                "label_separate_noise_token_max must be >= label_separate_noise_token_min, "
                f"got {self.label_separate_noise_token_max} < {self.label_separate_noise_token_min}"
            )
        mode = (self.separate_mode or self.SeparateMode).upper()
        if mode not in {"C", "D"}:
            raise ValueError(f"SeparateMode must be 'C' or 'D', got {self.separate_mode or self.SeparateMode!r}")
        return self

    def emits_answer_only_labels(self) -> bool:
        return bool(self.answer_only_labels or self.forward_mode in {"answer_only", "prefix_lm"})

    def uses_prefix_lm(self) -> bool:
        return bool(self.casual or self.forward_mode in {"prefix_lm", "casual"})

    def uses_casual_lm(self) -> bool:
        return self.forward_mode == "casual"


class PuzzleDataset(IterableDataset):
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        if not os.path.isdir(os.path.join(config.dataset_path, split)):
            raise FileNotFoundError(f"Dataset split {split} in {config.dataset_path} does not exist.")
        
        self.config = config
        self.split = split
        self.metadata = self._load_metadata()
        self._configure_casual_lm_end_token()
        
        # Checks
        assert self.config.global_batch_size % self.config.num_replicas == 0, f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State
        self._data = None
        self._iters = 0

    def _load_metadata(self) -> PuzzleDatasetMetadata:
        with open(os.path.join(self.config.dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _configure_casual_lm_end_token(self) -> None:
        self.casual_lm_end_token_id: Optional[int] = None
        if not self.config.uses_casual_lm():
            return

        self.casual_lm_end_token_id = int(self.metadata.vocab_size)
        self.metadata.vocab_size = int(self.metadata.vocab_size) + 1

    def _casual_lm_end_token_id(self) -> int:
        token_id = getattr(self, "casual_lm_end_token_id", None)
        if token_id is not None:
            return int(token_id)
        return int(self.metadata.vocab_size)

    def _strip_casual_eval_targets(self, batch: dict) -> dict:
        if not self.config.uses_casual_lm():
            return batch

        if "position_ids" in batch:
            batch["prompt_position_ids"] = batch.pop("position_ids")
        for key in (
            "labels",
            "answer_mask",
            "label_seq_lengths",
            "label_seq_offsets",
            "label_seq_shapes",
        ):
            batch.pop(key, None)
        return batch

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",

            # Keep indices in memory
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None
        }
        if self.metadata.variable_seq_lengths:
            field_mmap_modes["seq_offsets"] = None
            field_mmap_modes["seq_shapes"] = None

        # Load data
        self._data = {}
        for set_name in self.metadata.sets:
            split_dir = os.path.join(self.config.dataset_path, self.split)
            set_fields = dict(field_mmap_modes)
            labels_path = os.path.join(split_dir, f"{set_name}__labels.npy")
            if os.path.isfile(labels_path):
                set_fields["labels"] = "r"
                if self.metadata.variable_seq_lengths:
                    label_offsets_path = os.path.join(split_dir, f"{set_name}__label_seq_offsets.npy")
                    label_shapes_path = os.path.join(split_dir, f"{set_name}__label_seq_shapes.npy")
                    if os.path.isfile(label_offsets_path):
                        set_fields["label_seq_offsets"] = None
                    if os.path.isfile(label_shapes_path):
                        set_fields["label_seq_shapes"] = None
            position_ids_path = os.path.join(split_dir, f"{set_name}__position_ids.npy")
            if os.path.isfile(position_ids_path):
                set_fields["position_ids"] = "r"

            # Load subset
            self._data[set_name] = {
                field_name: np.load(os.path.join(split_dir, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                for field_name, mmap_mode in set_fields.items()
            }

    @staticmethod
    def _pad_flat_2d_tokens(
        tokens: np.ndarray,
        shape: np.ndarray,
        target_shape: np.ndarray,
        pad_token: int,
    ) -> np.ndarray:
        shape_tuple = tuple(int(v) for v in shape)
        target_tuple = tuple(int(v) for v in target_shape)
        if shape_tuple == target_tuple:
            return tokens

        if len(shape_tuple) != 2 or len(target_tuple) != 2:
            raise ValueError(
                "Padding variable-length inputs and labels currently requires 2D shapes, "
                f"got shape={shape_tuple}, target_shape={target_tuple}."
            )

        h, w = shape_tuple
        target_h, target_w = target_tuple
        if h > target_h or w > target_w:
            raise ValueError(
                f"Cannot pad shape={shape_tuple} into smaller target_shape={target_tuple}."
            )
        if int(tokens.shape[0]) != h * w:
            raise ValueError(
                f"Token length does not match shape: len={tokens.shape[0]}, shape={shape_tuple}."
            )

        padded = np.full((target_h, target_w), pad_token, dtype=tokens.dtype)
        padded[:h, :w] = tokens.reshape(h, w)
        return padded.reshape(-1)

    @staticmethod
    def _make_padded_position_ids(
        position_ids: np.ndarray,
        target_shape: np.ndarray,
    ) -> np.ndarray:
        target_h, target_w = (int(v) for v in target_shape)
        rows, cols = np.indices((target_h, target_w), dtype=np.int32)
        flat_rows = rows.reshape(-1)
        flat_cols = cols.reshape(-1)

        if position_ids.ndim != 2:
            raise ValueError(f"Expected position_ids with ndim=2, got shape={position_ids.shape}.")
        if position_ids.shape[1] == 2:
            return np.stack([flat_rows, flat_cols], axis=-1).astype(position_ids.dtype, copy=False)
        if position_ids.shape[1] >= 4:
            leading = np.zeros((target_h * target_w, position_ids.shape[1] - 2), dtype=position_ids.dtype)
            if position_ids.shape[0] > 0:
                leading[:] = position_ids[0, :-2]
            return np.concatenate(
                [leading, np.stack([flat_rows, flat_cols], axis=-1).astype(position_ids.dtype, copy=False)],
                axis=-1,
            )
        raise ValueError(f"Unsupported position_ids shape for padding: {position_ids.shape}.")

    def _should_pad_variable_inputs_and_labels(
        self,
        dataset: dict,
        input_shapes: np.ndarray,
        label_shapes: np.ndarray,
    ) -> bool:
        return (
            "labels" in dataset
            and "seq_shapes" in dataset
            and "label_seq_shapes" in dataset
            and input_shapes.ndim == 2
            and label_shapes.ndim == 2
            and input_shapes.shape[1] == 2
            and label_shapes.shape[1] == 2
        )

    def _select_examples(self, dataset: dict, indices: np.ndarray) -> dict:
        if not self.metadata.variable_seq_lengths:
            batch = {
                "inputs": dataset["inputs"][indices],
            }
            if "labels" in dataset:
                batch["labels"] = dataset["labels"][indices]
            if "position_ids" in dataset:
                batch["position_ids"] = dataset["position_ids"][indices]
            return batch

        offsets = dataset["seq_offsets"]
        shapes = dataset["seq_shapes"][indices]
        lengths = (offsets[indices + 1] - offsets[indices]).astype(np.int32, copy=False)
        label_offsets = dataset.get("label_seq_offsets", offsets)
        label_shapes = dataset.get("label_seq_shapes", dataset["seq_shapes"])
        selected_label_shapes = label_shapes[indices]
        label_lengths = (
            (label_offsets[indices + 1] - label_offsets[indices]).astype(np.int32, copy=False)
            if "labels" in dataset
            else None
        )
        pad_inputs_and_labels = self._should_pad_variable_inputs_and_labels(
            dataset,
            shapes,
            selected_label_shapes,
        )
        if pad_inputs_and_labels:
            if self.config.padding:
                target_shapes = np.full(shapes.shape, ARC_MAX_GRID_SIZE, dtype=np.int32)
            else:
                target_shapes = np.maximum(shapes, selected_label_shapes).astype(np.int32, copy=False)

        input_chunks = []
        position_chunks = []
        label_chunks = [] if "labels" in dataset else None
        for batch_idx, example_idx in enumerate(indices):
            start = int(offsets[example_idx])
            end = int(offsets[example_idx + 1])
            input_chunk = dataset["inputs"][start:end]
            if label_chunks is not None:
                label_start = int(label_offsets[example_idx])
                label_end = int(label_offsets[example_idx + 1])
                label_chunk = dataset["labels"][label_start:label_end]
                if pad_inputs_and_labels:
                    input_chunk = self._pad_flat_2d_tokens(
                        input_chunk,
                        shapes[batch_idx],
                        target_shapes[batch_idx],
                        self.metadata.pad_id,
                    )
                    label_chunk = self._pad_flat_2d_tokens(
                        label_chunk,
                        selected_label_shapes[batch_idx],
                        target_shapes[batch_idx],
                        self.metadata.pad_id,
                    )
                label_chunks.append(label_chunk)
            input_chunks.append(input_chunk)
            if "position_ids" in dataset:
                position_chunk = dataset["position_ids"][start:end]
                if pad_inputs_and_labels:
                    position_chunk = self._make_padded_position_ids(position_chunk, target_shapes[batch_idx])
                position_chunks.append(position_chunk)

        if pad_inputs_and_labels:
            shapes = target_shapes
            selected_label_shapes = target_shapes
            lengths = (target_shapes[:, 0] * target_shapes[:, 1]).astype(np.int32, copy=False)
            label_lengths = lengths.copy()

        if input_chunks:
            inputs = np.concatenate(input_chunks).astype(np.uint8, copy=False)
        else:
            inputs = np.empty((0,), dtype=np.uint8)

        batch = {
            "inputs": inputs,
            "seq_lengths": lengths,
            "seq_offsets": np.concatenate(
                [np.zeros((1,), dtype=np.int32), np.cumsum(lengths, dtype=np.int32)]
            ),
            "seq_shapes": shapes,
        }
        if label_chunks is not None:
            if label_chunks:
                batch["labels"] = np.concatenate(label_chunks).astype(np.uint8, copy=False)
            else:
                batch["labels"] = np.empty((0,), dtype=np.uint8)
            batch["label_seq_lengths"] = label_lengths
            batch["label_seq_offsets"] = np.concatenate(
                [np.zeros((1,), dtype=np.int32), np.cumsum(label_lengths, dtype=np.int32)]
            )
            batch["label_seq_shapes"] = selected_label_shapes
        if "position_ids" in dataset:
            if position_chunks:
                batch["position_ids"] = np.concatenate(position_chunks, axis=0).astype(
                    dataset["position_ids"].dtype,
                    copy=False,
                )
            else:
                batch["position_ids"] = np.empty((0, dataset["position_ids"].shape[1]), dtype=dataset["position_ids"].dtype)
        return batch

    def _make_masked_inputs(
        self,
        source_inputs: np.ndarray,
        labels: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        cfg = self.config.masked_input
        assert cfg is not None

        masked_inputs = source_inputs.astype(np.int32, copy=True)
        valid_mask = labels != IGNORE_LABEL_ID

        for row_idx in range(masked_inputs.shape[0]):
            if rng.random() > cfg.apply_probability:
                continue

            valid_positions = np.flatnonzero(valid_mask[row_idx])
            if valid_positions.size == 0:
                continue

            # Start from the full answer only for samples where masked-input mode is applied.
            masked_inputs[row_idx] = np.where(
                valid_mask[row_idx],
                labels[row_idx],
                self.metadata.pad_id,
            ).astype(np.int32, copy=False)

            if cfg.min_mask_ratio == cfg.max_mask_ratio:
                mask_ratio = cfg.min_mask_ratio
            else:
                mask_ratio = float(rng.uniform(cfg.min_mask_ratio, cfg.max_mask_ratio))

            mask_count = int(round(valid_positions.size * mask_ratio))
            mask_count = min(max(mask_count, 0), valid_positions.size)
            if mask_count == 0:
                continue

            masked_positions = rng.choice(valid_positions, size=mask_count, replace=False)
            masked_inputs[row_idx, masked_positions] = cfg.mask_token_id

        return masked_inputs

    def _should_apply_arc_output_mask(self, make_masked_inputs: bool) -> bool:
        cfg = self._arc_output_mask_cfg()
        return bool(
            make_masked_inputs
            and cfg is not None
            and cfg.enabled
            and self.split == "train"
            and self.metadata.train_target_mode == "random_output_pair"
        )

    def _arc_output_mask_cfg(self) -> Optional[ARCOutputMaskConfig]:
        if self.config.arc_output_mask is not None:
            return self.config.arc_output_mask
        if self.metadata.train_target_mode == "random_output_pair":
            return ARCOutputMaskConfig()
        return None

    def _arc_answer_slot_size(self) -> Optional[int]:
        cfg = self._arc_output_mask_cfg()
        if cfg is not None and cfg.answer_slot_max_grid_size is not None:
            return int(cfg.answer_slot_max_grid_size)
        if self.metadata.answer_slot_max_grid_size is not None:
            return int(self.metadata.answer_slot_max_grid_size)
        return None

    def _arc_min_context_pairs(self) -> Optional[int]:
        cfg = self._arc_output_mask_cfg()
        if cfg is not None and cfg.min_context_pairs is not None:
            return int(cfg.min_context_pairs)
        if self.metadata.min_context_pairs is not None:
            return int(self.metadata.min_context_pairs)
        return None

    def _arc_mask_fill_token(self, rng: np.random.Generator) -> int:
        cfg = self._arc_output_mask_cfg()
        assert cfg is not None
        if cfg.fill_mode == "zero":
            return 2
        return int(rng.integers(2, 12))

    @staticmethod
    def _make_arc_position_ids(pair_id: int, io_id: int, canvas_shape: Tuple[int, int]) -> np.ndarray:
        canvas_h, canvas_w = canvas_shape
        row_col_ids = np.moveaxis(np.indices((canvas_h, canvas_w), dtype=np.int32), 0, -1).reshape(-1, 2)
        pair_ids = np.full((row_col_ids.shape[0], 1), pair_id, dtype=np.int32)
        io_ids = np.full((row_col_ids.shape[0], 1), io_id, dtype=np.int32)
        return np.concatenate([pair_ids, io_ids, row_col_ids], axis=-1)

    def _extract_arc_pair_entries(
        self,
        sample_tokens: np.ndarray,
        sample_position_ids: np.ndarray,
    ) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        if sample_position_ids.ndim != 2 or sample_position_ids.shape[1] != 4:
            raise ValueError(
                "ARC output masking requires 4D position_ids with shape [seq_len, 4], "
                f"got {sample_position_ids.shape}"
            )

        pair_entries: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for pair_id in np.unique(sample_position_ids[:, 0]).astype(np.int32, copy=False).tolist():
            pair_mask = sample_position_ids[:, 0] == pair_id
            if not np.any(sample_tokens[pair_mask] != self.metadata.pad_id):
                continue

            canvases: Dict[int, np.ndarray] = {}
            for io_id in (0, 1):
                io_mask = pair_mask & (sample_position_ids[:, 1] == io_id)
                if not np.any(io_mask):
                    continue

                coords = sample_position_ids[io_mask][:, 2:].astype(np.int32, copy=False)
                canvas_h = int(coords[:, 0].max()) + 1
                canvas_w = int(coords[:, 1].max()) + 1
                canvas = np.zeros((canvas_h, canvas_w), dtype=np.int32)
                canvas[coords[:, 0], coords[:, 1]] = sample_tokens[io_mask]
                canvases[io_id] = canvas

            if 0 in canvases and 1 in canvases:
                pair_entries.append((pair_id, canvases[0], canvases[1]))

        return pair_entries

    def _sample_arc_training_pair_entries(
        self,
        pair_entries: List[Tuple[int, np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> Tuple[List[Tuple[int, np.ndarray, np.ndarray]], int]:
        if not pair_entries:
            raise ValueError("ARC output masking found an empty sample with no valid pair entries.")

        min_context_pairs = self._arc_min_context_pairs()
        if min_context_pairs is None:
            selected_indices = list(range(len(pair_entries)))
            if len(selected_indices) > 1:
                rng.shuffle(selected_indices)
            target_selected_index = int(rng.integers(len(selected_indices)))
            return [pair_entries[idx] for idx in selected_indices], target_selected_index

        if len(pair_entries) < min_context_pairs + 1:
            raise ValueError(
                "ARC output masking requires at least min_context_pairs + 1 pair entries, "
                f"got {len(pair_entries)} pairs for min_context_pairs={min_context_pairs}."
            )

        target_index = int(rng.integers(len(pair_entries)))
        candidate_indices = [idx for idx in range(len(pair_entries)) if idx != target_index]
        num_context = int(rng.integers(min_context_pairs, len(candidate_indices) + 1))
        context_indices = rng.choice(candidate_indices, size=num_context, replace=False).tolist()
        selected_indices = [*context_indices, target_index]
        if len(selected_indices) > 1:
            rng.shuffle(selected_indices)

        target_selected_index = selected_indices.index(target_index)
        return [pair_entries[idx] for idx in selected_indices], target_selected_index

    def _build_arc_masked_variable_sample(
        self,
        pair_entries: List[Tuple[int, np.ndarray, np.ndarray]],
        target_pair_index: int,
        fill_token: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        answer_slot_size = self._arc_answer_slot_size()

        masked_chunks = []
        label_chunks = []
        source_chunks = []
        position_chunks = []
        max_pair_h = 0
        max_pair_w = 0

        for pair_pos, (_pair_id, inp_canvas, out_canvas) in enumerate(pair_entries):
            masked_chunks.append(inp_canvas.reshape(-1).astype(np.int32, copy=False))
            label_chunks.append(np.zeros((inp_canvas.size,), dtype=np.int32))
            source_chunks.append(inp_canvas.reshape(-1).astype(np.int32, copy=False))
            position_chunks.append(self._make_arc_position_ids(pair_pos, 0, inp_canvas.shape))
            max_pair_h = max(max_pair_h, inp_canvas.shape[0])
            max_pair_w = max(max_pair_w, inp_canvas.shape[1])

            if pair_pos == target_pair_index:
                if answer_slot_size is None:
                    masked_out = np.full(out_canvas.shape, fill_token, dtype=np.int32)
                    label_out = out_canvas.astype(np.int32, copy=False)
                    source_out = out_canvas.astype(np.int32, copy=False)
                else:
                    masked_out = np.full((answer_slot_size, answer_slot_size), fill_token, dtype=np.int32)
                    label_out = np.zeros((answer_slot_size, answer_slot_size), dtype=np.int32)
                    label_out[: out_canvas.shape[0], : out_canvas.shape[1]] = out_canvas
                    source_out = label_out
            else:
                masked_out = out_canvas.astype(np.int32, copy=False)
                label_out = np.zeros_like(masked_out)
                source_out = masked_out

            masked_chunks.append(masked_out.reshape(-1).astype(np.int32, copy=False))
            label_chunks.append(label_out.reshape(-1).astype(np.int32, copy=False))
            source_chunks.append(source_out.reshape(-1).astype(np.int32, copy=False))
            position_chunks.append(self._make_arc_position_ids(pair_pos, 1, masked_out.shape))
            max_pair_h = max(max_pair_h, masked_out.shape[0])
            max_pair_w = max(max_pair_w, masked_out.shape[1])

        inputs = np.concatenate(masked_chunks).astype(np.int32, copy=False)
        labels = np.concatenate(label_chunks).astype(np.int32, copy=False)
        source_inputs = np.concatenate(source_chunks).astype(np.int32, copy=False)
        position_ids = np.concatenate(position_chunks, axis=0).astype(np.int32, copy=False)
        seq_shape = (len(pair_entries), 2, max_pair_h, max_pair_w)
        return inputs, labels, source_inputs, position_ids, seq_shape

    def _apply_arc_output_mask_variable(
        self,
        batch: dict,
        rng: np.random.Generator,
    ) -> dict:
        if "position_ids" not in batch:
            raise ValueError("ARC output masking requires position_ids in the batch.")

        masked_samples = []
        label_samples = []
        source_samples = []
        position_samples = []
        seq_lengths = []
        seq_shapes = []

        seq_offsets = batch["seq_offsets"].astype(np.int32, copy=False)
        for sample_idx in range(batch["puzzle_identifiers"].shape[0]):
            start = int(seq_offsets[sample_idx])
            end = int(seq_offsets[sample_idx + 1])
            pair_entries = self._extract_arc_pair_entries(
                batch["inputs"][start:end],
                batch["position_ids"][start:end],
            )
            selected_pair_entries, target_pair_index = self._sample_arc_training_pair_entries(pair_entries, rng)
            fill_token = self._arc_mask_fill_token(rng)
            (
                sample_inputs,
                sample_labels,
                sample_source_inputs,
                sample_position_ids,
                seq_shape,
            ) = self._build_arc_masked_variable_sample(
                pair_entries=selected_pair_entries,
                target_pair_index=target_pair_index,
                fill_token=fill_token,
            )
            masked_samples.append(sample_inputs)
            label_samples.append(sample_labels)
            source_samples.append(sample_source_inputs)
            position_samples.append(sample_position_ids)
            seq_lengths.append(sample_inputs.shape[0])
            seq_shapes.append(seq_shape)

        batch["inputs"] = np.concatenate(masked_samples).astype(np.int32, copy=False)
        batch["labels"] = np.concatenate(label_samples).astype(np.int32, copy=False)
        cfg = self._arc_output_mask_cfg()
        if cfg is not None and cfg.preserve_source_inputs:
            batch["source_inputs"] = np.concatenate(source_samples).astype(np.int32, copy=False)
        batch["position_ids"] = np.concatenate(position_samples, axis=0).astype(np.int32, copy=False)
        batch["seq_lengths"] = np.array(seq_lengths, dtype=np.int32)
        batch["seq_offsets"] = np.concatenate(
            [np.zeros((1,), dtype=np.int32), np.cumsum(batch["seq_lengths"], dtype=np.int32)]
        )
        batch["seq_shapes"] = np.array(seq_shapes, dtype=np.int32)
        batch["label_seq_lengths"] = batch["seq_lengths"].copy()
        batch["label_seq_offsets"] = batch["seq_offsets"].copy()
        batch["label_seq_shapes"] = batch["seq_shapes"].copy()
        # _debug_print_arc_variable_batch(
        #     inputs=batch["inputs"],
        #     labels=batch["labels"],
        #     position_ids=batch["position_ids"],
        #     seq_offsets=batch["seq_offsets"],
        # )
        return batch

    def _apply_arc_output_mask_fixed(
        self,
        batch: dict,
        rng: np.random.Generator,
    ) -> dict:
        if "position_ids" not in batch:
            raise ValueError("ARC output masking requires position_ids in the batch.")

        source_inputs = batch["inputs"].copy()
        batch["labels"] = np.zeros_like(batch["inputs"])

        for sample_idx in range(batch["inputs"].shape[0]):
            sample_tokens = source_inputs[sample_idx]
            sample_position_ids = batch["position_ids"][sample_idx]
            pair_entries = self._extract_arc_pair_entries(sample_tokens, sample_position_ids)
            if not pair_entries:
                continue

            target_pair_id = int(pair_entries[int(rng.integers(len(pair_entries)))][0])
            target_mask = (
                (sample_position_ids[:, 0] == target_pair_id)
                & (sample_position_ids[:, 1] == 1)
            )
            fill_token = self._arc_mask_fill_token(rng)
            batch["inputs"][sample_idx, target_mask] = fill_token
            batch["labels"][sample_idx, target_mask] = source_inputs[sample_idx, target_mask]

        cfg = self._arc_output_mask_cfg()
        if cfg is not None and cfg.preserve_source_inputs:
            batch["source_inputs"] = source_inputs
        return batch

    def _apply_arc_output_mask(
        self,
        batch: dict,
        rng: np.random.Generator,
    ) -> dict:
        if self.metadata.variable_seq_lengths:
            return self._apply_arc_output_mask_variable(batch, rng)
        return self._apply_arc_output_mask_fixed(batch, rng)

    def _flatten_fixed_example_fields(self, batch: dict) -> dict:
        if self.metadata.variable_seq_lengths:
            return batch

        for key in ("inputs", "labels", "source_inputs"):
            if key in batch and batch[key].ndim > 2:
                batch[key] = batch[key].reshape(batch[key].shape[0], -1)

        if "position_ids" in batch and batch["position_ids"].ndim > 3:
            batch["position_ids"] = batch["position_ids"].reshape(
                batch["position_ids"].shape[0],
                -1,
                batch["position_ids"].shape[-1],
            )

        return batch

    def _collate_batch(self, batch, rng: np.random.Generator, make_masked_inputs: bool = True):
        batch = self._flatten_fixed_example_fields(batch)

        # Most batch fields are int32 tokens/positions, but puzzle identifiers need
        # int64 so compiled sparse embedding gathers use 64-bit indices.
        batch = {
            k: v.astype(np.int64 if k == "puzzle_identifiers" else np.int32, copy=True)
            for k, v in batch.items()
        }

        arc_output_mask_enabled = self._should_apply_arc_output_mask(make_masked_inputs)
        if arc_output_mask_enabled:
            if self.config.masked_input is not None and self.config.masked_input.enabled:
                raise ValueError("masked_input cannot be combined with arc_output_mask on the same dataset.")
            batch = self._apply_arc_output_mask(batch, rng)
        elif "labels" not in batch:
            raise ValueError(
                f"Dataset split '{self.split}' does not contain labels, and no dynamic label generation is enabled."
            )

        # Convert ignore label IDs
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        masked_input_cfg = self.config.masked_input
        if masked_input_cfg is not None and masked_input_cfg.enabled and make_masked_inputs:
            if self.metadata.variable_seq_lengths:
                raise ValueError("masked_input is not supported for packed variable-length datasets.")
            if masked_input_cfg.preserve_source_inputs:
                batch["source_inputs"] = batch["inputs"].copy()
            batch["inputs"] = self._make_masked_inputs(batch["inputs"], batch["labels"], rng)

        # Pad
        if not self.metadata.variable_seq_lengths and batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size

            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id
            }
            if "source_inputs" in batch:
                pad_values["source_inputs"] = self.metadata.pad_id
            if "seq_lengths" in batch:
                pad_values["seq_lengths"] = 0
            if "seq_shapes" in batch:
                pad_values["seq_shapes"] = 1
            if "position_ids" in batch:
                pad_values["position_ids"] = 0
            batch = {k: np.pad(v, ((0, pad_size), ) + ((0, 0), ) * (v.ndim - 1), constant_values=pad_values[k]) for k, v in batch.items()}

        if "position_ids" not in batch and "seq_lengths" in batch and "seq_shapes" in batch:
            seq_lengths = batch["seq_lengths"]
            seq_widths = np.maximum(batch["seq_shapes"][:, 1], 1)
            if self.metadata.variable_seq_lengths:
                position_chunks = []
                for length, width in zip(seq_lengths, seq_widths):
                    positions = np.arange(int(length), dtype=np.int32)
                    position_chunks.append(np.stack([positions // int(width), positions % int(width)], axis=-1))
                batch["position_ids"] = (
                    np.concatenate(position_chunks, axis=0).astype(np.int32, copy=False)
                    if position_chunks
                    else np.empty((0, 2), dtype=np.int32)
                )
            else:
                positions = np.arange(batch["inputs"].shape[1], dtype=np.int32)
                row_ids = positions[None, :] // seq_widths[:, None]
                col_ids = positions[None, :] % seq_widths[:, None]
                valid_positions = positions[None, :] < seq_lengths[:, None]
                batch["position_ids"] = np.stack(
                    [
                        np.where(valid_positions, row_ids, 0),
                        np.where(valid_positions, col_ids, 0),
                    ],
                    axis=-1,
                ).astype(np.int32, copy=False)
        if "seq_shapes" in batch:
            del batch["seq_shapes"]

        if not make_masked_inputs:
            batch = self._strip_casual_eval_targets(batch)

        # To tensor
        return {k: torch.from_numpy(v) for k, v in batch.items()}
    
    def _iter_test(self):
        rng = np.random.Generator(np.random.Philox(seed=self.config.seed + 10_000 + self.config.rank))
        for set_name, dataset in self._data.items():  # type: ignore
            total_examples = (
                dataset["seq_offsets"].size - 1
                if self.metadata.variable_seq_lengths
                else len(dataset["inputs"])
            )

            # Load examples one by one
            start_index = 0
            while start_index < total_examples:
                # Compute indices
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end   = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)
                
                # Get batch of examples, and also puzzle IDs
                puzzle_indices = []
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                for i in range(local_start, local_end):
                    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
                        puzzle_index += 1

                    puzzle_indices.append(puzzle_index)
                
                local_indices = np.arange(local_start, local_end, dtype=np.int64)
                batch_fields = self._select_examples(dataset, local_indices)
                batch_fields["puzzle_identifiers"] = dataset["puzzle_identifiers"][puzzle_indices]
                batch = self._collate_batch(batch_fields, rng, make_masked_inputs=False)

                yield set_name, batch, end_index - start_index
                
                # Advance to next batch
                start_index += self.config.global_batch_size

    def _iter_train(self):
        for set_name, dataset in self._data.items():  # type: ignore
            # Increase epoch count
            self._iters += 1

            # Randomly shuffle groups
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            num_groups = dataset["group_indices"].size - 1
            group_order = np.concatenate([rng.permutation(num_groups) for _i in range(self.config.epochs_per_iter)])
            start_index = 0
            grad_accum_steps = max(1, self.config.grad_accum_steps)
            sample_global_batch_size = self.config.global_batch_size * grad_accum_steps

            while start_index < group_order.size:
                start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=sample_global_batch_size,
                    data_fraction=self.config.data_fraction,
                    examples_per_puzzle=self.config.examples_per_puzzle,
                )

                # Select current rank and collate
                global_effective_batch_size = batch_puzzle_indices.size  # Global effective batch size, excluding pads

                # Drop last batch
                if global_effective_batch_size < sample_global_batch_size:
                    break

                for accum_index in range(grad_accum_steps):
                    global_start = accum_index * self.config.global_batch_size
                    local_start = global_start + self.config.rank * self.local_batch_size
                    local_end = global_start + (self.config.rank + 1) * self.local_batch_size

                    local_batch_indices = batch_indices[local_start:local_end]
                    local_batch_puzzle_indices = batch_puzzle_indices[local_start:local_end]
                    batch_fields = self._select_examples(dataset, local_batch_indices)
                    batch_fields["puzzle_identifiers"] = dataset["puzzle_identifiers"][local_batch_puzzle_indices]
                    batch = self._collate_batch(batch_fields, rng)

                    if self.config.online_aug is not None and self.config.online_aug.enabled:
                        if self.metadata.variable_seq_lengths:
                            raise ValueError("online_aug is not supported for variable-length datasets.")
                        batch = apply_online_aug(batch, self.metadata.seq_len, self.config.online_aug)

                    yield set_name, batch, self.config.global_batch_size
                
    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, "Multithreaded data loading is not currently supported."
        
        self._lazy_load_dataset()
        
        # Iterate using specified mode
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()
            
    def __len__(self):
        # データがロードされていなければロードする
        self._lazy_load_dataset()
        # 全セットの合計サイズを返す
        if self.metadata.variable_seq_lengths:
            return sum(d["seq_offsets"].size - 1 for d in self._data.values())
        return sum(len(d["inputs"]) for d in self._data.values())


class PuzzleDatasetSeparate(PuzzleDataset):
    """Append noised answer tokens after problem tokens.

    The model sees task embedding + problem tokens + answer tokens. Loss labels
    ignore the problem span and supervise only the answer span.
    """

    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__(config, split=split)
        self._base_seq_len = int(self.metadata.seq_len)
        self.metadata.seq_len = self._base_seq_len * 2
        self._separate_position_slot_axis = False
        self.metadata.sequence_layout = (
            "label_separate"
            if self.metadata.sequence_layout is None
            else f"{self.metadata.sequence_layout}+label_separate"
        )
        self.metadata.train_target_mode = "label_separate"

        position_id_shape = self.metadata.position_id_shape
        if position_id_shape is None and self.metadata.variable_seq_lengths:
            position_id_shape = self._infer_generated_position_id_shape()

        if position_id_shape is not None:
            if len(position_id_shape) >= 4:
                raise ValueError(
                    "PuzzleDatasetSeparate adds a problem/answer position axis and supports "
                    "at most 3 existing position axes."
                )
            self.metadata.position_id_shape = [2, *position_id_shape]
            self._separate_position_slot_axis = True

    def _infer_generated_position_id_shape(self) -> Optional[List[int]]:
        """Infer the 2D position shape used by PuzzleDataset._collate_batch.

        Variable-length ARC-style datasets often store seq_shapes but leave
        metadata.position_id_shape unset. The base collator still generates
        2D row/column position_ids from those seq_shapes; label_separate then
        adds a problem/answer slot axis. Exposing that inferred slot shape in
        metadata lets the model select 3D RoPE instead of silently treating the
        slot axis as the row axis.
        """
        side = math.isqrt(self._base_seq_len)
        if side * side == self._base_seq_len:
            return [side, side]
        return None

    def _label_separate_noise_bounds(self) -> Tuple[int, int]:
        token_min = int(self.config.label_separate_noise_token_min)
        token_max = self.config.label_separate_noise_token_max
        token_max = self.metadata.vocab_size - 1 if token_max is None else int(token_max)
        token_max = min(token_max, self.metadata.vocab_size - 1)
        if token_min > token_max:
            raise ValueError(
                "label_separate noise token range is empty after clipping to vocab_size: "
                f"min={token_min}, max={token_max}, vocab_size={self.metadata.vocab_size}."
            )
        return token_min, token_max

    def _separate_mode(self) -> str:
        return (self.config.separate_mode or self.config.SeparateMode).upper()

    @staticmethod
    def _torch_generator_from_numpy(rng: np.random.Generator, device: torch.device) -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(rng.integers(0, np.iinfo(np.int64).max)))
        return generator

    def _make_answer_tokens(
        self,
        labels: torch.Tensor,
        rng: np.random.Generator,
        *,
        training: bool,
        sample_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        valid_mask = labels != IGNORE_LABEL_ID
        token_min, token_max = self._label_separate_noise_bounds()
        generator = self._torch_generator_from_numpy(rng, labels.device)
        noise = torch.randint(
            low=token_min,
            high=token_max + 1,
            size=tuple(labels.shape),
            generator=generator,
            device=labels.device,
            dtype=labels.dtype,
        )

        if self._separate_mode() == "C":
            return torch.where(valid_mask, noise, torch.full_like(labels, self.metadata.pad_id))

        safe_labels = torch.where(valid_mask, labels, torch.full_like(labels, self.metadata.pad_id))
        if training:
            replace_ratio = torch.rand(
                sample_shape,
                generator=generator,
                device=labels.device,
                dtype=torch.float32,
            )
            while replace_ratio.ndim < labels.ndim:
                replace_ratio = replace_ratio.unsqueeze(-1)
            replace_with_noise = torch.rand(
                tuple(labels.shape),
                generator=generator,
                device=labels.device,
                dtype=torch.float32,
            ) < replace_ratio
            answer_tokens = torch.where(valid_mask & replace_with_noise, noise, safe_labels)
        else:
            answer_tokens = noise

        return torch.where(valid_mask, answer_tokens, torch.full_like(labels, self.metadata.pad_id))

    @staticmethod
    def _add_separate_slot_axis(position_ids: torch.Tensor, slot_id: int) -> torch.Tensor:
        slot_ids = torch.full(
            position_ids.shape[:-1] + (1,),
            slot_id,
            dtype=position_ids.dtype,
            device=position_ids.device,
        )
        return torch.cat([slot_ids, position_ids], dim=-1)

    def _maybe_add_separate_slot_axis(self, position_ids: torch.Tensor, slot_id: int) -> torch.Tensor:
        if not self._separate_position_slot_axis:
            return position_ids
        return self._add_separate_slot_axis(position_ids, slot_id)

    def _should_pad_variable_inputs_and_labels(
        self,
        dataset: dict,
        input_shapes: np.ndarray,
        label_shapes: np.ndarray,
    ) -> bool:
        return False

    @staticmethod
    def _answer_positions_from_shape(
        problem_positions: torch.Tensor,
        label_shape: torch.Tensor,
        label_length: int,
    ) -> torch.Tensor:
        width = (
            max(int(label_shape.reshape(-1)[-1].item()), 1)
            if label_shape.numel() >= 2
            else max(label_length, 1)
        )
        positions = torch.arange(
            label_length,
            device=problem_positions.device,
            dtype=problem_positions.dtype,
        )
        row_col = torch.stack([positions // width, positions % width], dim=-1)
        if problem_positions.shape[-1] == 2:
            return row_col

        leading = torch.zeros(
            (label_length, problem_positions.shape[-1] - 2),
            device=problem_positions.device,
            dtype=problem_positions.dtype,
        )
        return torch.cat([leading, row_col], dim=-1)

    def _separate_fixed_batch(
        self,
        batch: Dict[str, torch.Tensor],
        rng: np.random.Generator,
        *,
        training: bool,
    ) -> Dict[str, torch.Tensor]:
        inputs = batch["inputs"]
        labels = batch["labels"]
        answer_tokens = self._make_answer_tokens(
            labels,
            rng,
            training=training,
            sample_shape=tuple(labels.shape[:1]),
        )
        ignore_problem = torch.full_like(labels, IGNORE_LABEL_ID)
        valid_answer = labels != IGNORE_LABEL_ID
        problem_mask = torch.zeros_like(valid_answer, dtype=torch.bool)

        batch["inputs"] = torch.cat([inputs, answer_tokens], dim=1)
        batch["labels"] = torch.cat([ignore_problem, labels], dim=1)
        batch["answer_mask"] = torch.cat([problem_mask, valid_answer], dim=1)
        batch["source_inputs"] = batch.get("source_inputs", inputs)

        if "position_ids" in batch:
            batch["position_ids"] = torch.cat(
                [
                    self._maybe_add_separate_slot_axis(batch["position_ids"], 0),
                    self._maybe_add_separate_slot_axis(batch["position_ids"], 1),
                ],
                dim=1,
            )
        if "seq_lengths" in batch:
            label_lengths = batch.get("label_seq_lengths", batch["seq_lengths"])
            batch["seq_lengths"] = batch["seq_lengths"] + label_lengths
        if "label_seq_lengths" in batch:
            batch["label_seq_lengths"] = batch["seq_lengths"].clone()
        return batch

    def _separate_packed_batch(
        self,
        batch: Dict[str, torch.Tensor],
        rng: np.random.Generator,
        *,
        training: bool,
    ) -> Dict[str, torch.Tensor]:
        seq_offsets = batch["seq_offsets"].to(torch.long)
        label_offsets = batch.get("label_seq_offsets", batch["seq_offsets"]).to(torch.long)
        seq_lengths = batch["seq_lengths"].to(torch.long)
        label_lengths = batch.get("label_seq_lengths", batch["seq_lengths"]).to(torch.long)
        label_shapes = batch.get("label_seq_shapes")
        answer_only_labels = self.config.emits_answer_only_labels()

        input_chunks = []
        label_chunks = []
        answer_mask_chunks = []
        source_chunks = []
        position_chunks = [] if "position_ids" in batch else None

        for sample_idx in range(int(batch["puzzle_identifiers"].shape[0])):
            input_start = int(seq_offsets[sample_idx].item())
            input_end = int(seq_offsets[sample_idx + 1].item())
            label_start = int(label_offsets[sample_idx].item())
            label_end = int(label_offsets[sample_idx + 1].item())

            problem_tokens = batch["inputs"][input_start:input_end]
            true_labels = batch["labels"][label_start:label_end]
            answer_tokens = self._make_answer_tokens(
                true_labels,
                rng,
                training=training,
                sample_shape=(),
            )
            valid_answer = true_labels != IGNORE_LABEL_ID
            safe_labels = torch.where(
                valid_answer,
                true_labels,
                torch.full_like(true_labels, self.metadata.pad_id),
            )
            answer_span_mask = torch.ones_like(true_labels, dtype=torch.bool)

            input_chunks.extend([problem_tokens, answer_tokens])
            if answer_only_labels:
                label_chunks.append(true_labels)
            else:
                label_chunks.extend([torch.full_like(problem_tokens, IGNORE_LABEL_ID), true_labels])
            answer_mask_chunks.extend(
                [
                    torch.zeros_like(problem_tokens, dtype=torch.bool),
                    answer_span_mask if answer_only_labels else valid_answer,
                ]
            )
            source_chunks.extend([batch.get("source_inputs", batch["inputs"])[input_start:input_end], safe_labels])

            if position_chunks is not None:
                problem_positions = batch["position_ids"][input_start:input_end]
                if label_shapes is not None:
                    answer_positions = self._answer_positions_from_shape(
                        problem_positions,
                        label_shapes[sample_idx],
                        int(true_labels.shape[0]),
                    )
                elif true_labels.shape[0] == problem_positions.shape[0]:
                    answer_positions = problem_positions
                else:
                    answer_positions = torch.zeros(
                        (true_labels.shape[0], problem_positions.shape[-1]),
                        device=problem_positions.device,
                        dtype=problem_positions.dtype,
                    )
                    answer_positions[:, -1] = torch.arange(
                        true_labels.shape[0],
                        device=problem_positions.device,
                        dtype=problem_positions.dtype,
                    )
                position_chunks.extend(
                    [
                        self._maybe_add_separate_slot_axis(problem_positions, 0),
                        self._maybe_add_separate_slot_axis(answer_positions, 1),
                    ]
                )

        new_seq_lengths = (seq_lengths + label_lengths).to(torch.int32)
        new_label_lengths = label_lengths.to(torch.int32) if answer_only_labels else new_seq_lengths
        batch["inputs"] = torch.cat(input_chunks, dim=0) if input_chunks else batch["inputs"].new_empty((0,))
        batch["labels"] = torch.cat(label_chunks, dim=0) if label_chunks else batch["labels"].new_empty((0,))
        batch["answer_mask"] = (
            torch.cat(answer_mask_chunks, dim=0)
            if answer_mask_chunks
            else torch.empty((0,), dtype=torch.bool, device=batch["inputs"].device)
        )
        batch["source_inputs"] = (
            torch.cat(source_chunks, dim=0)
            if source_chunks
            else batch["inputs"].new_empty((0,))
        )
        if position_chunks is not None:
            batch["position_ids"] = (
                torch.cat(position_chunks, dim=0)
                if position_chunks
                else batch["position_ids"].new_empty((0, batch["position_ids"].shape[-1] + 1))
            )

        batch["seq_lengths"] = new_seq_lengths
        batch["seq_offsets"] = torch.cat(
            [
                torch.zeros((1,), dtype=batch["seq_offsets"].dtype, device=batch["seq_offsets"].device),
                torch.cumsum(new_seq_lengths.to(batch["seq_offsets"].dtype), dim=0),
            ]
        )
        batch["label_seq_lengths"] = new_label_lengths
        batch["label_seq_offsets"] = torch.cat(
            [
                torch.zeros((1,), dtype=batch["seq_offsets"].dtype, device=batch["seq_offsets"].device),
                torch.cumsum(new_label_lengths.to(batch["seq_offsets"].dtype), dim=0),
            ]
        )
        if "label_seq_shapes" in batch:
            del batch["label_seq_shapes"]
        return batch

    def _collate_batch(
        self,
        batch,
        rng: np.random.Generator,
        make_masked_inputs: bool = True,
    ):
        if self.config.masked_input is not None and self.config.masked_input.enabled and make_masked_inputs:
            raise ValueError("masked_input cannot be combined with label_separate.")
        if self._should_apply_arc_output_mask(make_masked_inputs):
            raise ValueError("arc_output_mask cannot be combined with label_separate.")

        collated = super()._collate_batch(batch, rng, make_masked_inputs=False)
        training = bool(make_masked_inputs and self.split == "train")
        if self.metadata.variable_seq_lengths:
            return self._separate_packed_batch(collated, rng, training=training)
        return self._separate_fixed_batch(collated, rng, training=training)
