import os
import json
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from models.losses import IGNORE_LABEL_ID
from data.common import PuzzleDatasetMetadata
from data.online_aug import OnlineAugConfig, apply_online_aug


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


def _sample_batch(rng: np.random.Generator, group_order: np.ndarray, puzzle_indices: np.ndarray, group_indices: np.ndarray, start_index: int, global_batch_size: int, data_fraction: float = 1.0):
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

        append_size = min(puzzle_size, global_batch_size - current_size)

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

    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead.

    rank: int
    num_replicas: int

    data_fraction: float = 1.0  # Fraction of training groups to use per epoch (1.0 = all)

    # Online augmentation applied at training time (None = disabled)
    online_aug: Optional[OnlineAugConfig] = None

    # Replace model inputs with randomly masked labels.
    masked_input: Optional[MaskedInputConfig] = None

    # ARC full-context training: mask one output pair on the fly and generate labels from it.
    arc_output_mask: Optional[ARCOutputMaskConfig] = None


class PuzzleDataset(IterableDataset):
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        if not os.path.isdir(os.path.join(config.dataset_path, split)):
            raise FileNotFoundError(f"Dataset split {split} in {config.dataset_path} does not exist.")
        
        self.config = config
        self.split = split
        self.metadata = self._load_metadata()
        
        # Checks
        assert self.config.global_batch_size % self.config.num_replicas == 0, f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State
        self._data = None
        self._iters = 0

    def _load_metadata(self) -> PuzzleDatasetMetadata:
        with open(os.path.join(self.config.dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

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
            position_ids_path = os.path.join(split_dir, f"{set_name}__position_ids.npy")
            if os.path.isfile(position_ids_path):
                set_fields["position_ids"] = "r"

            # Load subset
            self._data[set_name] = {
                field_name: np.load(os.path.join(split_dir, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                for field_name, mmap_mode in set_fields.items()
            }

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

        input_chunks = []
        position_chunks = []
        label_chunks = [] if "labels" in dataset else None
        for example_idx in indices:
            start = int(offsets[example_idx])
            end = int(offsets[example_idx + 1])
            input_chunks.append(dataset["inputs"][start:end])
            if label_chunks is not None:
                label_chunks.append(dataset["labels"][start:end])
            if "position_ids" in dataset:
                position_chunks.append(dataset["position_ids"][start:end])

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

            while start_index < group_order.size:
                start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=self.config.global_batch_size,
                    data_fraction=self.config.data_fraction,
                )

                # Select current rank and collate
                global_effective_batch_size = batch_puzzle_indices.size  # Global effective batch size, excluding pads

                # Drop last batch
                if global_effective_batch_size < self.config.global_batch_size:
                    break

                batch_indices        = batch_indices       [self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch_puzzle_indices = batch_puzzle_indices[self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch_fields = self._select_examples(dataset, batch_indices)
                batch_fields["puzzle_identifiers"] = dataset["puzzle_identifiers"][batch_puzzle_indices]
                batch = self._collate_batch(batch_fields, rng)

                if self.config.online_aug is not None and self.config.online_aug.enabled:
                    if self.metadata.variable_seq_lengths:
                        raise ValueError("online_aug is not supported for variable-length datasets.")
                    batch = apply_online_aug(batch, self.metadata.seq_len, self.config.online_aug)

                yield set_name, batch, global_effective_batch_size
                
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
