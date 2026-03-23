import os
import json
from typing import Optional

import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from models.losses import IGNORE_LABEL_ID
from data.common import PuzzleDatasetMetadata
from data.online_aug import OnlineAugConfig, apply_online_aug


class MaskedInputConfig(pydantic.BaseModel):
    enabled: bool = False
    min_mask_ratio: float = 0.0
    max_mask_ratio: float = 0.0
    mask_token_id: int = 1
    preserve_source_inputs: bool = True

    @pydantic.model_validator(mode="after")
    def _validate_ranges(self):
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
            "labels": "r",

            # Keep indices in memory
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None
        }

        # Load data
        self._data = {}
        for set_name in self.metadata.sets:
            # Load subset
            self._data[set_name] = {
                field_name: np.load(os.path.join(self.config.dataset_path, self.split, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                for field_name, mmap_mode in field_mmap_modes.items()
            }

    def _make_masked_inputs(self, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        cfg = self.config.masked_input
        assert cfg is not None

        masked_inputs = np.where(labels == IGNORE_LABEL_ID, self.metadata.pad_id, labels).astype(np.int32, copy=True)
        valid_mask = labels != IGNORE_LABEL_ID

        for row_idx in range(masked_inputs.shape[0]):
            valid_positions = np.flatnonzero(valid_mask[row_idx])
            if valid_positions.size == 0:
                continue

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

    def _collate_batch(self, batch, rng: np.random.Generator, make_masked_inputs: bool = True):
        # Convert dtype
        batch = {k: v.astype(np.int32, copy=True) for k, v in batch.items()}

        # Convert ignore label IDs
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        masked_input_cfg = self.config.masked_input
        if masked_input_cfg is not None and masked_input_cfg.enabled and make_masked_inputs:
            if masked_input_cfg.preserve_source_inputs:
                batch["source_inputs"] = batch["inputs"].copy()
            batch["inputs"] = self._make_masked_inputs(batch["labels"], rng)

        # Pad
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size

            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id
            }
            if "source_inputs" in batch:
                pad_values["source_inputs"] = self.metadata.pad_id
            batch = {k: np.pad(v, ((0, pad_size), ) + ((0, 0), ) * (v.ndim - 1), constant_values=pad_values[k]) for k, v in batch.items()}

        # To tensor
        return {k: torch.from_numpy(v) for k, v in batch.items()}
    
    def _iter_test(self):
        rng = np.random.Generator(np.random.Philox(seed=self.config.seed + 10_000 + self.config.rank))
        for set_name, dataset in self._data.items():  # type: ignore
            total_examples = len(dataset["inputs"])

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
                
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][local_start: local_end],
                    "labels": dataset["labels"][local_start: local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices]
                }, rng, make_masked_inputs=False)

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
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][batch_indices],
                    "labels": dataset["labels"][batch_indices],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices]
                }, rng)

                if self.config.online_aug is not None and self.config.online_aug.enabled:
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
        return sum(len(d["inputs"]) for d in self._data.values())
