import os
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from data.build_arc_dataset import _print_color_data
from models.losses import IGNORE_LABEL_ID
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


ARC_MAX_GRID_SIZE = 30
ARC_FULL_IO_COUNT = 2
FULL_DUMMY_PUZZLE_IDENTIFIER = 0


def _to_numpy_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _debug_print_first_full_batch_sample(batch: dict) -> None:
    seq_offsets = _to_numpy_array(batch["seq_offsets"])
    if seq_offsets.shape[0] < 2:
        print("batch data0: <empty>")
        return

    start = int(seq_offsets[0])
    end = int(seq_offsets[1])
    inputs = _to_numpy_array(batch["inputs"])[start:end]
    labels = _to_numpy_array(batch["labels"])[start:end].copy()
    position_ids = _to_numpy_array(batch["position_ids"])[start:end]

    labels[labels == IGNORE_LABEL_ID] = 0

    print("batch data0")
    _print_color_data(inputs, position_ids, title="batch[0] inputs")
    _print_color_data(labels, position_ids, title="batch[0] labels")


def _sample_batch(
    rng: np.random.Generator,
    group_order: np.ndarray,
    puzzle_indices: np.ndarray,
    group_indices: np.ndarray,
    start_index: int,
    global_batch_size: int,
    data_fraction: float = 1.0,
    min_pairs: int = 2,
    max_pairs: int = 8,
) -> Tuple[int, List[np.ndarray], List[np.ndarray]]:
    if min_pairs <= 0:
        raise ValueError(f"min_pairs must be positive, got {min_pairs}")
    if max_pairs < min_pairs:
        raise ValueError(f"max_pairs must be >= min_pairs, got min_pairs={min_pairs}, max_pairs={max_pairs}")

    batch_example_indices: List[np.ndarray] = []
    batch_puzzle_indices: List[np.ndarray] = []

    while (start_index < group_order.size) and (len(batch_example_indices) < global_batch_size):
        group_id = int(group_order[start_index])
        group_start = int(group_indices[group_id])
        group_end = int(group_indices[group_id + 1])
        group_end_limited = group_start + max(1, round((group_end - group_start) * data_fraction))
        group_end_limited = min(group_end_limited, group_end)
        available_puzzles = group_end_limited - group_start
        start_index += 1

        if available_puzzles <= 0:
            continue

        group_puzzle_ids = np.arange(group_start, group_end_limited, dtype=np.int64)
        puzzle_sizes = puzzle_indices[group_puzzle_ids + 1] - puzzle_indices[group_puzzle_ids]
        eligible_puzzles = group_puzzle_ids[puzzle_sizes >= min_pairs]
        if eligible_puzzles.size == 0:
            continue

        selected_puzzle = int(rng.choice(eligible_puzzles))
        puzzle_start = int(puzzle_indices[selected_puzzle])
        puzzle_end = int(puzzle_indices[selected_puzzle + 1])
        puzzle_size = puzzle_end - puzzle_start
        pair_count = int(rng.integers(min_pairs, min(max_pairs, puzzle_size) + 1))
        available_examples = np.arange(puzzle_start, puzzle_end, dtype=np.int64)
        selected_examples = np.empty((pair_count,), dtype=np.int64)
        for pair_idx in range(pair_count):
            selected_idx = int(rng.integers(available_examples.size))
            selected_examples[pair_idx] = available_examples[selected_idx]
            available_examples[selected_idx] = available_examples[-1]
            available_examples = available_examples[:-1]

        batch_example_indices.append(selected_examples)
        batch_puzzle_indices.append(np.array([selected_puzzle], dtype=np.int64))

    return start_index, batch_example_indices, batch_puzzle_indices


class PuzzleFullDataset(PuzzleDataset):
    """Build ARC full-context samples from one-pair ARC examples.

    Each output sample contains several input/output pairs from the same group.
    One pair's output is replaced by 0 in inputs and kept as labels; all other
    positions are ignored by the loss.
    """

    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__(config, split=split)
        self._validate_full_config()
        self.metadata.position_id_shape = [
            int(self.config.full_max_pairs),
            ARC_FULL_IO_COUNT,
            ARC_MAX_GRID_SIZE,
            ARC_MAX_GRID_SIZE,
        ]
        self.metadata.sequence_layout = "arc_full_pairs"
        self.metadata.train_target_mode = "masked_output_pair"
        self.metadata.min_context_pairs = int(self.config.full_min_pairs) - 1
        self.metadata.num_puzzle_identifiers = 1

    def _validate_full_config(self) -> None:
        if self.config.full_min_pairs is None or self.config.full_max_pairs is None:
            raise ValueError("PuzzleFullDataset requires full_min_pairs and full_max_pairs.")
        if self.config.full_min_pairs <= 0:
            raise ValueError(f"full_min_pairs must be positive, got {self.config.full_min_pairs}")
        if self.config.full_max_pairs < self.config.full_min_pairs:
            raise ValueError(
                "full_max_pairs must be >= full_min_pairs, "
                f"got full_min_pairs={self.config.full_min_pairs}, full_max_pairs={self.config.full_max_pairs}"
            )
        if not self.metadata.variable_seq_lengths:
            raise ValueError("PuzzleFullDataset expects variable-length ARC examples.")

    def _lazy_load_dataset(self):
        if self._data is None:
            field_mmap_modes = {
                "inputs": "r",
                "puzzle_identifiers": None,
                "puzzle_indices": None,
                "group_indices": None,
            }
            if self.metadata.variable_seq_lengths:
                field_mmap_modes["seq_offsets"] = None
                field_mmap_modes["seq_shapes"] = None

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

                self._data[set_name] = {
                    field_name: np.load(os.path.join(split_dir, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                    for field_name, mmap_mode in set_fields.items()
                }

            if self.split == "test":
                for set_name, dataset in self._data.items():
                    split_dir = os.path.join(self.config.dataset_path, self.split)
                    examples_path = os.path.join(split_dir, f"{set_name}__examples.npy")
                    example_shapes_path = os.path.join(split_dir, f"{set_name}__example_shapes.npy")
                    if not os.path.isfile(examples_path) or not os.path.isfile(example_shapes_path):
                        raise FileNotFoundError(
                            "PuzzleFullDataset test mode requires all__examples.npy and "
                            "all__example_shapes.npy generated by data/build_arc_dataset.py."
                        )
                    dataset["examples"] = np.load(examples_path, allow_pickle=True)
                    dataset["example_shapes"] = np.load(example_shapes_path, allow_pickle=True)

    @staticmethod
    def _make_position_ids(pair_id: int, io_id: int, shape: Tuple[int, int]) -> np.ndarray:
        height, width = shape
        row_col_ids = np.moveaxis(np.indices((height, width), dtype=np.int32), 0, -1).reshape(-1, 2)
        pair_ids = np.full((row_col_ids.shape[0], 1), pair_id, dtype=np.int32)
        io_ids = np.full((row_col_ids.shape[0], 1), io_id, dtype=np.int32)
        return np.concatenate([pair_ids, io_ids, row_col_ids], axis=-1)

    def _read_example_array(self, dataset: dict, field_name: str, example_index: int) -> np.ndarray:
        offsets = dataset.get("label_seq_offsets", dataset["seq_offsets"]) if field_name == "labels" else dataset["seq_offsets"]
        start = int(offsets[example_index])
        end = int(offsets[example_index + 1])
        return dataset[field_name][start:end].astype(np.int32, copy=False)

    def _make_answer_initial_tokens(
        self,
        solution: np.ndarray,
    ) -> np.ndarray:
        if self.config.full_answer_initial_mode not in {"black", "noised_label"}:
            raise ValueError(f"Unknown full_answer_initial_mode: {self.config.full_answer_initial_mode}")
        return np.full_like(solution, int(self.config.full_answer_initial_black_token_id))

    def _sample_answer_initial_gamma(self, rng: Optional[np.random.Generator]) -> Optional[np.ndarray]:
        if self.config.full_answer_initial_mode != "noised_label":
            return None
        if self.split != "train":
            gamma = 0.0
        else:
            if rng is None:
                raise ValueError("noised_label answer initialization requires an rng.")
            distribution = self.config.full_answer_initial_gamma_distribution
            if distribution == "uniform":
                gamma_min = float(self.config.full_answer_initial_gamma_min)
                gamma_max = float(self.config.full_answer_initial_gamma_max)
                gamma = gamma_min if gamma_min == gamma_max else float(rng.uniform(gamma_min, gamma_max))
            elif distribution == "logistic_normal":
                log_snr = float(
                    rng.normal(
                        loc=float(self.config.full_answer_initial_log_snr_mean),
                        scale=float(self.config.full_answer_initial_log_snr_std),
                    )
                )
                gamma = 1.0 / (1.0 + math.exp(-0.5 * log_snr))
            else:
                raise ValueError(f"Unknown full_answer_initial_gamma_distribution: {distribution}")
        return np.array(gamma, dtype=np.float32)

    def _build_pairs_sample(
        self,
        pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
        shapes: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]],
        target_pair_index: int,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        if not pairs:
            raise ValueError("Cannot build a full sample from zero pairs.")
        if len(pairs) != len(shapes):
            raise ValueError(f"pairs and shapes length mismatch: {len(pairs)} != {len(shapes)}")

        answer_only_labels = bool(self.config.answer_only_labels)

        input_chunks = []
        label_chunks = []
        answer_mask_chunks = []
        source_chunks = []
        position_chunks = []
        label_seq_shape = None

        for pair_pos, ((problem, solution), shape_pair) in enumerate(zip(pairs, shapes)):
            input_shape = tuple(int(v) for v in shape_pair[0])
            label_shape = tuple(int(v) for v in shape_pair[1])
            problem = problem.astype(np.int32, copy=False)
            solution = solution.astype(np.int32, copy=False)

            expected_input_size = input_shape[0] * input_shape[1]
            if problem.size != expected_input_size:
                raise ValueError(
                    "input shape does not match pair length, "
                    f"got shape={input_shape} length={problem.size} for pair {pair_pos}."
                )
            expected_label_size = label_shape[0] * label_shape[1]
            if solution.size != expected_label_size:
                raise ValueError(
                    "label shape does not match pair length, "
                    f"got shape={label_shape} length={solution.size} for pair {pair_pos}."
                )

            input_chunks.append(problem)
            if not answer_only_labels:
                label_chunks.append(np.full_like(problem, IGNORE_LABEL_ID))
            answer_mask_chunks.append(np.zeros(problem.shape, dtype=np.bool_))
            source_chunks.append(problem)
            position_chunks.append(self._make_position_ids(pair_pos, 0, input_shape))

            if pair_pos == target_pair_index:
                input_solution = self._make_answer_initial_tokens(solution)
                label_solution = solution
                answer_mask = np.ones(solution.shape, dtype=np.bool_)
                label_seq_shape = label_shape
            else:
                input_solution = solution
                label_solution = np.full_like(solution, IGNORE_LABEL_ID)
                answer_mask = np.zeros(solution.shape, dtype=np.bool_)

            input_chunks.append(input_solution)
            if answer_only_labels:
                if pair_pos == target_pair_index:
                    label_chunks.append(label_solution)
            else:
                label_chunks.append(label_solution)
            answer_mask_chunks.append(answer_mask)
            source_chunks.append(solution)
            position_chunks.append(self._make_position_ids(pair_pos, 1, label_shape))

        inputs = np.concatenate(input_chunks).astype(np.int32, copy=False)
        labels = np.concatenate(label_chunks).astype(np.int32, copy=False)
        answer_mask = np.concatenate(answer_mask_chunks).astype(np.bool_, copy=False)
        source_inputs = np.concatenate(source_chunks).astype(np.int32, copy=False)
        position_ids = np.concatenate(position_chunks, axis=0).astype(np.int32, copy=False)

        sample = {
            "inputs": inputs,
            "labels": labels,
            "answer_mask": answer_mask,
            "source_inputs": source_inputs,
            "position_ids": position_ids,
            "seq_lengths": np.array(inputs.shape[0], dtype=np.int32),
        }
        answer_initial_gamma = self._sample_answer_initial_gamma(rng)
        if answer_initial_gamma is not None:
            sample["answer_initial_gamma"] = answer_initial_gamma
        if answer_only_labels:
            if label_seq_shape is None:
                raise ValueError("target_pair_index did not match any pair while building answer-only labels.")
            sample["label_seq_lengths"] = np.array(labels.shape[0], dtype=np.int32)
            sample["label_seq_shapes"] = np.array(label_seq_shape, dtype=np.int32)
        return sample

    def _build_full_sample(
        self,
        dataset: dict,
        example_indices: np.ndarray,
        rng: np.random.Generator,
        target_pair_index: Optional[int] = None,
    ) -> dict:
        if example_indices.size == 0:
            raise ValueError("Cannot build a full sample from zero examples.")
        if "labels" not in dataset:
            raise ValueError("PuzzleFullDataset requires labels for input/output pair construction.")

        if target_pair_index is None:
            target_pair_index = int(rng.integers(example_indices.size))

        pairs = []
        shapes = []
        seq_shapes = dataset["seq_shapes"][example_indices].astype(np.int32, copy=False)
        label_seq_shapes = dataset.get("label_seq_shapes", dataset["seq_shapes"])[example_indices].astype(np.int32, copy=False)

        for pair_pos, example_index in enumerate(example_indices.astype(np.int64, copy=False)):
            input_shape = tuple(int(v) for v in seq_shapes[pair_pos])
            label_shape = tuple(int(v) for v in label_seq_shapes[pair_pos])
            problem = self._read_example_array(dataset, "inputs", int(example_index))
            solution = self._read_example_array(dataset, "labels", int(example_index))
            pairs.append((problem, solution))
            shapes.append((input_shape, label_shape))

        return self._build_pairs_sample(pairs, shapes, target_pair_index, rng=rng)

    def _context_pairs_for_test_example(self, dataset: dict, example_index: int):
        context_examples = dataset["examples"][example_index]
        context_shapes = dataset["example_shapes"][example_index]

        pairs = []
        shapes = []
        for pair_index in range(len(context_examples)):
            problem = np.asarray(context_examples[pair_index][0], dtype=np.int32)
            solution = np.asarray(context_examples[pair_index][1], dtype=np.int32)
            shape_array = np.asarray(context_shapes[pair_index], dtype=np.int32)
            if shape_array.shape == (2, 2):
                input_shape = tuple(int(v) for v in shape_array[0].tolist())
                label_shape = tuple(int(v) for v in shape_array[1].tolist())
            else:
                input_shape = tuple(int(v) for v in shape_array.tolist())
                label_shape = input_shape
            pairs.append((problem, solution))
            shapes.append((input_shape, label_shape))

        return pairs, shapes

    def _build_test_sample(
        self,
        dataset: dict,
        example_index: int,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        pairs, shapes = self._context_pairs_for_test_example(dataset, example_index)
        query_problem = self._read_example_array(dataset, "inputs", example_index)
        query_solution = self._read_example_array(dataset, "labels", example_index)
        query_input_shape = tuple(int(v) for v in dataset["seq_shapes"][example_index])
        query_label_shape = tuple(
            int(v)
            for v in dataset.get("label_seq_shapes", dataset["seq_shapes"])[example_index]
        )

        pairs.append((query_problem, query_solution))
        shapes.append((query_input_shape, query_label_shape))
        return self._build_pairs_sample(pairs, shapes, target_pair_index=len(pairs) - 1, rng=rng)

    def _collate_full_samples(
        self,
        dataset: dict,
        sample_example_indices: List[np.ndarray],
        sample_puzzle_indices: List[np.ndarray],
        rng: np.random.Generator,
    ) -> dict:
        samples = [
            self._build_full_sample(dataset, example_indices, rng)
            for example_indices in sample_example_indices
        ]

        batch = {
            "inputs": np.concatenate([sample["inputs"] for sample in samples]).astype(np.int32, copy=False),
            "labels": np.concatenate([sample["labels"] for sample in samples]).astype(np.int32, copy=False),
            "answer_mask": np.concatenate([sample["answer_mask"] for sample in samples]).astype(np.bool_, copy=False),
            "source_inputs": np.concatenate([sample["source_inputs"] for sample in samples]).astype(np.int32, copy=False),
            "position_ids": np.concatenate([sample["position_ids"] for sample in samples], axis=0).astype(np.int32, copy=False),
            "seq_lengths": np.array([int(sample["seq_lengths"]) for sample in samples], dtype=np.int32),
            "puzzle_identifiers": np.full(
                len(samples),
                FULL_DUMMY_PUZZLE_IDENTIFIER,
                dtype=np.int64,
            ),
        }
        batch["seq_offsets"] = np.concatenate(
            [np.zeros((1,), dtype=np.int32), np.cumsum(batch["seq_lengths"], dtype=np.int32)]
        )
        if "label_seq_lengths" in samples[0]:
            batch["label_seq_lengths"] = np.array(
                [int(sample["label_seq_lengths"]) for sample in samples],
                dtype=np.int32,
            )
            batch["label_seq_offsets"] = np.concatenate(
                [np.zeros((1,), dtype=np.int32), np.cumsum(batch["label_seq_lengths"], dtype=np.int32)]
            )
            batch["label_seq_shapes"] = np.stack(
                [sample["label_seq_shapes"] for sample in samples],
                axis=0,
            ).astype(np.int32, copy=False)
        if "answer_initial_gamma" in samples[0]:
            batch["answer_initial_gamma"] = np.array(
                [float(sample["answer_initial_gamma"]) for sample in samples],
                dtype=np.float32,
            )

        return {k: torch.from_numpy(v) for k, v in batch.items()}

    def _collate_built_samples(self, samples: List[dict], puzzle_identifiers: np.ndarray) -> dict:
        batch = {
            "inputs": np.concatenate([sample["inputs"] for sample in samples]).astype(np.int32, copy=False),
            "labels": np.concatenate([sample["labels"] for sample in samples]).astype(np.int32, copy=False),
            "answer_mask": np.concatenate([sample["answer_mask"] for sample in samples]).astype(np.bool_, copy=False),
            "source_inputs": np.concatenate([sample["source_inputs"] for sample in samples]).astype(np.int32, copy=False),
            "position_ids": np.concatenate([sample["position_ids"] for sample in samples], axis=0).astype(np.int32, copy=False),
            "seq_lengths": np.array([int(sample["seq_lengths"]) for sample in samples], dtype=np.int32),
            "puzzle_identifiers": np.full(
                len(samples),
                FULL_DUMMY_PUZZLE_IDENTIFIER,
                dtype=np.int64,
            ),
            "arc_identifiers": puzzle_identifiers.astype(np.int64, copy=False),
        }
        batch["seq_offsets"] = np.concatenate(
            [np.zeros((1,), dtype=np.int32), np.cumsum(batch["seq_lengths"], dtype=np.int32)]
        )
        if "label_seq_lengths" in samples[0]:
            batch["label_seq_lengths"] = np.array(
                [int(sample["label_seq_lengths"]) for sample in samples],
                dtype=np.int32,
            )
            batch["label_seq_offsets"] = np.concatenate(
                [np.zeros((1,), dtype=np.int32), np.cumsum(batch["label_seq_lengths"], dtype=np.int32)]
            )
            batch["label_seq_shapes"] = np.stack(
                [sample["label_seq_shapes"] for sample in samples],
                axis=0,
            ).astype(np.int32, copy=False)
        if "answer_initial_gamma" in samples[0]:
            batch["answer_initial_gamma"] = np.array(
                [float(sample["answer_initial_gamma"]) for sample in samples],
                dtype=np.float32,
            )

        return {k: torch.from_numpy(v) for k, v in batch.items()}

    def _iter_train(self):
        for set_name, dataset in self._data.items():  # type: ignore
            self._iters += 1
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            num_groups = dataset["group_indices"].size - 1
            group_order = np.concatenate([rng.permutation(num_groups) for _i in range(self.config.epochs_per_iter)])
            start_index = 0
            grad_accum_steps = max(1, self.config.grad_accum_steps)
            sample_global_batch_size = self.config.global_batch_size * grad_accum_steps

            while start_index < group_order.size:
                start_index, batch_example_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=sample_global_batch_size,
                    data_fraction=self.config.data_fraction,
                    min_pairs=int(self.config.full_min_pairs),
                    max_pairs=int(self.config.full_max_pairs),
                )
                
                if len(batch_example_indices) < sample_global_batch_size:
                    break

                for accum_index in range(grad_accum_steps):
                    global_start = accum_index * self.config.global_batch_size
                    local_start = global_start + self.config.rank * self.local_batch_size
                    local_end = global_start + (self.config.rank + 1) * self.local_batch_size

                    batch = self._collate_full_samples(
                        dataset,
                        batch_example_indices[local_start:local_end],
                        batch_puzzle_indices[local_start:local_end],
                        rng,
                    )
                    #_debug_print_first_full_batch_sample(batch)
                    yield set_name, batch, self.config.global_batch_size

    def _iter_test(self):
        rng = np.random.Generator(np.random.Philox(seed=self.config.seed + 10_000 + self.config.rank))

        for set_name, dataset in self._data.items():  # type: ignore
            start_index = 0
            total_examples = dataset["seq_offsets"].size - 1

            while start_index < total_examples:
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)
                if local_start >= local_end:
                    break

                local_indices = np.arange(local_start, local_end, dtype=np.int64)
                puzzle_indices = np.searchsorted(dataset["puzzle_indices"], local_indices, side="right") - 1
                samples = [self._build_test_sample(dataset, int(example_index), rng=rng) for example_index in local_indices]
                batch = self._collate_built_samples(
                    samples,
                    dataset["puzzle_identifiers"][puzzle_indices],
                )
                #_debug_print_first_full_batch_sample(batch)
                yield set_name, batch, end_index - start_index
                start_index += self.config.global_batch_size

    def __len__(self):
        self._lazy_load_dataset()
        if self.config.test_set_mode:
            total_examples = sum(d["seq_offsets"].size - 1 for d in self._data.values())
            return math.ceil(total_examples / self.config.global_batch_size)
        return sum(d["group_indices"].size - 1 for d in self._data.values())
