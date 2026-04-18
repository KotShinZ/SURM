from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if "argdantic" not in sys.modules:
    argdantic = types.ModuleType("argdantic")

    class ArgParser:
        def command(self, singleton: bool = False):
            def _decorator(fn):
                return fn

            return _decorator

    argdantic.ArgParser = ArgParser
    sys.modules["argdantic"] = argdantic


from data.build_arc_dataset_full import DataProcessConfig, convert_dataset  # noqa: E402
from models.losses import IGNORE_LABEL_ID  # noqa: E402
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig  # noqa: E402


def _encode_fixed_canvas(grid: list[list[int]]) -> np.ndarray:
    arr = np.array(grid, dtype=np.uint8)
    canvas = np.zeros((30, 30), dtype=np.uint8)
    h, w = arr.shape
    canvas[:h, :w] = arr + 2
    if h < 30:
        canvas[h, :w] = 1
    if w < 30:
        canvas[:h, w] = 1
    return canvas


def _encode_unpadded_canvas(grid: list[list[int]], canvas_shape: tuple[int, int]) -> np.ndarray:
    arr = np.array(grid, dtype=np.uint8)
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    h, w = arr.shape
    canvas[:h, :w] = arr + 2
    if h < canvas_shape[0]:
        canvas[h, :w] = 1
    if w < canvas_shape[1]:
        canvas[:h, w] = 1
    return canvas


def _encode_unpadded_canvas_no_eos(grid: list[list[int]], canvas_shape: tuple[int, int]) -> np.ndarray:
    arr = np.array(grid, dtype=np.uint8)
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    h, w = arr.shape
    canvas[:h, :w] = arr + 2
    return canvas


def _encode_fixed_canvas_no_eos(grid: list[list[int]]) -> np.ndarray:
    arr = np.array(grid, dtype=np.uint8)
    canvas = np.zeros((30, 30), dtype=np.uint8)
    h, w = arr.shape
    canvas[:h, :w] = arr + 2
    return canvas


def _masked_answer_canvas(fill_token: int = 2) -> np.ndarray:
    return np.full((30, 30), fill_token, dtype=np.uint8)


def _reconstruct_pair_canvas(
    flat_tokens: np.ndarray,
    flat_position_ids: np.ndarray,
    pair_index: int,
    io_index: int,
) -> np.ndarray:
    mask = (flat_position_ids[:, 0] == pair_index) & (flat_position_ids[:, 1] == io_index)
    rows = flat_position_ids[mask, 2]
    cols = flat_position_ids[mask, 3]
    canvas = np.zeros((int(rows.max()) + 1, int(cols.max()) + 1), dtype=np.uint8)
    canvas[rows, cols] = flat_tokens[mask]
    return canvas


def _pad_into_canvas(canvas: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    padded = np.zeros(target_shape, dtype=canvas.dtype)
    padded[: canvas.shape[0], : canvas.shape[1]] = canvas
    return padded


class BuildARCDatasetFullTests(unittest.TestCase):
    def _write_subset(
        self,
        prefix: Path,
        subset_name: str,
        challenge_puzzles: dict,
        solution_puzzles: dict,
    ) -> None:
        with open(prefix.parent / f"{prefix.name}_{subset_name}-challenges.json", "w", encoding="utf-8") as f:
            json.dump(challenge_puzzles, f)
        with open(prefix.parent / f"{prefix.name}_{subset_name}-solutions.json", "w", encoding="utf-8") as f:
            json.dump(solution_puzzles, f)

    def test_build_dataset_writes_13_by_2_arc_layout_and_dataset_flattens_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_prefix = tmp_path / "arc"
            output_dir = tmp_path / "dataset"

            mini_puzzle = {
                "p_test": {
                    "train": [
                        {"input": [[1]], "output": [[2]]},
                        {"input": [[3, 3]], "output": [[4, 4]]},
                    ],
                    "test": [
                        {"input": [[5], [5]]},
                    ],
                }
            }
            aux_puzzle = {
                "p_train": {
                    "train": [
                        {"input": [[1, 0]], "output": [[0, 1]]},
                        {"input": [[2], [2]], "output": [[3], [3]]},
                    ],
                    "test": [
                        {"input": [[4, 4], [0, 4]]},
                    ],
                }
            }

            self._write_subset(
                input_prefix,
                "mini",
                mini_puzzle,
                {"p_test": [[[6], [6]]]},
            )
            self._write_subset(
                input_prefix,
                "aux",
                aux_puzzle,
                {"p_train": [[[5, 5], [0, 5]]]},
            )

            convert_dataset(
                DataProcessConfig(
                    input_file_prefix=str(input_prefix),
                    output_dir=str(output_dir),
                    subsets=["mini", "aux"],
                    test_set_name="mini",
                    seed=0,
                    num_aug=0,
                    no_padding=False,
                    no_padding_mode="sample",
                    min_context_pairs=2,
                )
            )

            test_inputs = np.load(output_dir / "test" / "all__inputs.npy")
            test_labels = np.load(output_dir / "test" / "all__labels.npy")
            test_position_ids = np.load(output_dir / "test" / "all__position_ids.npy")
            with open(output_dir / "test" / "dataset.json", encoding="utf-8") as f:
                test_metadata = json.load(f)

            self.assertEqual(test_inputs.shape, (1, 13, 2, 30, 30))
            self.assertEqual(test_labels.shape, (1, 13, 2, 30, 30))
            self.assertEqual(test_position_ids.shape, (1, 13, 2, 30, 30, 4))
            self.assertEqual(test_metadata["position_id_shape"], [13, 2, 30, 30])
            self.assertEqual(test_metadata["seq_len"], 13 * 2 * 30 * 30)
            self.assertEqual(test_metadata["answer_slot_max_grid_size"], 30)
            self.assertIsNone(test_metadata["train_target_mode"])

            target_slot = 2
            expected_query_input = _encode_fixed_canvas([[5], [5]])
            expected_query_mask = _masked_answer_canvas()
            expected_query_output = _encode_fixed_canvas([[6], [6]])

            np.testing.assert_array_equal(test_inputs[0, target_slot, 0], expected_query_input)
            np.testing.assert_array_equal(test_inputs[0, target_slot, 1], expected_query_mask)
            np.testing.assert_array_equal(test_labels[0, target_slot, 1], expected_query_output)

            self.assertEqual(int(np.count_nonzero(test_labels[0, :target_slot])), 0)
            self.assertEqual(int(np.count_nonzero(test_labels[0, target_slot, 0])), 0)
            self.assertEqual(int(np.count_nonzero(test_labels[0, target_slot + 1 :])), 0)
            self.assertGreater(int(np.count_nonzero(test_labels[0, target_slot, 1])), 0)

            np.testing.assert_array_equal(
                test_position_ids[0, 0, 0, 0, 0],
                np.array([0, 0, 0, 0], dtype=np.uint8),
            )
            np.testing.assert_array_equal(
                test_position_ids[0, 12, 1, 29, 29],
                np.array([12, 1, 29, 29], dtype=np.uint8),
            )

            dataset = PuzzleDataset(
                PuzzleDatasetConfig(
                    seed=3,
                    dataset_path=str(output_dir),
                    global_batch_size=1,
                    test_set_mode=True,
                    epochs_per_iter=1,
                    rank=0,
                    num_replicas=1,
                ),
                split="test",
            )
            set_name, batch, effective_batch_size = next(iter(dataset))

            self.assertEqual(set_name, "all")
            self.assertEqual(effective_batch_size, 1)
            self.assertEqual(tuple(batch["inputs"].shape), (1, 13 * 2 * 30 * 30))
            self.assertEqual(tuple(batch["labels"].shape), (1, 13 * 2 * 30 * 30))
            self.assertEqual(tuple(batch["position_ids"].shape), (1, 13 * 2 * 30 * 30, 4))

            reshaped_inputs = batch["inputs"].numpy().reshape(1, 13, 2, 30, 30)
            reshaped_position_ids = batch["position_ids"].numpy().reshape(1, 13, 2, 30, 30, 4)
            np.testing.assert_array_equal(reshaped_inputs[0, target_slot, 1], expected_query_mask.astype(np.int32))
            np.testing.assert_array_equal(
                reshaped_position_ids[0, 12, 1, 29, 29],
                np.array([12, 1, 29, 29], dtype=np.int32),
            )

    def test_build_dataset_uses_variable_pair_axis_in_no_padding_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_prefix = tmp_path / "arc"
            output_dir = tmp_path / "dataset"

            mini_puzzle = {
                "p_test": {
                    "train": [
                        {"input": [[1]], "output": [[2]]},
                        {"input": [[3, 3]], "output": [[4, 4]]},
                    ],
                    "test": [
                        {"input": [[5], [5]]},
                    ],
                }
            }

            self._write_subset(
                input_prefix,
                "mini",
                mini_puzzle,
                {"p_test": [[[6], [6]]]},
            )

            convert_dataset(
                DataProcessConfig(
                    input_file_prefix=str(input_prefix),
                    output_dir=str(output_dir),
                    subsets=["mini"],
                    test_set_name="mini",
                    seed=0,
                    num_aug=0,
                    no_padding=True,
                    no_padding_mode="sample",
                    min_context_pairs=2,
                )
            )

            test_inputs = np.load(output_dir / "test" / "all__inputs.npy")
            test_labels = np.load(output_dir / "test" / "all__labels.npy")
            test_position_ids = np.load(output_dir / "test" / "all__position_ids.npy")
            test_seq_offsets = np.load(output_dir / "test" / "all__seq_offsets.npy")
            test_seq_shapes = np.load(output_dir / "test" / "all__seq_shapes.npy")
            with open(output_dir / "test" / "dataset.json", encoding="utf-8") as f:
                test_metadata = json.load(f)

            self.assertEqual(test_inputs.shape, (5400,))
            self.assertEqual(test_labels.shape, (5400,))
            self.assertEqual(test_position_ids.shape, (5400, 4))
            np.testing.assert_array_equal(test_seq_offsets, np.array([0, 5400], dtype=np.int64))
            np.testing.assert_array_equal(test_seq_shapes, np.array([[3, 2, 30, 30]], dtype=np.int32))
            self.assertTrue(test_metadata["variable_seq_lengths"])
            self.assertEqual(test_metadata["position_id_shape"], [3, 2, 30, 30])
            self.assertEqual(test_metadata["seq_len"], 3 * 2 * 30 * 30)
            self.assertEqual(test_metadata["sequence_layout"], "sample")

            reshaped_inputs = test_inputs.reshape(3, 2, 30, 30)
            reshaped_labels = test_labels.reshape(3, 2, 30, 30)
            reshaped_position_ids = test_position_ids.reshape(3, 2, 30, 30, 4)

            expected_query_input = _pad_into_canvas(_encode_unpadded_canvas([[5], [5]], (3, 2)), (30, 30))
            expected_query_output = _encode_fixed_canvas([[6], [6]])
            expected_query_mask = _masked_answer_canvas()

            np.testing.assert_array_equal(reshaped_inputs[2, 0], expected_query_input)
            np.testing.assert_array_equal(reshaped_inputs[2, 1], expected_query_mask)
            np.testing.assert_array_equal(reshaped_labels[2, 1], expected_query_output)
            self.assertEqual(int(np.count_nonzero(reshaped_labels[:2])), 0)
            self.assertEqual(int(np.count_nonzero(reshaped_labels[2, 0])), 0)
            np.testing.assert_array_equal(
                reshaped_position_ids[2, 1, 29, 29],
                np.array([2, 1, 29, 29], dtype=np.uint8),
            )

            dataset = PuzzleDataset(
                PuzzleDatasetConfig(
                    seed=3,
                    dataset_path=str(output_dir),
                    global_batch_size=1,
                    test_set_mode=True,
                    epochs_per_iter=1,
                    rank=0,
                    num_replicas=1,
                ),
                split="test",
            )
            set_name, batch, effective_batch_size = next(iter(dataset))

            self.assertEqual(set_name, "all")
            self.assertEqual(effective_batch_size, 1)
            self.assertEqual(tuple(batch["inputs"].shape), (5400,))
            self.assertEqual(tuple(batch["labels"].shape), (5400,))
            self.assertEqual(tuple(batch["position_ids"].shape), (5400, 4))
            np.testing.assert_array_equal(batch["seq_lengths"].numpy(), np.array([5400], dtype=np.int32))
            np.testing.assert_array_equal(batch["seq_offsets"].numpy(), np.array([0, 5400], dtype=np.int32))

    def test_build_dataset_supports_pair_eos_layout_without_sample_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_prefix = tmp_path / "arc"
            output_dir = tmp_path / "dataset"

            mini_puzzle = {
                "p_test": {
                    "train": [
                        {"input": [[1]], "output": [[2]]},
                        {"input": [[3, 3]], "output": [[4, 4]]},
                    ],
                    "test": [
                        {"input": [[5], [5]]},
                    ],
                }
            }

            self._write_subset(
                input_prefix,
                "mini",
                mini_puzzle,
                {"p_test": [[[6], [6]]]},
            )

            convert_dataset(
                DataProcessConfig(
                    input_file_prefix=str(input_prefix),
                    output_dir=str(output_dir),
                    subsets=["mini"],
                    test_set_name="mini",
                    seed=0,
                    num_aug=0,
                    no_padding=True,
                    no_padding_mode="pair_eos",
                    min_context_pairs=2,
                )
            )

            test_inputs = np.load(output_dir / "test" / "all__inputs.npy")
            test_labels = np.load(output_dir / "test" / "all__labels.npy")
            test_position_ids = np.load(output_dir / "test" / "all__position_ids.npy")
            test_seq_offsets = np.load(output_dir / "test" / "all__seq_offsets.npy")
            test_seq_shapes = np.load(output_dir / "test" / "all__seq_shapes.npy")
            with open(output_dir / "test" / "dataset.json", encoding="utf-8") as f:
                test_metadata = json.load(f)

            self.assertEqual(test_inputs.shape, (926,))
            self.assertEqual(test_labels.shape, (926,))
            self.assertEqual(test_position_ids.shape, (926, 4))
            np.testing.assert_array_equal(test_seq_offsets, np.array([0, 926], dtype=np.int64))
            np.testing.assert_array_equal(test_seq_shapes, np.array([[3, 2, 30, 30]], dtype=np.int32))
            self.assertEqual(test_metadata["seq_len"], 926)
            self.assertEqual(test_metadata["position_id_shape"], [3, 2, 30, 30])
            self.assertEqual(test_metadata["sequence_layout"], "pair_eos")

            target_pair_index = int(np.unique(test_position_ids[test_labels != 0][:, 0]).item())
            expected_query_input = _encode_unpadded_canvas([[5], [5]], (3, 2))
            expected_query_output = _encode_fixed_canvas([[6], [6]])
            expected_query_mask = _masked_answer_canvas()

            np.testing.assert_array_equal(
                _reconstruct_pair_canvas(test_inputs, test_position_ids, target_pair_index, 0),
                expected_query_input,
            )
            np.testing.assert_array_equal(
                _reconstruct_pair_canvas(test_inputs, test_position_ids, target_pair_index, 1),
                expected_query_mask,
            )
            np.testing.assert_array_equal(
                _reconstruct_pair_canvas(test_labels, test_position_ids, target_pair_index, 1),
                expected_query_output,
            )

    def test_build_dataset_supports_pair_no_eos_layout_without_sample_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_prefix = tmp_path / "arc"
            output_dir = tmp_path / "dataset"

            mini_puzzle = {
                "p_test": {
                    "train": [
                        {"input": [[1]], "output": [[2]]},
                        {"input": [[3, 3]], "output": [[4, 4]]},
                    ],
                    "test": [
                        {"input": [[5], [5]]},
                    ],
                }
            }

            self._write_subset(
                input_prefix,
                "mini",
                mini_puzzle,
                {"p_test": [[[6], [6]]]},
            )

            convert_dataset(
                DataProcessConfig(
                    input_file_prefix=str(input_prefix),
                    output_dir=str(output_dir),
                    subsets=["mini"],
                    test_set_name="mini",
                    seed=0,
                    num_aug=0,
                    no_padding=True,
                    no_padding_mode="pair_no_eos",
                    min_context_pairs=2,
                )
            )

            test_inputs = np.load(output_dir / "test" / "all__inputs.npy")
            test_labels = np.load(output_dir / "test" / "all__labels.npy")
            test_position_ids = np.load(output_dir / "test" / "all__position_ids.npy")
            test_seq_offsets = np.load(output_dir / "test" / "all__seq_offsets.npy")
            test_seq_shapes = np.load(output_dir / "test" / "all__seq_shapes.npy")
            with open(output_dir / "test" / "dataset.json", encoding="utf-8") as f:
                test_metadata = json.load(f)

            self.assertEqual(test_inputs.shape, (908,))
            self.assertEqual(test_labels.shape, (908,))
            self.assertEqual(test_position_ids.shape, (908, 4))
            np.testing.assert_array_equal(test_seq_offsets, np.array([0, 908], dtype=np.int64))
            np.testing.assert_array_equal(test_seq_shapes, np.array([[3, 2, 30, 30]], dtype=np.int32))
            self.assertEqual(test_metadata["seq_len"], 908)
            self.assertEqual(test_metadata["position_id_shape"], [3, 2, 30, 30])
            self.assertEqual(test_metadata["sequence_layout"], "pair_no_eos")

            target_pair_index = int(np.unique(test_position_ids[test_labels != 0][:, 0]).item())
            expected_query_input = _encode_unpadded_canvas_no_eos([[5], [5]], (2, 1))
            expected_query_output = _encode_fixed_canvas_no_eos([[6], [6]])
            expected_query_mask = _masked_answer_canvas()

            np.testing.assert_array_equal(
                _reconstruct_pair_canvas(test_inputs, test_position_ids, target_pair_index, 0),
                expected_query_input,
            )
            np.testing.assert_array_equal(
                _reconstruct_pair_canvas(test_inputs, test_position_ids, target_pair_index, 1),
                expected_query_mask,
            )
            np.testing.assert_array_equal(
                _reconstruct_pair_canvas(test_labels, test_position_ids, target_pair_index, 1),
                expected_query_output,
            )

    def test_train_dataset_masks_one_random_output_pair_during_collation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_prefix = tmp_path / "arc"
            output_dir = tmp_path / "dataset"

            train_puzzle = {
                "p_train": {
                    "train": [
                        {"input": [[1]], "output": [[2]]},
                        {"input": [[3, 3]], "output": [[4, 4]]},
                    ],
                    "test": [
                        {"input": [[5], [5]]},
                    ],
                }
            }

            self._write_subset(
                input_prefix,
                "aux",
                train_puzzle,
                {"p_train": [[[6], [6]]]},
            )

            convert_dataset(
                DataProcessConfig(
                    input_file_prefix=str(input_prefix),
                    output_dir=str(output_dir),
                    subsets=["aux"],
                    test_set_name="heldout",
                    seed=0,
                    num_aug=0,
                    no_padding=True,
                    no_padding_mode="pair_eos",
                    min_context_pairs=2,
                )
            )

            train_inputs = np.load(output_dir / "train" / "all__inputs.npy")
            train_position_ids = np.load(output_dir / "train" / "all__position_ids.npy")
            train_seq_offsets = np.load(output_dir / "train" / "all__seq_offsets.npy")
            with open(output_dir / "train" / "dataset.json", encoding="utf-8") as f:
                train_metadata = json.load(f)

            self.assertEqual(train_metadata["train_target_mode"], "random_output_pair")
            self.assertEqual(train_metadata["answer_slot_max_grid_size"], 30)
            self.assertEqual(train_metadata["seq_len"], 928)
            self.assertEqual(train_metadata["position_id_shape"], [3, 2, 30, 30])
            self.assertEqual(train_inputs.shape, (32,))
            self.assertEqual(train_position_ids.shape, (32, 4))
            np.testing.assert_array_equal(train_seq_offsets, np.array([0, 32], dtype=np.int64))
            self.assertFalse((output_dir / "train" / "all__labels.npy").exists())

            dataset = PuzzleDataset(
                PuzzleDatasetConfig(
                    seed=3,
                    dataset_path=str(output_dir),
                    global_batch_size=1,
                    test_set_mode=False,
                    epochs_per_iter=1,
                    rank=0,
                    num_replicas=1,
                ),
                split="train",
            )
            set_name, batch, effective_batch_size = next(iter(dataset))

            self.assertEqual(set_name, "all")
            self.assertEqual(effective_batch_size, 1)
            self.assertIn(int(batch["seq_lengths"].item()), {926, 928})
            self.assertIn("source_inputs", batch)

            position_ids = batch["position_ids"].numpy()
            labels = batch["labels"].numpy()
            inputs = batch["inputs"].numpy()
            source_inputs = batch["source_inputs"].numpy()

            labeled_mask = labels != IGNORE_LABEL_ID
            self.assertTrue(np.any(labeled_mask))
            target_pair_index = int(np.unique(position_ids[labeled_mask][:, 0]).item())
            target_output_mask = (position_ids[:, 0] == target_pair_index) & (position_ids[:, 1] == 1)
            self.assertEqual(int(target_output_mask.sum()), 30 * 30)

            np.testing.assert_array_equal(inputs[target_output_mask], np.full((30 * 30,), 2, dtype=np.int32))

            target_source_canvas = _reconstruct_pair_canvas(
                source_inputs.astype(np.uint8),
                position_ids.astype(np.uint8),
                target_pair_index,
                1,
            )
            target_label_canvas = np.where(
                labels[target_output_mask] == IGNORE_LABEL_ID,
                0,
                labels[target_output_mask],
            ).reshape(30, 30).astype(np.uint8)
            np.testing.assert_array_equal(target_label_canvas, target_source_canvas)
            expected_target_canvases = [
                _encode_fixed_canvas([[2]]),
                _encode_fixed_canvas([[4, 4]]),
                _encode_fixed_canvas([[6], [6]]),
            ]
            self.assertTrue(
                any(np.array_equal(target_source_canvas, expected) for expected in expected_target_canvases)
            )
            self.assertTrue(np.all(labels[~target_output_mask] == IGNORE_LABEL_ID))

            non_target_output_pairs = {
                int(pair_id)
                for pair_id in np.unique(position_ids[:, 0]).tolist()
                if int(pair_id) != target_pair_index
            }
            for pair_id in non_target_output_pairs:
                output_mask = (position_ids[:, 0] == pair_id) & (position_ids[:, 1] == 1)
                np.testing.assert_array_equal(inputs[output_mask], source_inputs[output_mask])


if __name__ == "__main__":
    unittest.main()
