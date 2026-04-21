from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from tqdm import tqdm


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


from data.build_arc_dataset_full_2 import DataProcessConfig, convert_dataset  # noqa: E402
from models.losses import IGNORE_LABEL_ID  # noqa: E402
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig  # noqa: E402


ARC_MAX_GRID_SIZE = 30


def _pair_canvas_shape(inp: list[list[int]], out: list[list[int]]) -> tuple[int, int]:
    inp_arr = np.array(inp, dtype=np.uint8)
    out_arr = np.array(out, dtype=np.uint8)
    height = max(inp_arr.shape[0], out_arr.shape[0])
    width = max(inp_arr.shape[1], out_arr.shape[1])
    if height < ARC_MAX_GRID_SIZE:
        height += 1
    if width < ARC_MAX_GRID_SIZE:
        width += 1
    return height, width


def _encode_unpadded_canvas(grid: list[list[int]], canvas_shape: tuple[int, int]) -> np.ndarray:
    arr = np.array(grid, dtype=np.uint8)
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    height, width = arr.shape
    canvas[:height, :width] = arr + 2
    if height < canvas_shape[0]:
        canvas[height, :width] = 1
    if width < canvas_shape[1]:
        canvas[:height, width] = 1
    return canvas


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


def _iter_with_progress(iterable, total: int, desc: str):
    return tqdm(iterable, total=total, desc=desc, leave=False, unit="pair")


class BuildARCDatasetFull2Tests(unittest.TestCase):
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

    def test_train_dataset_saves_full_pairs_without_labels_and_masks_subset_dynamically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_prefix = tmp_path / "arc"
            output_dir = tmp_path / "dataset"

            train_puzzle = {
                "p_train": {
                    "train": [
                        {"input": [[1]], "output": [[2]]},
                        {"input": [[3, 3]], "output": [[4, 4]]},
                        {"input": [[5], [5]], "output": [[6], [6]]},
                        {"input": [[7, 0], [7, 7]], "output": [[8, 8], [0, 8]]},
                    ],
                    "test": [
                        {"input": [[9, 9, 9]], "output": [[1, 1, 1]]},
                    ],
                }
            }

            self._write_subset(
                input_prefix,
                "aux",
                train_puzzle,
                {"p_train": [[[1, 1, 1]]]},
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
                    min_context_pairs=2,
                )
            )

            train_inputs = np.load(output_dir / "train" / "all__inputs.npy")
            train_position_ids = np.load(output_dir / "train" / "all__position_ids.npy")
            train_seq_offsets = np.load(output_dir / "train" / "all__seq_offsets.npy")
            with open(output_dir / "train" / "dataset.json", encoding="utf-8") as f:
                train_metadata = json.load(f)

            self.assertFalse((output_dir / "train" / "all__labels.npy").exists())
            self.assertEqual(train_metadata["train_target_mode"], "random_output_pair")
            self.assertEqual(train_metadata["min_context_pairs"], 2)
            np.testing.assert_array_equal(train_seq_offsets, np.array([0, train_inputs.shape[0]], dtype=np.int64))

            pair_ids = np.unique(train_position_ids[:, 0]).astype(np.int32)
            self.assertEqual(pair_ids.tolist(), [0, 1, 2, 3, 4])

            expected_pairs = [
                ([[1]], [[2]]),
                ([[3, 3]], [[4, 4]]),
                ([[5], [5]], [[6], [6]]),
                ([[7, 0], [7, 7]], [[8, 8], [0, 8]]),
                ([[9, 9, 9]], [[1, 1, 1]]),
            ]
            for pair_index, (inp_grid, out_grid) in _iter_with_progress(
                enumerate(expected_pairs),
                total=len(expected_pairs),
                desc="verify train pairs",
            ):
                canvas_shape = _pair_canvas_shape(inp_grid, out_grid)
                expected_inp = _encode_unpadded_canvas(inp_grid, canvas_shape)
                expected_out = _encode_unpadded_canvas(out_grid, canvas_shape)
                np.testing.assert_array_equal(
                    _reconstruct_pair_canvas(train_inputs, train_position_ids, pair_index, 0),
                    expected_inp,
                )
                np.testing.assert_array_equal(
                    _reconstruct_pair_canvas(train_inputs, train_position_ids, pair_index, 1),
                    expected_out,
                )

            dataset = PuzzleDataset(
                PuzzleDatasetConfig(
                    seed=7,
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
            self.assertIn("source_inputs", batch)

            position_ids = batch["position_ids"].numpy()
            labels = batch["labels"].numpy()
            inputs = batch["inputs"].numpy()
            source_inputs = batch["source_inputs"].numpy()

            selected_pair_ids = np.unique(position_ids[:, 0]).astype(np.int32)
            self.assertGreaterEqual(selected_pair_ids.size, 3)
            self.assertLessEqual(selected_pair_ids.size, 5)

            labeled_mask = labels != IGNORE_LABEL_ID
            self.assertTrue(np.any(labeled_mask))
            target_pair_ids = np.unique(position_ids[labeled_mask][:, 0]).astype(np.int32)
            self.assertEqual(target_pair_ids.size, 1)
            target_pair_id = int(target_pair_ids[0])

            target_output_mask = (position_ids[:, 0] == target_pair_id) & (position_ids[:, 1] == 1)
            self.assertTrue(np.all(inputs[target_output_mask] == 2))
            self.assertTrue(np.all(labels[~target_output_mask] == IGNORE_LABEL_ID))

            target_source_canvas = _reconstruct_pair_canvas(
                source_inputs.astype(np.uint8),
                position_ids.astype(np.uint8),
                target_pair_id,
                1,
            )
            target_label_canvas = np.where(
                labels[target_output_mask] == IGNORE_LABEL_ID,
                0,
                labels[target_output_mask],
            ).reshape(target_source_canvas.shape).astype(np.uint8)
            np.testing.assert_array_equal(target_label_canvas, target_source_canvas)

            expected_output_canvases = []
            for inp_grid, out_grid in _iter_with_progress(
                expected_pairs,
                total=len(expected_pairs),
                desc="build expected outputs",
            ):
                expected_output_canvases.append(
                    _encode_unpadded_canvas(out_grid, _pair_canvas_shape(inp_grid, out_grid))
                )
            self.assertTrue(
                any(np.array_equal(target_source_canvas, expected) for expected in expected_output_canvases)
            )

            for pair_id in _iter_with_progress(
                selected_pair_ids.tolist(),
                total=selected_pair_ids.size,
                desc="verify visible outputs",
            ):
                if int(pair_id) == target_pair_id:
                    continue
                output_mask = (position_ids[:, 0] == pair_id) & (position_ids[:, 1] == 1)
                np.testing.assert_array_equal(inputs[output_mask], source_inputs[output_mask])

    def test_test_dataset_layout_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_prefix = tmp_path / "arc"
            output_dir = tmp_path / "dataset"

            heldout_puzzle = {
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
                heldout_puzzle,
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
                    min_context_pairs=2,
                )
            )

            test_inputs = np.load(output_dir / "test" / "all__inputs.npy")
            test_labels = np.load(output_dir / "test" / "all__labels.npy")
            test_position_ids = np.load(output_dir / "test" / "all__position_ids.npy")
            test_seq_offsets = np.load(output_dir / "test" / "all__seq_offsets.npy")

            np.testing.assert_array_equal(test_seq_offsets, np.array([0, test_inputs.shape[0]], dtype=np.int64))
            self.assertEqual(np.unique(test_position_ids[:, 0]).astype(np.int32).tolist(), [0, 1, 2])

            expected_pairs = [
                ([[1]], [[2]]),
                ([[3, 3]], [[4, 4]]),
                ([[5], [5]], [[6], [6]]),
            ]

            for pair_index, (inp_grid, out_grid) in _iter_with_progress(
                enumerate(expected_pairs[:-1]),
                total=len(expected_pairs) - 1,
                desc="verify test context pairs",
            ):
                canvas_shape = _pair_canvas_shape(inp_grid, out_grid)
                np.testing.assert_array_equal(
                    _reconstruct_pair_canvas(test_inputs, test_position_ids, pair_index, 0),
                    _encode_unpadded_canvas(inp_grid, canvas_shape),
                )
                np.testing.assert_array_equal(
                    _reconstruct_pair_canvas(test_inputs, test_position_ids, pair_index, 1),
                    _encode_unpadded_canvas(out_grid, canvas_shape),
                )

            query_canvas_shape = _pair_canvas_shape([[5], [5]], [[6], [6]])
            np.testing.assert_array_equal(
                _reconstruct_pair_canvas(test_inputs, test_position_ids, 2, 0),
                _encode_unpadded_canvas([[5], [5]], query_canvas_shape),
            )
            np.testing.assert_array_equal(
                _reconstruct_pair_canvas(test_inputs, test_position_ids, 2, 1),
                np.zeros(query_canvas_shape, dtype=np.uint8),
            )
            np.testing.assert_array_equal(
                _reconstruct_pair_canvas(test_labels, test_position_ids, 2, 1),
                _encode_unpadded_canvas([[6], [6]], query_canvas_shape),
            )
            self.assertEqual(int(np.count_nonzero(test_labels[test_position_ids[:, 0] != 2])), 0)

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
            np.testing.assert_array_equal(batch["inputs"].numpy(), test_inputs.astype(np.int32))
            expected_batch_labels = np.where(
                test_labels.astype(np.int32) == 0,
                IGNORE_LABEL_ID,
                test_labels.astype(np.int32),
            )
            np.testing.assert_array_equal(batch["labels"].numpy(), expected_batch_labels)
            np.testing.assert_array_equal(batch["position_ids"].numpy(), test_position_ids.astype(np.int32))


if __name__ == "__main__":
    unittest.main()
