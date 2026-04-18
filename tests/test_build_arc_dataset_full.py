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

            target_slot = 2
            expected_query_input = _encode_fixed_canvas([[5], [5]])
            expected_query_output = _encode_fixed_canvas([[6], [6]])

            np.testing.assert_array_equal(test_inputs[0, target_slot, 0], expected_query_input)
            np.testing.assert_array_equal(test_inputs[0, target_slot, 1], expected_query_input)
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
            np.testing.assert_array_equal(reshaped_inputs[0, target_slot, 1], expected_query_input.astype(np.int32))
            np.testing.assert_array_equal(
                reshaped_position_ids[0, 12, 1, 29, 29],
                np.array([12, 1, 29, 29], dtype=np.int32),
            )


if __name__ == "__main__":
    unittest.main()
