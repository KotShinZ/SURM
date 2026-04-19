from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.build_arc_dataset_full import (  # noqa: E402
    ARCFullPuzzle,
    _build_full_context_example,
    _extract_pair_grid,
    convert_single_arc_puzzle,
)


def _grid(value: int) -> list[list[int]]:
    return [[value]]


def _pair(inp: int, out: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([[inp]], dtype=np.uint8),
        np.array([[out]], dtype=np.uint8),
    )


class BuildArcDatasetFullTests(unittest.TestCase):
    def test_convert_single_arc_puzzle_uses_only_last_pair_as_test_target(self) -> None:
        puzzle = {
            "train": [
                {"input": _grid(1), "output": _grid(2)},
                {"input": _grid(3), "output": _grid(4)},
                {"input": _grid(5), "output": _grid(6)},
            ],
            "test": [
                {"input": _grid(7), "output": _grid(8)},
                {"input": _grid(9), "output": _grid(1)},
            ],
        }
        results: dict = {}

        converted = convert_single_arc_puzzle(
            results=results,
            name="demo",
            puzzle=puzzle,
            aug_count=0,
            min_context_pairs=2,
            dest_mapping={"train": ("train", "all"), "test": ("test", "all")},
        )

        self.assertTrue(converted)

        train_template = results["train"]["all"][0][0]
        test_template = results["test"]["all"][0][0]

        self.assertEqual(train_template.target_indices, [0, 1, 2])
        self.assertEqual(test_template.target_indices, [4])
        self.assertEqual(len(test_template.pairs), 5)

    def test_build_full_context_example_keeps_eval_pairs_in_original_order(self) -> None:
        puzzle = ARCFullPuzzle(
            id="demo",
            pairs=[
                _pair(1, 2),
                _pair(3, 4),
                _pair(5, 6),
            ],
            target_indices=[2],
        )

        built = _build_full_context_example(
            puzzle=puzzle,
            target_idx=2,
            min_context_pairs=2,
            enable_translational_augment=False,
            no_padding=True,
            use_all_pairs_in_order=True,
        )

        self.assertIsNotNone(built)
        sample_input, sample_label, seq_shape, position_ids = built

        self.assertEqual(seq_shape, (1, int(sample_input.shape[0])))
        self.assertEqual(int(position_ids[:, 0].max()) + 1, 3)

        for pair_idx, expected_input in enumerate([1, 3, 5]):
            input_grid = _extract_pair_grid(sample_input, position_ids, pair_idx, is_output=False)
            self.assertEqual(int(input_grid[0, 0]), expected_input + 2)

        first_output = _extract_pair_grid(sample_input, position_ids, 0, is_output=True)
        second_output = _extract_pair_grid(sample_input, position_ids, 1, is_output=True)
        target_output = _extract_pair_grid(sample_input, position_ids, 2, is_output=True)
        label_target_output = _extract_pair_grid(sample_label, position_ids, 2, is_output=True)

        self.assertEqual(int(first_output[0, 0]), 4)
        self.assertEqual(int(second_output[0, 0]), 6)
        self.assertTrue(np.all(target_output == 0))
        self.assertEqual(int(label_target_output[0, 0]), 8)


if __name__ == "__main__":
    unittest.main()
