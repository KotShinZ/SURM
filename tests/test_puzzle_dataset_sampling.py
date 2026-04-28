from __future__ import annotations

import unittest

import numpy as np

from puzzle_dataset import _sample_batch
from puzzle_full_dataset import _sample_batch as _sample_full_batch


class PuzzleDatasetSamplingTests(unittest.TestCase):
    def test_sample_batch_limits_examples_per_puzzle(self) -> None:
        rng = np.random.default_rng(0)

        start_index, batch_indices, batch_puzzle_indices = _sample_batch(
            rng,
            group_order=np.array([0], dtype=np.int32),
            puzzle_indices=np.array([0, 5], dtype=np.int32),
            group_indices=np.array([0, 1], dtype=np.int32),
            start_index=0,
            global_batch_size=5,
            examples_per_puzzle=2,
        )

        self.assertEqual(start_index, 1)
        self.assertEqual(batch_indices.size, 2)
        self.assertEqual(batch_puzzle_indices.tolist(), [0, 0])
        self.assertEqual(np.unique(batch_indices).size, 2)
        self.assertTrue(np.all((0 <= batch_indices) & (batch_indices < 5)))

    def test_sample_batch_allows_previous_full_puzzle_behavior(self) -> None:
        rng = np.random.default_rng(0)

        _start_index, batch_indices, batch_puzzle_indices = _sample_batch(
            rng,
            group_order=np.array([0], dtype=np.int32),
            puzzle_indices=np.array([0, 5], dtype=np.int32),
            group_indices=np.array([0, 1], dtype=np.int32),
            start_index=0,
            global_batch_size=5,
            examples_per_puzzle=None,
        )

        self.assertEqual(batch_indices.size, 5)
        self.assertEqual(batch_puzzle_indices.tolist(), [0, 0, 0, 0, 0])

    def test_sample_batch_rejects_non_positive_examples_per_puzzle(self) -> None:
        rng = np.random.default_rng(0)

        with self.assertRaisesRegex(ValueError, "examples_per_puzzle"):
            _sample_batch(
                rng,
                group_order=np.array([0], dtype=np.int32),
                puzzle_indices=np.array([0, 5], dtype=np.int32),
                group_indices=np.array([0, 1], dtype=np.int32),
                start_index=0,
                global_batch_size=5,
                examples_per_puzzle=0,
            )


class PuzzleFullDatasetSamplingTests(unittest.TestCase):
    def test_full_sample_batch_uses_examples_from_one_puzzle(self) -> None:
        rng = np.random.default_rng(0)

        start_index, batch_example_indices, batch_puzzle_indices = _sample_full_batch(
            rng,
            group_order=np.array([0], dtype=np.int32),
            puzzle_indices=np.array([0, 2, 8, 11], dtype=np.int32),
            group_indices=np.array([0, 3], dtype=np.int32),
            start_index=0,
            global_batch_size=1,
            min_pairs=3,
            max_pairs=5,
        )

        self.assertEqual(start_index, 1)
        self.assertEqual(len(batch_example_indices), 1)
        self.assertEqual(len(batch_puzzle_indices), 1)

        selected_examples = batch_example_indices[0]
        selected_puzzle = int(batch_puzzle_indices[0][0])
        puzzle_start = int(np.array([0, 2, 8, 11], dtype=np.int32)[selected_puzzle])
        puzzle_end = int(np.array([0, 2, 8, 11], dtype=np.int32)[selected_puzzle + 1])

        self.assertGreaterEqual(selected_examples.size, 3)
        self.assertLessEqual(selected_examples.size, 5)
        self.assertEqual(np.unique(selected_examples).size, selected_examples.size)
        self.assertTrue(np.all((puzzle_start <= selected_examples) & (selected_examples < puzzle_end)))

    def test_full_sample_batch_skips_puzzles_with_too_few_examples(self) -> None:
        rng = np.random.default_rng(0)

        start_index, batch_example_indices, batch_puzzle_indices = _sample_full_batch(
            rng,
            group_order=np.array([0, 1], dtype=np.int32),
            puzzle_indices=np.array([0, 2, 7], dtype=np.int32),
            group_indices=np.array([0, 1, 2], dtype=np.int32),
            start_index=0,
            global_batch_size=1,
            min_pairs=3,
            max_pairs=5,
        )

        self.assertEqual(start_index, 2)
        self.assertEqual(len(batch_example_indices), 1)
        self.assertEqual(batch_puzzle_indices[0].tolist(), [1])
        self.assertTrue(np.all((2 <= batch_example_indices[0]) & (batch_example_indices[0] < 7)))


if __name__ == "__main__":
    unittest.main()
