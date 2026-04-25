from __future__ import annotations

import unittest

import numpy as np

from puzzle_dataset import _sample_batch


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


if __name__ == "__main__":
    unittest.main()
