from __future__ import annotations

import unittest

import numpy as np

from data.common import PuzzleDatasetMetadata
from models.losses import IGNORE_LABEL_ID
from puzzle_dataset import ARC_MAX_GRID_SIZE, PuzzleDataset, PuzzleDatasetConfig, _sample_batch
from puzzle_full_dataset import PuzzleFullDataset, _sample_batch as _sample_full_batch


def _config(**overrides) -> PuzzleDatasetConfig:
    values = dict(
        seed=0,
        dataset_path="/tmp/unused",
        global_batch_size=1,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )
    values.update(overrides)
    return PuzzleDatasetConfig(**values)


def _metadata() -> PuzzleDatasetMetadata:
    return PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        vocab_size=12,
        seq_len=ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE,
        num_puzzle_identifiers=1,
        total_groups=1,
        mean_puzzle_examples=1.0,
        sets=["all"],
        variable_seq_lengths=True,
    )


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

        self.assertGreaterEqual(selected_examples.size, 2)
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


class PuzzleDatasetPaddingTests(unittest.TestCase):
    def _dataset(self, *, padding: bool) -> PuzzleDataset:
        dataset = PuzzleDataset.__new__(PuzzleDataset)
        dataset.config = _config(padding=padding)
        dataset.metadata = _metadata()
        dataset.split = "train"
        return dataset

    def test_padding_true_pads_variable_pair_to_30x30(self) -> None:
        dataset = {
            "inputs": np.array([2, 3, 4, 5], dtype=np.uint8),
            "labels": np.array([6, 7], dtype=np.uint8),
            "seq_offsets": np.array([0, 4], dtype=np.int64),
            "label_seq_offsets": np.array([0, 2], dtype=np.int64),
            "seq_shapes": np.array([[2, 2]], dtype=np.int32),
            "label_seq_shapes": np.array([[1, 2]], dtype=np.int32),
        }

        batch = self._dataset(padding=True)._select_examples(dataset, np.array([0], dtype=np.int64))

        self.assertEqual(batch["seq_lengths"].tolist(), [ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE])
        self.assertEqual(batch["label_seq_lengths"].tolist(), [ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE])
        np.testing.assert_array_equal(batch["seq_shapes"], np.array([[30, 30]], dtype=np.int32))
        np.testing.assert_array_equal(batch["label_seq_shapes"], np.array([[30, 30]], dtype=np.int32))

        input_grid = batch["inputs"].reshape(30, 30)
        label_grid = batch["labels"].reshape(30, 30)
        np.testing.assert_array_equal(input_grid[:2, :2], np.array([[2, 3], [4, 5]], dtype=np.uint8))
        np.testing.assert_array_equal(label_grid[0, :2], np.array([6, 7], dtype=np.uint8))
        self.assertEqual(int(input_grid[2:, :].sum() + input_grid[:, 2:].sum()), 0)
        self.assertEqual(int(label_grid[1:, :].sum() + label_grid[:, 2:].sum()), 0)

    def test_padding_false_keeps_current_pair_max_padding(self) -> None:
        dataset = {
            "inputs": np.array([2, 3, 4, 5], dtype=np.uint8),
            "labels": np.array([6, 7], dtype=np.uint8),
            "seq_offsets": np.array([0, 4], dtype=np.int64),
            "label_seq_offsets": np.array([0, 2], dtype=np.int64),
            "seq_shapes": np.array([[2, 2]], dtype=np.int32),
            "label_seq_shapes": np.array([[1, 2]], dtype=np.int32),
        }

        batch = self._dataset(padding=False)._select_examples(dataset, np.array([0], dtype=np.int64))

        self.assertEqual(batch["seq_lengths"].tolist(), [4])
        self.assertEqual(batch["label_seq_lengths"].tolist(), [4])
        np.testing.assert_array_equal(batch["seq_shapes"], np.array([[2, 2]], dtype=np.int32))
        np.testing.assert_array_equal(batch["label_seq_shapes"], np.array([[2, 2]], dtype=np.int32))


class PuzzleFullDatasetPaddingTests(unittest.TestCase):
    def _dataset(self, *, padding: bool) -> PuzzleFullDataset:
        dataset = PuzzleFullDataset.__new__(PuzzleFullDataset)
        dataset.config = _config(
            padding=padding,
            full_answer_initial_mode="black",
            full_answer_initial_black_token_id=2,
        )
        dataset.metadata = _metadata()
        dataset.split = "train"
        return dataset

    def test_padding_true_pads_only_target_answer_slot_to_30x30(self) -> None:
        dataset = self._dataset(padding=True)
        pairs = [
            (
                np.array([2, 3], dtype=np.int32),
                np.array([4, 5], dtype=np.int32),
            ),
            (
                np.array([6, 7, 8, 9], dtype=np.int32),
                np.array([10], dtype=np.int32),
            ),
        ]
        shapes = [((1, 2), (2, 1)), ((2, 2), (1, 1))]

        sample = dataset._build_pairs_sample(
            pairs,
            shapes,
            target_pair_index=1,
            rng=np.random.default_rng(0),
        )

        position_ids = sample["position_ids"]
        target_answer_mask = (position_ids[:, 0] == 1) & (position_ids[:, 1] == 1)
        target_problem_mask = (position_ids[:, 0] == 1) & (position_ids[:, 1] == 0)
        context_answer_mask = (position_ids[:, 0] == 0) & (position_ids[:, 1] == 1)

        self.assertEqual(int(sample["seq_lengths"].item()), 2 + 2 + 4 + 900)
        self.assertEqual(int(target_answer_mask.sum()), 900)
        self.assertEqual(int(target_problem_mask.sum()), 4)
        self.assertEqual(int(context_answer_mask.sum()), 2)
        self.assertTrue(np.all(sample["inputs"][target_answer_mask] == 2))
        self.assertTrue(np.all(sample["labels"][context_answer_mask] == IGNORE_LABEL_ID))
        self.assertEqual(int(sample["answer_mask"][target_answer_mask].sum()), 900)

        target_positions = position_ids[target_answer_mask]
        self.assertEqual(int(target_positions[:, 2].max()), 29)
        self.assertEqual(int(target_positions[:, 3].max()), 29)
        target_labels = sample["labels"][target_answer_mask].reshape(30, 30)
        self.assertEqual(int(target_labels[0, 0]), 10)
        self.assertTrue(np.all(target_labels.reshape(-1)[1:] == IGNORE_LABEL_ID))

    def test_padding_false_keeps_target_answer_slot_shape(self) -> None:
        sample = self._dataset(padding=False)._build_pairs_sample(
            [(np.array([2], dtype=np.int32), np.array([3], dtype=np.int32))],
            [((1, 1), (1, 1))],
            target_pair_index=0,
            rng=np.random.default_rng(0),
        )

        position_ids = sample["position_ids"]
        target_answer_mask = (position_ids[:, 0] == 0) & (position_ids[:, 1] == 1)
        self.assertEqual(int(target_answer_mask.sum()), 1)


if __name__ == "__main__":
    unittest.main()
