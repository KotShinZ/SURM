from __future__ import annotations

import unittest

import numpy as np
import torch

from data.common import PuzzleDatasetMetadata
from evaluators.arc import ARC
from models.losses import IGNORE_LABEL_ID
from puzzle_dataset import ARC_MAX_GRID_SIZE, PuzzleDataset, PuzzleDatasetConfig
from puzzle_full_dataset import PuzzleFullDataset


ARC_FIXED_TOKEN_COUNT = ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE


def _arc_with_recording(records: list) -> ARC:
    arc = ARC.__new__(ARC)

    def record_prediction(identifier, input_grid, pred_grid, q, q_log_prob):
        records.append((identifier, input_grid, pred_grid, q, q_log_prob))

    arc._record_prediction = record_prediction
    return arc


def _config(**overrides) -> PuzzleDatasetConfig:
    values = dict(
        seed=0,
        dataset_path="/tmp/unused",
        global_batch_size=1,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
        padding=True,
    )
    values.update(overrides)
    return PuzzleDatasetConfig(**values)


def _metadata() -> PuzzleDatasetMetadata:
    return PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        vocab_size=12,
        seq_len=ARC_FIXED_TOKEN_COUNT,
        num_puzzle_identifiers=1,
        total_groups=1,
        mean_puzzle_examples=1.0,
        sets=["all"],
        variable_seq_lengths=True,
    )


class ARCFixedEvalTests(unittest.TestCase):
    def test_puzzle_dataset_padding_true_eval_batch_pads_inputs_to_30x30(self) -> None:
        puzzle_dataset = PuzzleDataset.__new__(PuzzleDataset)
        puzzle_dataset.config = _config()
        puzzle_dataset.metadata = _metadata()
        puzzle_dataset.split = "test"
        puzzle_dataset.local_batch_size = 1
        data = {
            "inputs": np.array([2, 3], dtype=np.uint8),
            "labels": np.array([4, 0], dtype=np.uint8),
            "position_ids": np.array([[0, 0], [0, 1]], dtype=np.int32),
            "seq_offsets": np.array([0, 2], dtype=np.int64),
            "label_seq_offsets": np.array([0, 2], dtype=np.int64),
            "seq_shapes": np.array([[1, 2]], dtype=np.int32),
            "label_seq_shapes": np.array([[1, 2]], dtype=np.int32),
        }

        batch_fields = puzzle_dataset._select_examples(data, np.array([0], dtype=np.int64))
        batch_fields["puzzle_identifiers"] = np.array([1], dtype=np.int64)
        fixed = puzzle_dataset._collate_batch(
            batch_fields,
            np.random.default_rng(0),
            make_masked_inputs=False,
        )

        self.assertEqual(fixed["inputs"].numel(), ARC_FIXED_TOKEN_COUNT)
        self.assertEqual(fixed["labels"].numel(), ARC_FIXED_TOKEN_COUNT)
        self.assertEqual(fixed["seq_lengths"].tolist(), [ARC_FIXED_TOKEN_COUNT])
        self.assertEqual(fixed["inputs"][0].item(), 2)
        self.assertEqual(fixed["inputs"][1].item(), 3)
        self.assertTrue(torch.all(fixed["inputs"][2:] == 0).item())
        self.assertEqual(fixed["labels"][0].item(), 4)
        self.assertTrue(torch.all(fixed["labels"][1:] == IGNORE_LABEL_ID).item())
        self.assertEqual(fixed["position_ids"][-1].tolist(), [29, 29])

    def test_puzzle_full_padding_true_eval_batch_expands_target_answer_slot_to_30x30(self) -> None:
        dataset = PuzzleFullDataset.__new__(PuzzleFullDataset)
        dataset.config = _config(
            full_answer_initial_mode="black",
            full_answer_initial_black_token_id=2,
            full_answer_initial_noise_token_min=2,
            full_answer_initial_noise_token_max=11,
        )
        dataset.metadata = _metadata()
        dataset.split = "test"

        sample = dataset._build_pairs_sample(
            [
                (np.array([2], dtype=np.int32), np.array([3], dtype=np.int32)),
                (np.array([4], dtype=np.int32), np.array([9], dtype=np.int32)),
            ],
            [((1, 1), (1, 1)), ((1, 1), (1, 1))],
            target_pair_index=1,
            rng=np.random.default_rng(0),
        )
        target_mask = sample["answer_mask"]

        self.assertEqual(int(target_mask.sum()), ARC_FIXED_TOKEN_COUNT)
        self.assertEqual(int(sample["seq_lengths"].item()), ARC_FIXED_TOKEN_COUNT + 3)
        self.assertTrue(np.all(sample["inputs"][target_mask] == 2))
        self.assertEqual(int(sample["labels"][target_mask][0]), 9)
        self.assertTrue(np.all(sample["labels"][target_mask][1:] == 0))
        self.assertEqual(sample["position_ids"][target_mask][-1].tolist(), [1, 1, 29, 29])

    def test_puzzle_full_padding_false_eval_keeps_target_answer_slot_shape(self) -> None:
        dataset = PuzzleFullDataset.__new__(PuzzleFullDataset)
        dataset.config = _config(
            padding=False,
            full_answer_initial_mode="black",
            full_answer_initial_black_token_id=2,
        )
        dataset.metadata = _metadata()
        dataset.split = "test"

        sample = dataset._build_pairs_sample(
            [(np.array([4], dtype=np.int32), np.array([9], dtype=np.int32))],
            [((1, 1), (1, 1))],
            target_pair_index=0,
            rng=np.random.default_rng(0),
        )

        self.assertEqual(int(sample["answer_mask"].sum()), 1)
        self.assertEqual(int(sample["seq_lengths"].item()), 2)

    def test_arc_packed_update_uses_answer_mask_when_labels_are_none(self) -> None:
        records = []
        arc = _arc_with_recording(records)
        batch = {
            "inputs": torch.tensor([2, 4, 4, 2, 2, 2], dtype=torch.int32),
            "source_inputs": torch.tensor([2, 4, 4, 5, 0, 0], dtype=torch.int32),
            "answer_mask": torch.tensor([False, False, False, True, True, True], dtype=torch.bool),
            "position_ids": torch.tensor(
                [
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [1, 1, 0, 0],
                    [1, 1, 0, 1],
                    [1, 1, 1, 0],
                ],
                dtype=torch.int32,
            ),
            "seq_offsets": torch.tensor([0, 6], dtype=torch.int32),
            "puzzle_identifiers": torch.tensor([7], dtype=torch.int64),
            "labels": None,
        }
        preds = {
            "preds": torch.tensor([0, 0, 0, 9, 0, 0], dtype=torch.int64),
            "q_halt_logits": torch.tensor([0.0], dtype=torch.float32),
        }

        arc.update_batch(batch, preds)

        self.assertEqual(len(records), 1)
        identifier, input_grid, pred_grid, _q, _q_log_prob = records[0]
        self.assertEqual(identifier, 7)
        self.assertEqual(input_grid.tolist(), [[2]])
        self.assertEqual(pred_grid.tolist(), [[7]])


if __name__ == "__main__":
    unittest.main()
