from __future__ import annotations

import types
import unittest

import torch

from evaluate_trained_model import (
    _compute_batch_metric_sums,
    _select_prediction_examples,
    _valid_example_mask,
)
from models.losses import IGNORE_LABEL_ID


class EvaluatePackedLabelLengthTests(unittest.TestCase):
    def test_uses_label_lengths_when_labels_are_shorter_than_inputs(self) -> None:
        batch = {
            "inputs": torch.arange(8, dtype=torch.int32),
            "labels": torch.tensor([1, IGNORE_LABEL_ID, 2, 3, 0], dtype=torch.int32),
            "puzzle_identifiers": torch.tensor([10, 11], dtype=torch.int32),
            "seq_lengths": torch.tensor([4, 4], dtype=torch.int32),
            "seq_offsets": torch.tensor([0, 4, 8], dtype=torch.int32),
            "label_seq_lengths": torch.tensor([2, 3], dtype=torch.int32),
            "label_seq_offsets": torch.tensor([0, 2, 5], dtype=torch.int32),
        }
        preds = {
            "preds": torch.tensor([1, 0, 2, 9, 0], dtype=torch.int64),
            "logits": torch.tensor(
                [
                    [0.0, 5.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 5.0, 0.0],
                    [0.0, 5.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
        }

        valid_mask = _valid_example_mask(batch)
        self.assertEqual(valid_mask.tolist(), [True, True])

        selected = _select_prediction_examples(preds, batch, torch.tensor([1]))
        self.assertEqual(selected["preds"].tolist(), [2, 9, 0])

        metrics = _compute_batch_metric_sums(
            model=types.SimpleNamespace(),
            batch=batch,
            preds=preds,
            final_steps=torch.tensor([4, 5], dtype=torch.int32),
            keep_mask=valid_mask,
        )

        self.assertEqual(metrics["count"], 2.0)
        self.assertAlmostEqual(metrics["accuracy"], 1.0 + 2.0 / 3.0, places=6)
        self.assertEqual(metrics["exact_accuracy"], 1.0)
        self.assertEqual(metrics["steps"], 9.0)


if __name__ == "__main__":
    unittest.main()
