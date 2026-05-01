from __future__ import annotations

import types
import unittest

import torch

from models.losses import ACTLossHead, IGNORE_LABEL_ID, softmax_cross_entropy


class _DummyModel(torch.nn.Module):
    def __init__(self, carry, outputs, use_act: bool = False):
        super().__init__()
        self._carry = carry
        self._outputs = outputs
        self.config = types.SimpleNamespace(use_act=use_act, diff_L_loss_enabled=False)

    def initial_carry(self, *args, **kwargs):
        return self._carry

    def forward(self, **kwargs):
        return self._carry, self._outputs


class PackedLossMetricTests(unittest.TestCase):
    def test_packed_metrics_match_manual_segment_reduction(self) -> None:
        lengths = torch.tensor([3, 1, 2], dtype=torch.int32)
        labels = torch.tensor([1, 2, IGNORE_LABEL_ID, 0, 2, 1], dtype=torch.int32)
        logits = torch.tensor(
            [
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 4.0],
                [3.0, 1.0, 0.0, 0.0],
                [6.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 5.0],
            ],
            dtype=torch.float32,
        )
        halted = torch.tensor([True, False, True], dtype=torch.bool)
        steps = torch.tensor([2, 5, 3], dtype=torch.int32)

        carry = types.SimpleNamespace(
            current_data={"labels": labels, "seq_lengths": lengths},
            halted=halted,
            steps=steps,
        )
        outputs = {"logits": logits}
        loss_head = ACTLossHead(_DummyModel(carry, outputs), loss_type="softmax_cross_entropy")

        _, total_loss, metrics, _, _ = loss_head(return_keys=set())

        expected_loss_counts = torch.tensor([2, 1, 2], dtype=torch.long)
        expected_correct_counts = torch.tensor([1, 1, 1], dtype=torch.long)
        expected_seq_is_correct = expected_correct_counts == expected_loss_counts
        expected_valid = halted & (expected_loss_counts > 0)

        self.assertEqual(int(metrics["count"].item()), int(expected_valid.sum().item()))
        self.assertAlmostEqual(float(metrics["accuracy"].item()), 1.0, places=6)
        self.assertEqual(int(metrics["exact_accuracy"].item()), int((expected_valid & expected_seq_is_correct).sum().item()))
        self.assertEqual(int(metrics["steps"].item()), 5)

        expected_token_divisor = torch.tensor([2.0, 2.0, 2.0, 1.0, 2.0, 2.0], dtype=torch.float32)
        expected_loss_values = softmax_cross_entropy(logits, labels, ignore_index=IGNORE_LABEL_ID)
        expected_lm_loss = (expected_loss_values / expected_token_divisor).sum()

        self.assertTrue(torch.allclose(metrics["lm_loss"], expected_lm_loss, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(total_loss, expected_lm_loss, atol=1e-6, rtol=1e-6))

    def test_label_mask_one_masks_all_training_labels(self) -> None:
        labels = torch.tensor([[1, 2, IGNORE_LABEL_ID], [0, 1, 2]], dtype=torch.int32)
        logits = torch.zeros((2, 3, 4), dtype=torch.float32)
        halted = torch.tensor([True, True], dtype=torch.bool)
        steps = torch.tensor([1, 1], dtype=torch.int32)

        carry = types.SimpleNamespace(
            current_data={"labels": labels, "seq_lengths": torch.tensor([3, 3], dtype=torch.int32)},
            halted=halted,
            steps=steps,
        )
        outputs = {"logits": logits}
        loss_head = ACTLossHead(
            _DummyModel(carry, outputs),
            loss_type="softmax_cross_entropy",
            label_mask=1.0,
        )

        _, total_loss, metrics, _, _ = loss_head(return_keys=set())

        self.assertEqual(int(metrics["count"].item()), 0)
        self.assertTrue(torch.allclose(metrics["lm_loss"], torch.tensor(0.0), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(total_loss, torch.tensor(0.0), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(carry.current_data["labels"], labels))

    def test_label_mask_is_disabled_during_eval(self) -> None:
        labels = torch.tensor([[1, 2, IGNORE_LABEL_ID], [0, 1, 2]], dtype=torch.int32)
        logits = torch.zeros((2, 3, 4), dtype=torch.float32)
        halted = torch.tensor([True, True], dtype=torch.bool)
        steps = torch.tensor([1, 1], dtype=torch.int32)

        carry = types.SimpleNamespace(
            current_data={"labels": labels, "seq_lengths": torch.tensor([3, 3], dtype=torch.int32)},
            halted=halted,
            steps=steps,
        )
        outputs = {"logits": logits}
        loss_head = ACTLossHead(
            _DummyModel(carry, outputs),
            loss_type="softmax_cross_entropy",
            label_mask=1.0,
        )
        loss_head.eval()

        _, total_loss, metrics, _, _ = loss_head(return_keys=set())
        expected_loss = (
            softmax_cross_entropy(logits, labels, ignore_index=IGNORE_LABEL_ID)
            / torch.tensor([[2.0], [3.0]])
        ).sum()

        self.assertEqual(int(metrics["count"].item()), 2)
        self.assertTrue(torch.allclose(metrics["lm_loss"], expected_loss, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(total_loss, expected_loss, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
