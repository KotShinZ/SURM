from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_trained_model import evaluate_model  # noqa: E402
from models.losses import IGNORE_LABEL_ID  # noqa: E402


@dataclass
class DummyCarry:
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class DummyEvalModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(act_inference=False, eval_act_early_stop=False)
        self.call_batch_sizes: List[int] = []

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> DummyCarry:
        batch_size = batch["inputs"].shape[0]
        return DummyCarry(
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={key: torch.empty_like(value) for key, value in batch.items()},
        )

    def forward(
        self,
        carry: DummyCarry,
        batch: Dict[str, torch.Tensor],
        return_keys,
    ) -> Tuple[DummyCarry, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], bool]:
        self.call_batch_sizes.append(int(batch["inputs"].shape[0]))

        halted_view_cache = {
            key: carry.halted.view((-1,) + (1,) * (value.ndim - 1))
            for key, value in batch.items()
        }
        current_data = {
            key: torch.where(halted_view_cache[key], value, carry.current_data[key])
            for key, value in batch.items()
        }
        new_steps = torch.where(carry.halted, 0, carry.steps) + 1
        required_steps = current_data["puzzle_identifiers"].to(torch.int32) + 1
        halted = new_steps >= required_steps

        labels = current_data["labels"]
        valid_mask = labels != IGNORE_LABEL_ID
        solved_mask = halted.view(-1, 1) & valid_mask
        wrong_preds = torch.where(valid_mask, (labels + 1) % self.vocab_size, torch.zeros_like(labels))
        preds = torch.where(solved_mask, labels, wrong_preds).to(torch.int64)
        logits = F.one_hot(preds.to(torch.long), num_classes=self.vocab_size).to(torch.float32) * 9.0 - 4.5

        outputs = {
            "preds": preds,
            "logits": logits,
            "q_halt_logits": torch.zeros((preds.shape[0],), dtype=torch.float32),
            "q_continue_logits": torch.zeros((preds.shape[0],), dtype=torch.float32),
        }
        returned_outputs = {key: outputs[key] for key in return_keys}

        return (
            DummyCarry(steps=new_steps, halted=halted, current_data=current_data),
            torch.tensor(0.0),
            {},
            returned_outputs,
            bool(halted.all().item()),
        )


class ToyLoader:
    def __init__(self, batches: Iterable[Tuple[str, Dict[str, torch.Tensor], int]]) -> None:
        self._batches = list(batches)

    def __iter__(self):
        return iter(self._batches)


class EvaluateTrainedModelTests(unittest.TestCase):
    def _make_loader(self) -> ToyLoader:
        return ToyLoader(
            [
                self._make_batch("alpha", [4, 1, 4, 1], start_value=0),
                self._make_batch("beta", [4, 1, 4, 1], start_value=20),
            ]
        )

    def _make_batch(
        self,
        set_name: str,
        required_steps: List[int],
        start_value: int,
    ) -> Tuple[str, Dict[str, torch.Tensor], int]:
        batch_size = len(required_steps)
        seq_len = 3
        labels = torch.arange(start_value, start_value + batch_size * seq_len, dtype=torch.int64).view(batch_size, seq_len)
        labels = (labels % 5) + 1
        inputs = labels + 10
        source_inputs = inputs + 100
        puzzle_identifiers = torch.tensor([step - 1 for step in required_steps], dtype=torch.int64)
        batch = {
            "inputs": inputs,
            "labels": labels,
            "puzzle_identifiers": puzzle_identifiers,
            "source_inputs": source_inputs,
        }
        return set_name, batch, batch_size

    def _assert_nested_dicts_close(self, actual, expected, path: str = "") -> None:
        self.assertEqual(set(actual.keys()), set(expected.keys()), msg=path)
        for key in actual:
            current_path = f"{path}.{key}" if path else str(key)
            actual_value = actual[key]
            expected_value = expected[key]
            if isinstance(actual_value, dict):
                self._assert_nested_dicts_close(actual_value, expected_value, path=current_path)
            else:
                self.assertAlmostEqual(actual_value, expected_value, places=6, msg=current_path)

    def test_refill_matches_shrink_metrics_and_saved_outputs(self) -> None:
        shrink_model = DummyEvalModel(vocab_size=8)
        refill_model = DummyEvalModel(vocab_size=8)
        loop_checkpoints = [1, 2, 4]

        shrink_result = evaluate_model(
            model=shrink_model,
            dataloader=self._make_loader(),
            device=torch.device("cpu"),
            max_problems=None,
            max_batches=None,
            save_predictions=True,
            hidden_diff_threshold=None,
            loop_checkpoints=loop_checkpoints,
            active_batch_strategy="shrink",
        )
        refill_result = evaluate_model(
            model=refill_model,
            dataloader=self._make_loader(),
            device=torch.device("cpu"),
            max_problems=None,
            max_batches=None,
            save_predictions=True,
            hidden_diff_threshold=None,
            loop_checkpoints=loop_checkpoints,
            active_batch_strategy="refill",
        )

        shrink_metrics, shrink_loop_metrics, shrink_outputs, shrink_batches, shrink_problems = shrink_result
        refill_metrics, refill_loop_metrics, refill_outputs, refill_batches, refill_problems = refill_result

        self.assertEqual(shrink_batches, 2)
        self.assertEqual(refill_batches, 2)
        self.assertEqual(shrink_problems, 8)
        self.assertEqual(refill_problems, 8)
        self._assert_nested_dicts_close(refill_metrics, shrink_metrics)
        self._assert_nested_dicts_close(refill_loop_metrics, shrink_loop_metrics)

        self.assertEqual(set(refill_outputs.keys()), set(shrink_outputs.keys()))
        for set_name in shrink_outputs:
            self.assertEqual(set(refill_outputs[set_name].keys()), set(shrink_outputs[set_name].keys()))
            for key in shrink_outputs[set_name]:
                torch.testing.assert_close(
                    refill_outputs[set_name][key],
                    shrink_outputs[set_name][key],
                    rtol=0.0,
                    atol=0.0,
                )

    def test_refill_keeps_batches_fuller_than_shrink(self) -> None:
        shrink_model = DummyEvalModel(vocab_size=8)
        refill_model = DummyEvalModel(vocab_size=8)

        evaluate_model(
            model=shrink_model,
            dataloader=self._make_loader(),
            device=torch.device("cpu"),
            max_problems=None,
            max_batches=None,
            save_predictions=False,
            hidden_diff_threshold=None,
            loop_checkpoints=None,
            active_batch_strategy="shrink",
        )
        evaluate_model(
            model=refill_model,
            dataloader=self._make_loader(),
            device=torch.device("cpu"),
            max_problems=None,
            max_batches=None,
            save_predictions=False,
            hidden_diff_threshold=None,
            loop_checkpoints=None,
            active_batch_strategy="refill",
        )

        shrink_average_batch = sum(shrink_model.call_batch_sizes) / len(shrink_model.call_batch_sizes)
        refill_average_batch = sum(refill_model.call_batch_sizes) / len(refill_model.call_batch_sizes)

        self.assertGreater(refill_average_batch, shrink_average_batch)
        self.assertIn(2, shrink_model.call_batch_sizes)
        self.assertGreaterEqual(refill_model.call_batch_sizes.count(4), 3)


if __name__ == "__main__":
    unittest.main()
