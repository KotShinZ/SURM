from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eggroll_arc_finetune import GroupScore, rank_groups_by_hardness, split_hard_groups  # noqa: E402
from eggroll_utils import (  # noqa: E402
    PopulationPerturbations,
    apply_updates_,
    collect_evolvable_parameters,
    extract_parameter_state,
    load_parameter_state_,
    normalize_fitness,
)


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.puzzle_emb = nn.Embedding(32, 8)
        self.linear = nn.Linear(8, 4, bias=True)
        self.q_head = nn.Linear(4, 2, bias=True)
        self.dwconv = nn.Conv1d(4, 4, kernel_size=2, groups=4, bias=True)


class EggrollUtilsTests(unittest.TestCase):
    def test_collect_evolvable_parameters_respects_v1_and_widened_rules(self) -> None:
        model = ToyModel()

        v1_specs = collect_evolvable_parameters(model, include_small_tensors=False)
        self.assertEqual([spec.name for spec in v1_specs], ["linear.weight", "q_head.weight"])

        widened_specs = collect_evolvable_parameters(model, include_small_tensors=True)
        widened_names = [spec.name for spec in widened_specs]

        self.assertIn("linear.weight", widened_names)
        self.assertIn("q_head.weight", widened_names)
        self.assertIn("linear.bias", widened_names)
        self.assertIn("q_head.bias", widened_names)
        self.assertIn("dwconv.weight", widened_names)
        self.assertIn("dwconv.bias", widened_names)
        self.assertNotIn("embed_tokens.weight", widened_names)
        self.assertNotIn("puzzle_emb.weight", widened_names)

    def test_population_roundtrip_leaves_parameters_unchanged(self) -> None:
        model = ToyModel()
        specs = collect_evolvable_parameters(model, include_small_tensors=False)
        before = extract_parameter_state(specs)
        perturbations = PopulationPerturbations(specs, population=4, rank=2, sigma=0.1, seed=123)

        for member_index in range(4):
            active_deltas = perturbations.apply_member_(member_index)
            perturbations.revert_member_(active_deltas)

        after = extract_parameter_state(specs)
        for name in before:
            self.assertTrue(torch.allclose(before[name], after[name], atol=1e-6, rtol=0.0), name)

    def test_updates_touch_only_selected_parameters(self) -> None:
        model = ToyModel()
        specs = collect_evolvable_parameters(model, include_small_tensors=False)
        full_before = {name: value.detach().clone() for name, value in model.named_parameters()}

        perturbations = PopulationPerturbations(specs, population=4, rank=2, sigma=0.1, seed=7)
        fitness = normalize_fitness([1.0, -1.0, 0.5, -0.5])
        updates = perturbations.compute_update_tensors(fitness)
        apply_updates_(specs, updates, lr=0.01)

        changed = {
            name: not torch.equal(full_before[name], param.detach())
            for name, param in model.named_parameters()
        }
        self.assertTrue(changed["linear.weight"])
        self.assertTrue(changed["q_head.weight"])
        self.assertFalse(changed["linear.bias"])
        self.assertFalse(changed["q_head.bias"])
        self.assertFalse(changed["embed_tokens.weight"])
        self.assertFalse(changed["puzzle_emb.weight"])
        self.assertFalse(changed["dwconv.weight"])
        self.assertFalse(changed["dwconv.bias"])

    def test_state_extract_and_restore_roundtrip(self) -> None:
        model = ToyModel()
        specs = collect_evolvable_parameters(model, include_small_tensors=False)
        saved = extract_parameter_state(specs)

        with torch.no_grad():
            for spec in specs:
                spec.parameter.add_(1.0)

        load_parameter_state_(specs, saved)
        restored = extract_parameter_state(specs)
        for name in saved:
            self.assertTrue(torch.equal(saved[name], restored[name]), name)

    def test_rank_and_split_hard_groups_are_deterministic(self) -> None:
        scores = [
            GroupScore(group_id=0, failure_rate=0.1, exact_accuracy=0.9, accuracy=0.95, lm_loss=0.2, steps=2.0, count=4),
            GroupScore(group_id=1, failure_rate=0.5, exact_accuracy=0.5, accuracy=0.6, lm_loss=0.8, steps=2.0, count=4),
            GroupScore(group_id=2, failure_rate=0.5, exact_accuracy=0.5, accuracy=0.6, lm_loss=0.7, steps=2.0, count=4),
            GroupScore(group_id=3, failure_rate=0.3, exact_accuracy=0.7, accuracy=0.75, lm_loss=0.4, steps=2.0, count=4),
        ]
        ranked = rank_groups_by_hardness(scores, hard_group_pool=3)
        self.assertEqual(ranked, [1, 2, 3])

        split_a = split_hard_groups(ranked, es_train_groups=2, es_val_groups=1, seed=0)
        split_b = split_hard_groups(ranked, es_train_groups=2, es_val_groups=1, seed=0)
        self.assertEqual(split_a, split_b)


if __name__ == "__main__":
    unittest.main()
