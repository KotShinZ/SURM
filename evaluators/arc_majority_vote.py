from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from data.build_arc_dataset import arc_grid_to_np, grid_hash
from evaluators.arc import ARC


class StandaloneARCMajorityVoteEvaluator(ARC):
    """Single-process ARC evaluator adapter for offline checkpoint evaluation."""

    metric_prefix = ARC.__name__

    def result(self, save_path: Optional[str]) -> Dict[str, float]:
        return self._result_from_gathered_predictions(
            global_hmap_preds=[(self._local_hmap, self._local_preds)],
            save_path=save_path,
        )

    def _result_from_gathered_predictions(
        self,
        global_hmap_preds: List[Tuple[dict, dict]],
        save_path: Optional[str],
    ) -> Dict[str, float]:
        submission = {}
        correct = [0.0 for _ in range(len(self.pass_Ks))]
        maj_correct = {size: 0.0 for size in self.maj_sample_sizes}

        for name, puzzle in self.test_puzzles.items():
            submission[name] = []
            num_test_correct = [0 for _ in range(len(self.pass_Ks))]
            maj_test_correct = {size: 0 for size in self.maj_sample_sizes}

            for pair in puzzle["test"]:
                input_hash = grid_hash(arc_grid_to_np(pair["input"]))
                label_hash = grid_hash(arc_grid_to_np(pair["output"]))

                p_map = {}
                pred_samples = []
                for hmap, preds in global_hmap_preds:
                    for h, q, q_log_prob in preds.get(name, {}).get(input_hash, []):
                        p_map.setdefault(h, [0, 0.0, -np.inf])
                        p_map[h][0] += 1
                        p_map[h][1] += q
                        p_map[h][2] = max(p_map[h][2], q_log_prob)
                        pred_samples.append((h, q_log_prob))

                if not p_map:
                    print(f"Puzzle {name} has no predictions.")
                    continue

                for h, stats in p_map.items():
                    stats[1] /= stats[0]

                p_map = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)

                for i, k in enumerate(self.pass_Ks):
                    ok = False
                    for h, _stats in p_map[:k]:
                        ok |= h == label_hash
                    num_test_correct[i] += ok

                pred_grids = []
                for h, _stats in p_map[: self.submission_K]:
                    for hmap, _preds in global_hmap_preds:
                        if h in hmap:
                            pred_grids.append(hmap[h])
                            break

                while len(pred_grids) < self.submission_K:
                    pred_grids.append(pred_grids[0])

                submission[name].append(
                    {f"attempt_{i + 1}": grid.tolist() for i, grid in enumerate(pred_grids)}
                )

                if pred_samples:
                    logps = np.array([lp for _, lp in pred_samples], dtype=np.float64)
                    max_logp = logps.max()
                    probs = np.exp(logps - max_logp)
                    prob_sum = probs.sum()
                    if prob_sum > 0:
                        probs /= prob_sum
                    else:
                        probs = np.full_like(probs, 1.0 / len(probs))

                    for sample_size in self.maj_sample_sizes:
                        sampled_indices = np.random.choice(
                            len(pred_samples),
                            size=sample_size,
                            replace=True,
                            p=probs,
                        )
                        sampled_logps = logps[sampled_indices]
                        best_idx = sampled_indices[np.argmax(sampled_logps)]
                        maj_test_correct[sample_size] += pred_samples[best_idx][0] == label_hash

            for i in range(len(self.pass_Ks)):
                correct[i] += num_test_correct[i] / len(puzzle["test"])
            for sample_size in self.maj_sample_sizes:
                maj_correct[sample_size] += maj_test_correct[sample_size] / len(puzzle["test"])

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "submission.json"), "w", encoding="utf-8") as f:
                json.dump(submission, f)

        result = {
            f"{self.metric_prefix}/pass@{k}": correct[i] / len(self.test_puzzles)
            for i, k in enumerate(self.pass_Ks)
        }
        result.update(
            {
                f"{self.metric_prefix}/maj@{k}": maj_correct[k] / len(self.test_puzzles)
                for k in self.maj_sample_sizes
            }
        )
        return result


def maybe_create_arc_majority_vote_evaluator(
    data_path: str,
    eval_metadata,
    split: str,
    submission_K: int = 2,
    pass_Ks: Sequence[int] = (1, 2, 5, 10, 100, 1000),
    aggregated_voting: bool = True,
) -> Optional[StandaloneARCMajorityVoteEvaluator]:
    dataset_root = Path(data_path)
    required_files = (
        dataset_root / "identifiers.json",
        dataset_root / "test_puzzles.json",
    )
    if split != "test" or not all(path.is_file() for path in required_files):
        return None

    return StandaloneARCMajorityVoteEvaluator(
        data_path=data_path,
        eval_metadata=eval_metadata,
        submission_K=submission_K,
        pass_Ks=pass_Ks,
        aggregated_voting=aggregated_voting,
    )


def default_arc_submission_dir(summary_output_path: Path) -> Path:
    return summary_output_path.with_name(f"{summary_output_path.stem}_arc_majority_vote")
