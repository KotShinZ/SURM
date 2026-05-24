from typing import Dict, Sequence, Optional
import os
import json

import torch
import torch.nn.functional as F
import numpy as np
from numba import njit
import torch.distributed as dist

from data.build_arc_dataset import inverse_aug, grid_hash, arc_grid_to_np
from data.common import PuzzleDatasetMetadata
from models.losses import IGNORE_LABEL_ID


@njit
def _crop(grid: np.ndarray):
    """Find maximum-sized rectangle without any EOS token inside. """
    grid = grid.reshape(30, 30)

    max_area = 0
    max_size = (0, 0)
    nr, nc = grid.shape
    
    num_c = nc
    for num_r in range(1, nr + 1):
        # Scan for maximum c
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if (x < 2) | (x > 11):
                num_c = c - 1
                break
        
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)

    return (grid[:max_size[0], :max_size[1]] - 2).astype(np.uint8)


class ARC:
    required_outputs = {
        "inputs",
        "labels",
        "source_inputs",
        "position_ids",
        "puzzle_identifiers",
        "arc_identifiers",
        "seq_offsets",
        "q_halt_logits",
        "preds",
    }
    
    def __init__(self, data_path: str, eval_metadata: PuzzleDatasetMetadata, submission_K: int = 2, pass_Ks: Sequence[int] = (1, 2, 5, 10, 100, 1000), aggregated_voting: bool = True):
        super().__init__()
        self.pass_Ks = pass_Ks
        self.submission_K = submission_K
        self.aggregated_voting = aggregated_voting
        self.blank_identifier_id = eval_metadata.blank_identifier_id

        # Majority vote evaluation settings
        self.maj_sample_sizes = (10, 100, 1000, 10000)

        # Load identifiers and test puzzles
        with open(os.path.join(data_path, "identifiers.json"), "r") as f:
            self.identifier_map = json.load(f)
        with open(os.path.join(data_path, "test_puzzles.json"), "r") as f:
            self.test_puzzles = json.load(f)
            
        # States
        self._local_hmap = {}
        self._local_preds = {}
        
    def begin_eval(self):
        if not self.aggregated_voting:
            # Clear previous predictions
            self._local_hmap = {}
            self._local_preds = {}

    @staticmethod
    def _grid_from_tokens(tokens: np.ndarray, position_ids: Optional[np.ndarray] = None) -> np.ndarray:
        if position_ids is None:
            return _crop(tokens)

        canvas = np.zeros((30, 30), dtype=np.int32)
        row_col = position_ids[:, -2:]
        rows = row_col[:, 0].astype(np.int64, copy=False)
        cols = row_col[:, 1].astype(np.int64, copy=False)
        valid = (rows >= 0) & (rows < 30) & (cols >= 0) & (cols < 30)
        canvas[rows[valid], cols[valid]] = tokens[valid]
        return _crop(canvas.reshape(-1))

    def _record_prediction(
        self,
        identifier: int,
        input_grid: np.ndarray,
        pred_grid: np.ndarray,
        q: float,
        q_log_prob: float,
    ) -> None:
        name = self.identifier_map[int(identifier)]
        orig_name, inverse_fn = inverse_aug(name)

        input_hash = grid_hash(inverse_fn(input_grid))
        pred = inverse_fn(pred_grid)
        assert np.all((pred >= 0) & (pred <= 9)), f"Puzzle {name}'s prediction out of 0-9 range."

        pred_hash = grid_hash(pred)
        self._local_hmap[pred_hash] = pred
        self._local_preds.setdefault(orig_name, {})
        self._local_preds[orig_name].setdefault(input_hash, [])
        self._local_preds[orig_name][input_hash].append((pred_hash, float(q), float(q_log_prob)))

    def _update_fixed_batch(
        self,
        outputs: Dict[str, torch.Tensor],
        q_values: torch.Tensor,
        q_log_probs: torch.Tensor,
    ) -> None:
        mask = outputs["puzzle_identifiers"] != self.blank_identifier_id
        input_key = "source_inputs" if "source_inputs" in outputs else "inputs"
        labels = outputs.get("labels")

        for row_idx, (identifier, input, pred, q, q_log_prob) in enumerate(zip(
            outputs["puzzle_identifiers"][mask].numpy(),
            outputs[input_key][mask].numpy(),
            outputs["preds"][mask].numpy(),
            q_values[mask].numpy(),
            q_log_probs[mask].numpy(),
        )):
            if labels is not None and pred.shape != input.shape:
                label = labels[mask][row_idx].numpy()
                target_mask = label != IGNORE_LABEL_ID
                if np.any(target_mask):
                    pred = pred[target_mask]
            self._record_prediction(
                int(identifier),
                self._grid_from_tokens(input),
                self._grid_from_tokens(pred),
                float(q),
                float(q_log_prob),
            )

    def _update_packed_full_batch(
        self,
        outputs: Dict[str, torch.Tensor],
        q_values: torch.Tensor,
        q_log_probs: torch.Tensor,
    ) -> None:
        identifiers = outputs.get("arc_identifiers")
        if identifiers is None:
            return

        seq_offsets = outputs["seq_offsets"].numpy()
        labels = outputs["labels"].numpy()
        position_ids = outputs["position_ids"].numpy()
        source_inputs = outputs.get("source_inputs", outputs["inputs"]).numpy()
        preds = outputs["preds"].numpy()

        for sample_idx, identifier in enumerate(identifiers.numpy()):
            start = int(seq_offsets[sample_idx])
            end = int(seq_offsets[sample_idx + 1])
            sample_labels = labels[start:end]
            sample_positions = position_ids[start:end]
            target_mask = sample_labels != IGNORE_LABEL_ID
            if not np.any(target_mask):
                continue

            target_pair_id = int(sample_positions[target_mask][0, 0])
            input_mask = (sample_positions[:, 0] == target_pair_id) & (sample_positions[:, 1] == 0)

            self._record_prediction(
                int(identifier),
                self._grid_from_tokens(source_inputs[start:end][input_mask], sample_positions[input_mask]),
                self._grid_from_tokens(preds[start:end][target_mask], sample_positions[target_mask]),
                float(q_values[sample_idx]),
                float(q_log_probs[sample_idx]),
            )

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        # Collect required outputs to CPU
        outputs = {}
        q_values = None
        q_log_probs = None

        for collection in (batch, preds):
            for k, v in collection.items():
                if k in self.required_outputs:
                    if k == "q_halt_logits":
                        q_values = v.to(torch.float64).sigmoid().cpu()
                        q_log_probs = F.logsigmoid(v.to(torch.float64)).cpu()
                    else:
                        outputs[k] = v.cpu()

        assert q_values is not None and q_log_probs is not None

        if "seq_offsets" in outputs and outputs["labels"].ndim == 1:
            self._update_packed_full_batch(outputs, q_values, q_log_probs)
        else:
            self._update_fixed_batch(outputs, q_values, q_log_probs)
    
    def result(self, save_path: Optional[str], rank: int, world_size: int, group: Optional[torch.distributed.ProcessGroup] = None) -> Optional[Dict[str, float]]:
        # Gather predictions to rank 0 for voting
        if world_size == 1 and not dist.is_initialized():
            global_hmap_preds = [(self._local_hmap, self._local_preds)]
        else:
            global_hmap_preds = [None for _ in range(world_size)] if rank == 0 else None
            dist.gather_object((self._local_hmap, self._local_preds), global_hmap_preds, dst=0, group=group)
        
        # Rank 0 logic
        if rank != 0:
            return

        submission = {}
        correct = [0.0 for _ in range(len(self.pass_Ks))]
        cons_available = [0.0 for _ in range(len(self.pass_Ks))]
        cons_pass_correct = [0.0 for _ in range(len(self.pass_Ks))]
        cons_rank_correct = [0.0 for _ in range(len(self.pass_Ks))]
        maj_correct = {size: 0.0 for size in self.maj_sample_sizes}
        missing_predictions = 0

        for name, puzzle in self.test_puzzles.items():
            # Process test examples in this puzzle
            submission[name] = []
            num_test_correct = [0 for _ in range(len(self.pass_Ks))]
            cons_test_available = [0 for _ in range(len(self.pass_Ks))]
            cons_test_pass_correct = [0 for _ in range(len(self.pass_Ks))]
            cons_rank_test_correct = [0 for _ in range(len(self.pass_Ks))]
            maj_test_correct = {size: 0 for size in self.maj_sample_sizes}
            for pair in puzzle["test"]:
                input_hash = grid_hash(arc_grid_to_np(pair["input"]))
                label_hash = grid_hash(arc_grid_to_np(pair["output"]))

                p_map = {}
                pred_samples = []
                for hmap, preds in global_hmap_preds:  # type: ignore
                    for h, q, q_log_prob in preds.get(name, {}).get(input_hash, []):
                        p_map.setdefault(h, [0, 0.0, -np.inf])
                        p_map[h][0] += 1
                        p_map[h][1] += q
                        p_map[h][2] = max(p_map[h][2], q_log_prob)
                        pred_samples.append((h, q_log_prob))

                if not len(p_map):
                    missing_predictions += 1
                    continue

                for h, stats in p_map.items():
                    stats[1] /= stats[0]

                p_map = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)
                cons_map = sorted(p_map, key=lambda kv: (kv[1][0], kv[1][1], kv[1][2]), reverse=True)

                # vote for different Ks
                for i, k in enumerate(self.pass_Ks):
                    ok = False
                    for h, stats in p_map[:k]:
                        ok |= h == label_hash
                        
                    num_test_correct[i] += ok

                    cons_candidates = [(h, stats) for h, stats in cons_map if stats[0] >= k]
                    cons_test_available[i] += len(cons_candidates) > 0
                    cons_test_pass_correct[i] += any(h == label_hash for h, _stats in cons_candidates)
                    cons_rank_test_correct[i] += any(h == label_hash for h, _stats in cons_map[:k])

                # Query grids
                pred_grids = []
                for h, stats in p_map[:self.submission_K]:
                    for hmap, preds in global_hmap_preds:  # type: ignore
                        if h in hmap:
                            pred_grids.append(hmap[h])
                            break

                # Pad to K
                while len(pred_grids) < self.submission_K:
                    pred_grids.append(pred_grids[0])

                submission[name].append({f"attempt_{i + 1}": grid.tolist() for i, grid in enumerate(pred_grids)})

                # Majority voting metrics (best-of-N with log-probability ranking)
                if len(pred_samples):
                    logps = np.array([lp for _, lp in pred_samples], dtype=np.float64)
                    max_logp = logps.max()
                    probs = np.exp(logps - max_logp)
                    prob_sum = probs.sum()
                    if prob_sum > 0:
                        probs /= prob_sum
                    else:
                        probs = np.full_like(probs, 1.0 / len(probs))

                    for sample_size in self.maj_sample_sizes:
                        sampled_indices = np.random.choice(len(pred_samples), size=sample_size, replace=True, p=probs)
                        sampled_logps = logps[sampled_indices]
                        best_idx = sampled_indices[np.argmax(sampled_logps)]
                        maj_test_correct[sample_size] += pred_samples[best_idx][0] == label_hash

            # Total correctness
            for i in range(len(self.pass_Ks)):
                correct[i] += num_test_correct[i] / len(puzzle["test"])
                cons_available[i] += cons_test_available[i] / len(puzzle["test"])
                cons_pass_correct[i] += cons_test_pass_correct[i] / len(puzzle["test"])
                cons_rank_correct[i] += cons_rank_test_correct[i] / len(puzzle["test"])
            for sample_size in self.maj_sample_sizes:
                maj_correct[sample_size] += maj_test_correct[sample_size] / len(puzzle["test"])

        # Save submission
        if save_path is not None:
            with open(os.path.join(save_path, "submission.json"), "w") as f:
                json.dump(submission, f)

        if missing_predictions:
            print(f"ARC evaluator skipped {missing_predictions} test pair(s) with no predictions.")

        # Final result
        result = {f"{self.__class__.__name__}/pass@{k}": correct[i] / len(self.test_puzzles) for i, k in enumerate(self.pass_Ks)}
        result.update({f"{self.__class__.__name__}/cons@{k}": cons_available[i] / len(self.test_puzzles) for i, k in enumerate(self.pass_Ks)})
        result.update({f"{self.__class__.__name__}/cons_pass@{k}": cons_pass_correct[i] / len(self.test_puzzles) for i, k in enumerate(self.pass_Ks)})
        result.update({f"{self.__class__.__name__}/cons_rank_pass@{k}": cons_rank_correct[i] / len(self.test_puzzles) for i, k in enumerate(self.pass_Ks)})
        result.update({f"{self.__class__.__name__}/cons@{k}/pass@{k}": cons_rank_correct[i] / len(self.test_puzzles) for i, k in enumerate(self.pass_Ks)})
        result.update({f"{self.__class__.__name__}/maj@{k}": maj_correct[k] / len(self.test_puzzles) for k in self.maj_sample_sizes})
        return result
