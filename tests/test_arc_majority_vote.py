from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.common import PuzzleDatasetMetadata  # noqa: E402
from evaluators.arc_majority_vote import (  # noqa: E402
    default_arc_submission_dir,
    maybe_create_arc_majority_vote_evaluator,
)


def _write_arc_metadata_files(dataset_root: Path) -> None:
    (dataset_root / "identifiers.json").write_text(
        json.dumps(["", "task_1"]),
        encoding="utf-8",
    )
    (dataset_root / "test_puzzles.json").write_text(
        json.dumps(
            {
                "task_1": {
                    "test": [
                        {
                            "input": [[0]],
                            "output": [[1]],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )


def _metadata() -> PuzzleDatasetMetadata:
    return PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        vocab_size=12,
        seq_len=900,
        num_puzzle_identifiers=2,
        total_groups=1,
        mean_puzzle_examples=1.0,
        sets=["all"],
    )


def _encoded_grid(value: int) -> torch.Tensor:
    grid = torch.zeros((1, 900), dtype=torch.int64)
    grid[0, 0] = value
    return grid


class ArcMajorityVoteTests(unittest.TestCase):
    def test_finalize_returns_expected_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            _write_arc_metadata_files(dataset_root)

            evaluator = maybe_create_arc_majority_vote_evaluator(
                data_path=str(dataset_root),
                eval_metadata=_metadata(),
                split="test",
            )

            self.assertIsNotNone(evaluator)
            assert evaluator is not None

            batch = {
                "inputs": _encoded_grid(2),
                "puzzle_identifiers": torch.tensor([1], dtype=torch.int64),
            }
            preds = {
                "preds": _encoded_grid(3),
                "q_halt_logits": torch.tensor([0.75], dtype=torch.float32),
            }

            evaluator.update_batch(batch, preds)

            save_dir = dataset_root / "arc_eval"
            metrics = evaluator.result(str(save_dir))

            self.assertEqual(metrics["ARC/pass@1"], 1.0)
            self.assertEqual(metrics["ARC/pass@1000"], 1.0)
            self.assertEqual(metrics["ARC/maj@10"], 1.0)
            self.assertTrue((save_dir / "submission.json").is_file())

    def test_detection_requires_test_split_and_metadata_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            self.assertIsNone(
                maybe_create_arc_majority_vote_evaluator(
                    data_path=str(dataset_root),
                    eval_metadata=_metadata(),
                    split="test",
                )
            )

            _write_arc_metadata_files(dataset_root)

            self.assertIsNone(
                maybe_create_arc_majority_vote_evaluator(
                    data_path=str(dataset_root),
                    eval_metadata=_metadata(),
                    split="validation",
                )
            )

    def test_default_arc_submission_dir_uses_summary_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "result.json"
            self.assertEqual(
                default_arc_submission_dir(summary_path),
                Path(tmp_dir) / "result_arc_majority_vote",
            )


if __name__ == "__main__":
    unittest.main()
