from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from puzzle_dataset import PuzzleDatasetConfig, PuzzleDatasetSeparate  # noqa: E402


def _write_variable_dataset(root: Path) -> None:
    split_dir = root / "train"
    split_dir.mkdir(parents=True)
    metadata = {
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "vocab_size": 12,
        "seq_len": 9,
        "num_puzzle_identifiers": 2,
        "total_groups": 1,
        "mean_puzzle_examples": 2.0,
        "sets": ["all"],
        "variable_seq_lengths": True,
        "position_id_shape": None,
    }
    (split_dir / "dataset.json").write_text(json.dumps(metadata), encoding="utf-8")

    np.save(split_dir / "all__inputs.npy", np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint8))
    np.save(split_dir / "all__labels.npy", np.array([2, 0, 3, 1, 4, 5, 0, 1], dtype=np.uint8))
    np.save(split_dir / "all__seq_offsets.npy", np.array([0, 4, 8], dtype=np.int64))
    np.save(split_dir / "all__label_seq_offsets.npy", np.array([0, 4, 8], dtype=np.int64))
    np.save(split_dir / "all__seq_shapes.npy", np.array([[2, 2], [2, 2]], dtype=np.int32))
    np.save(split_dir / "all__label_seq_shapes.npy", np.array([[2, 2], [2, 2]], dtype=np.int32))
    np.save(split_dir / "all__puzzle_indices.npy", np.array([0, 2], dtype=np.int32))
    np.save(split_dir / "all__group_indices.npy", np.array([0, 1], dtype=np.int32))
    np.save(split_dir / "all__puzzle_identifiers.npy", np.array([1], dtype=np.int32))


def _separate_config(dataset_path: Path, **overrides) -> PuzzleDatasetConfig:
    values = dict(
        seed=0,
        dataset_path=str(dataset_path),
        global_batch_size=1,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
        label_separate=True,
        SeparateMode="D",
    )
    values.update(overrides)
    return PuzzleDatasetConfig(**values)


class PuzzleDatasetSeparateTests(unittest.TestCase):
    def test_variable_dataset_infers_slot_position_shape_from_generated_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir)
            _write_variable_dataset(dataset_path)

            dataset = PuzzleDatasetSeparate(_separate_config(dataset_path), split="train")

            self.assertEqual(dataset.metadata.position_id_shape, [2, 3, 3])

            _set_name, batch, _effective_batch_size = next(iter(dataset))
            self.assertEqual(tuple(batch["position_ids"].shape), (8, 3))
            self.assertEqual(
                batch["position_ids"][:4].tolist(),
                [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]],
            )
            self.assertEqual(
                batch["position_ids"][4:].tolist(),
                [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
            )

    def test_mode_d_training_answer_tokens_mix_labels_and_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir)
            _write_variable_dataset(dataset_path)
            dataset = PuzzleDatasetSeparate(
                _separate_config(
                    dataset_path,
                    label_separate_noise_token_min=9,
                    label_separate_noise_token_max=9,
                ),
                split="train",
            )
            labels = torch.full((200,), 3, dtype=torch.int32)

            answer_tokens = dataset._make_answer_tokens(
                labels,
                np.random.Generator(np.random.Philox(seed=0)),
                training=True,
                sample_shape=(),
            )

            self.assertEqual(set(answer_tokens.tolist()), {3, 9})


if __name__ == "__main__":
    unittest.main()
