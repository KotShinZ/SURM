from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.build_nca1d_dataset import (  # noqa: E402
    NCA1DDataConfig,
    build_dataset,
    trajectory_to_time_image,
    unflatten_time_image,
)
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig  # noqa: E402


class BuildNCA1DDatasetTests(unittest.TestCase):
    def test_trajectory_to_time_image_converts_h_by_1_frames_to_h_by_t_image(self) -> None:
        trajectory = np.array(
            [
                [[0], [1], [2]],
                [[3], [4], [5]],
                [[6], [7], [8]],
                [[9], [10], [11]],
            ],
            dtype=np.int16,
        )

        time_image = trajectory_to_time_image(trajectory)

        self.assertEqual(time_image.shape, (3, 4))
        np.testing.assert_array_equal(
            time_image,
            np.array(
                [
                    [0, 3, 6, 9],
                    [1, 4, 7, 10],
                    [2, 5, 8, 11],
                ],
                dtype=np.int16,
            ),
        )

    def test_build_dataset_writes_puzzle_dataset_compatible_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "nca1d"
            config = NCA1DDataConfig(
                output_dir=str(dataset_dir),
                train_size=3,
                test_size=2,
                seed=7,
                state_height=6,
                num_colors=4,
                patch_size=1,
                rollout_steps=5,
                time_subsample=1,
                batch_candidate_size=4,
                max_sampling_rounds=20,
                save_dtype="int32",
            )

            build_dataset(config)

            with open(dataset_dir / "config.json") as f:
                saved_config = json.load(f)
            self.assertEqual(saved_config["final_image_shape"], [6, 5])
            self.assertEqual(saved_config["seq_len"], 30)
            self.assertEqual(saved_config["vocab_size"], 6)

            train_inputs = np.load(dataset_dir / "train" / "all__inputs.npy")
            train_labels = np.load(dataset_dir / "train" / "all__labels.npy")
            train_gzip = np.load(dataset_dir / "train" / "all__gzip_ratio.npy")
            train_puzzle_indices = np.load(dataset_dir / "train" / "all__puzzle_indices.npy")
            train_group_indices = np.load(dataset_dir / "train" / "all__group_indices.npy")

            self.assertEqual(train_inputs.shape, (3, config.seq_len))
            self.assertEqual(train_labels.shape, (3, config.seq_len))
            self.assertEqual(train_gzip.shape, (3,))
            self.assertEqual(train_puzzle_indices.tolist(), [0, 1, 2, 3])
            self.assertEqual(train_group_indices.tolist(), [0, 1, 2, 3])
            self.assertTrue(np.all(train_inputs >= config.token_offset))
            self.assertTrue(np.all(train_inputs <= config.token_offset + config.num_colors - 1))
            np.testing.assert_array_equal(train_inputs, train_labels)

            decoded = unflatten_time_image(
                train_inputs[0],
                image_height=config.state_height,
                num_frames=config.num_frames,
                token_offset=config.token_offset,
            )
            self.assertEqual(decoded.shape, config.final_image_shape)
            self.assertTrue(np.all(decoded >= 0))
            self.assertTrue(np.all(decoded < config.num_colors))

            dataset = PuzzleDataset(
                PuzzleDatasetConfig(
                    seed=3,
                    dataset_path=str(dataset_dir),
                    global_batch_size=2,
                    test_set_mode=True,
                    epochs_per_iter=1,
                    rank=0,
                    num_replicas=1,
                ),
                split="train",
            )
            set_name, batch, effective_batch_size = next(iter(dataset))

            self.assertEqual(set_name, "all")
            self.assertEqual(effective_batch_size, 2)
            self.assertEqual(tuple(batch["inputs"].shape), (2, config.seq_len))
            self.assertEqual(tuple(batch["labels"].shape), (2, config.seq_len))
            self.assertEqual(tuple(batch["puzzle_identifiers"].shape), (2,))


if __name__ == "__main__":
    unittest.main()
