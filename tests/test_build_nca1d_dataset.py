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
    unflatten_time_volume,
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
                state_height_min=6,
                state_height_max=6,
                num_colors=4,
                patch_size=1,
                rollout_steps=5,
                rollout_steps_min=5,
                rollout_steps_max=5,
                counts=3,
                counts_min=3,
                counts_max=3,
                time_subsample=1,
                gzip_threshold_low=None,
                gzip_threshold_high=None,
                target_mask_ratio=0.4,
                batch_candidate_size=4,
                max_sampling_rounds=20,
                save_dtype="int32",
            )

            build_dataset(config)

            with open(dataset_dir / "config.json") as f:
                saved_config = json.load(f)
            self.assertEqual(saved_config["final_image_shape"], [6, 5, 3])
            self.assertEqual(saved_config["position_id_shape"], [3, 6, 5])
            self.assertEqual(saved_config["seq_len"], 90)
            self.assertEqual(saved_config["vocab_size"], 6)
            self.assertEqual(saved_config["resolved_state_height_min"], 6)
            self.assertEqual(saved_config["resolved_state_height_max"], 6)
            self.assertEqual(saved_config["resolved_rollout_steps_min"], 5)
            self.assertEqual(saved_config["resolved_rollout_steps_max"], 5)
            self.assertEqual(saved_config["resolved_counts_max"], 3)
            self.assertNotIn("time_end", saved_config)

            train_inputs = np.load(dataset_dir / "train" / "all__inputs.npy")
            train_labels = np.load(dataset_dir / "train" / "all__labels.npy")
            train_position_ids = np.load(dataset_dir / "train" / "all__position_ids.npy")
            train_gzip = np.load(dataset_dir / "train" / "all__gzip_ratio.npy")
            train_puzzle_indices = np.load(dataset_dir / "train" / "all__puzzle_indices.npy")
            train_group_indices = np.load(dataset_dir / "train" / "all__group_indices.npy")
            train_state_heights = np.load(dataset_dir / "train" / "all__state_heights.npy")
            train_num_frames = np.load(dataset_dir / "train" / "all__num_frames.npy")
            train_rollout_steps = np.load(dataset_dir / "train" / "all__rollout_steps.npy")
            train_counts = np.load(dataset_dir / "train" / "all__counts.npy")

            with open(dataset_dir / "train" / "dataset.json") as f:
                train_metadata = json.load(f)

            self.assertEqual(train_inputs.shape, (3, config.seq_len))
            self.assertEqual(train_labels.shape, (3, config.seq_len))
            self.assertEqual(train_position_ids.shape, (3, config.seq_len, 3))
            self.assertEqual(train_gzip.shape, (3,))
            self.assertEqual(train_puzzle_indices.tolist(), [0, 1, 2, 3])
            self.assertEqual(train_group_indices.tolist(), [0, 1, 2, 3])
            self.assertEqual(train_state_heights.tolist(), [6, 6, 6])
            self.assertEqual(train_num_frames.tolist(), [5, 5, 5])
            self.assertEqual(train_rollout_steps.tolist(), [5, 5, 5])
            self.assertEqual(train_counts.tolist(), [3, 3, 3])
            self.assertEqual(train_metadata["position_id_shape"], [3, 6, 5])
            self.assertTrue(np.all((train_inputs == 0) | (train_inputs == config.mask_token_id) | (train_inputs >= config.token_offset)))
            self.assertTrue(np.all((train_labels == 0) | (train_labels >= config.token_offset)))

            decoded = unflatten_time_volume(
                train_inputs[0],
                image_height=config.state_height,
                num_frames=config.num_frames,
                counts=config.resolved_counts_max,
                token_offset=config.token_offset,
                padded_height=config.final_image_shape[0],
                padded_num_frames=config.final_image_shape[1],
                padded_counts=config.final_image_shape[2],
            )
            self.assertEqual(decoded.shape, config.final_image_shape)
            self.assertTrue(np.all(decoded <= config.num_colors - 1))
            self.assertTrue(np.all(decoded >= config.mask_token_id - config.token_offset))

            input_canvas = train_inputs[0].reshape(config.final_image_shape)
            label_canvas = train_labels[0].reshape(config.final_image_shape)
            self.assertTrue(np.all(label_canvas[:, :, :-1] == 0))
            self.assertTrue(np.all(label_canvas[:, :, -1] >= config.token_offset))
            self.assertGreater(int(np.count_nonzero(input_canvas[:, :, -1] == config.mask_token_id)), 0)
            self.assertTrue(np.all(input_canvas[:, :, :-1] >= config.token_offset))

            position_grid = train_position_ids[0].reshape(*config.final_image_shape, 3)
            np.testing.assert_array_equal(position_grid[0, 0, 0], np.array([0, 0, 0], dtype=np.int32))
            np.testing.assert_array_equal(position_grid[5, 4, 2], np.array([2, 5, 4], dtype=np.int32))
            np.testing.assert_array_equal(position_grid[5, 4, 1], np.array([1, 5, 4], dtype=np.int32))

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
            self.assertEqual(tuple(batch["position_ids"].shape), (2, config.seq_len, 3))
            self.assertEqual(tuple(batch["puzzle_identifiers"].shape), (2,))

    def test_build_dataset_supports_random_state_heights_and_rollout_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "nca1d-variable"
            config = NCA1DDataConfig(
                output_dir=str(dataset_dir),
                train_size=8,
                test_size=4,
                seed=13,
                state_height=6,
                state_height_min=4,
                state_height_max=8,
                num_colors=4,
                patch_size=1,
                rollout_steps=6,
                rollout_steps_min=3,
                rollout_steps_max=7,
                counts=4,
                counts_min=2,
                counts_max=5,
                time_subsample=1,
                gzip_threshold_low=None,
                gzip_threshold_high=None,
                target_mask_ratio=0.25,
                batch_candidate_size=4,
                max_sampling_rounds=20,
                save_dtype="int32",
            )

            build_dataset(config)

            with open(dataset_dir / "config.json") as f:
                saved_config = json.load(f)

            self.assertEqual(saved_config["resolved_state_height_min"], 4)
            self.assertEqual(saved_config["resolved_state_height_max"], 8)
            self.assertEqual(saved_config["resolved_rollout_steps_min"], 3)
            self.assertEqual(saved_config["resolved_rollout_steps_max"], 7)
            self.assertEqual(saved_config["final_image_shape"], [8, 7, 5])
            self.assertEqual(saved_config["position_id_shape"], [5, 8, 7])
            self.assertEqual(saved_config["seq_len"], 280)

            train_inputs = np.load(dataset_dir / "train" / "all__inputs.npy")
            train_state_heights = np.load(dataset_dir / "train" / "all__state_heights.npy")
            train_num_frames = np.load(dataset_dir / "train" / "all__num_frames.npy")
            train_rollout_steps = np.load(dataset_dir / "train" / "all__rollout_steps.npy")
            train_counts = np.load(dataset_dir / "train" / "all__counts.npy")

            self.assertEqual(train_inputs.shape, (8, 280))
            self.assertTrue(np.all(train_state_heights >= 4))
            self.assertTrue(np.all(train_state_heights <= 8))
            self.assertTrue(np.all(train_rollout_steps >= 3))
            self.assertTrue(np.all(train_rollout_steps <= 7))
            self.assertTrue(np.all(train_counts >= 2))
            self.assertTrue(np.all(train_counts <= 5))
            self.assertTrue(np.all(train_num_frames >= 3))
            self.assertTrue(np.all(train_num_frames <= 7))

            self.assertGreater(len(np.unique(train_state_heights)), 1)
            self.assertGreater(len(np.unique(train_rollout_steps)), 1)
            self.assertGreater(len(np.unique(train_counts)), 1)

            decoded = unflatten_time_volume(
                train_inputs[0],
                image_height=int(train_state_heights[0]),
                num_frames=int(train_num_frames[0]),
                counts=int(train_counts[0]),
                token_offset=config.token_offset,
                padded_height=config.final_image_shape[0],
                padded_num_frames=config.final_image_shape[1],
                padded_counts=config.final_image_shape[2],
            )
            self.assertEqual(
                decoded.shape,
                (
                    int(train_state_heights[0]),
                    int(train_num_frames[0]),
                    int(train_counts[0]),
                ),
            )
            self.assertTrue(np.all(decoded <= config.num_colors - 1))
            self.assertTrue(np.all(decoded >= config.mask_token_id - config.token_offset))

    def test_counts_are_capped_to_respect_max_data_seq_len(self) -> None:
        config = NCA1DDataConfig(
            train_size=1,
            test_size=1,
            state_height=30,
            patch_size=1,
            rollout_steps=30,
            counts=20,
            counts_max=20,
            max_data_seq_len=9000,
        )

        self.assertEqual(config.resolved_counts_max, 9)
        self.assertEqual(config.final_image_shape, (30, 30, 9))
        self.assertEqual(config.seq_len, 8100)

    def test_time_start_sets_sampling_start_without_time_end(self) -> None:
        config = NCA1DDataConfig(
            train_size=1,
            test_size=1,
            time_start=30,
            start_step=0,
            rollout_steps=9,
            rollout_steps_min=6,
            rollout_steps_max=14,
        )

        self.assertEqual(config.resolved_start_step, 30)
        self.assertEqual(config.sampled_rollout_steps_min, 6)
        self.assertEqual(config.sampled_rollout_steps_max, 14)


if __name__ == "__main__":
    unittest.main()
