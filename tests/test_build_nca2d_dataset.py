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

from data.build_nca2d_dataset import (  # noqa: E402
    NCA2DDataConfig,
    build_dataset,
    extract_label_grid,
    get_device,
    make_nca_config_for_sample,
    pair_windows_to_input_label_grids,
    rollout_pair_windows_batched,
    rollout_trajectories_batched,
    seed_everything,
    unflatten_input_grid,
)
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig  # noqa: E402


class BuildNCA2DDatasetTests(unittest.TestCase):
    def test_pair_windows_to_input_label_grids_interleaves_examples_and_query(self) -> None:
        pair_windows = np.array(
            [
                [
                    [[1, 2], [3, 4]],
                    [[5, 6], [7, 8]],
                ],
                [
                    [[9, 10], [11, 12]],
                    [[13, 14], [15, 16]],
                ],
                [
                    [[17, 18], [19, 20]],
                    [[21, 22], [23, 24]],
                ],
            ],
            dtype=np.int16,
        )

        input_grid, label_grid = pair_windows_to_input_label_grids(pair_windows)

        self.assertEqual(input_grid.shape, (2, 2, 5))
        self.assertEqual(label_grid.shape, (2, 2, 1))
        np.testing.assert_array_equal(
            input_grid,
            np.array(
                [
                    [[1, 5, 9, 13, 17], [2, 6, 10, 14, 18]],
                    [[3, 7, 11, 15, 19], [4, 8, 12, 16, 20]],
                ],
                dtype=np.int16,
            ),
        )
        np.testing.assert_array_equal(
            label_grid[:, :, 0],
            np.array(
                [
                    [21, 22],
                    [23, 24],
                ],
                dtype=np.int16,
            ),
        )

    def test_build_dataset_writes_puzzle_dataset_compatible_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "nca2d"
            config = NCA2DDataConfig(
                output_dir=str(dataset_dir),
                train_size=3,
                test_size=2,
                seed=7,
                state_height=4,
                state_height_min=4,
                state_height_max=4,
                state_width=5,
                state_width_min=5,
                state_width_max=5,
                num_colors=4,
                patch_size=1,
                answer_steps=3,
                answer_steps_min=3,
                answer_steps_max=3,
                counts=2,
                counts_min=2,
                counts_max=2,
                time_start=12,
                time_span=1,
                gzip_threshold_low=None,
                gzip_threshold_high=None,
                batch_candidate_size=8,
                max_sampling_rounds=20,
                save_dtype="int32",
            )

            build_dataset(config)

            with open(dataset_dir / "config.json") as f:
                saved_config = json.load(f)
            self.assertEqual(saved_config["final_image_shape"], [4, 5, 5])
            self.assertEqual(saved_config["position_id_shape"], [5, 4, 5])
            self.assertEqual(saved_config["seq_len"], 100)
            self.assertEqual(saved_config["vocab_size"], 6)
            self.assertEqual(saved_config["resolved_out_colors"], 4)
            self.assertEqual(saved_config["resolved_state_height_min"], 4)
            self.assertEqual(saved_config["resolved_state_height_max"], 4)
            self.assertEqual(saved_config["resolved_state_width_min"], 5)
            self.assertEqual(saved_config["resolved_state_width_max"], 5)
            self.assertEqual(saved_config["resolved_answer_steps_min"], 3)
            self.assertEqual(saved_config["resolved_answer_steps_max"], 3)
            self.assertEqual(saved_config["resolved_counts_max"], 2)

            train_inputs = np.load(dataset_dir / "train" / "all__inputs.npy")
            train_labels = np.load(dataset_dir / "train" / "all__labels.npy")
            train_position_ids = np.load(dataset_dir / "train" / "all__position_ids.npy")
            train_gzip = np.load(dataset_dir / "train" / "all__gzip_ratio.npy")
            train_puzzle_indices = np.load(dataset_dir / "train" / "all__puzzle_indices.npy")
            train_group_indices = np.load(dataset_dir / "train" / "all__group_indices.npy")
            train_state_heights = np.load(dataset_dir / "train" / "all__state_heights.npy")
            train_state_widths = np.load(dataset_dir / "train" / "all__state_widths.npy")
            train_answer_steps = np.load(dataset_dir / "train" / "all__answer_steps.npy")
            train_counts = np.load(dataset_dir / "train" / "all__counts.npy")
            train_input_channels = np.load(dataset_dir / "train" / "all__input_channels.npy")
            train_query_channel_indices = np.load(dataset_dir / "train" / "all__query_channel_indices.npy")

            with open(dataset_dir / "train" / "dataset.json") as f:
                train_metadata = json.load(f)

            self.assertEqual(train_inputs.shape, (3, config.seq_len))
            self.assertEqual(train_labels.shape, (3, config.seq_len))
            self.assertEqual(train_position_ids.shape, (3, config.seq_len, 3))
            self.assertEqual(train_gzip.shape, (3,))
            self.assertTrue(np.all(train_gzip == -1.0))
            self.assertEqual(train_puzzle_indices.tolist(), [0, 1, 2, 3])
            self.assertEqual(train_group_indices.tolist(), [0, 1, 2, 3])
            self.assertEqual(train_state_heights.tolist(), [4, 4, 4])
            self.assertEqual(train_state_widths.tolist(), [5, 5, 5])
            self.assertEqual(train_answer_steps.tolist(), [3, 3, 3])
            self.assertEqual(train_counts.tolist(), [2, 2, 2])
            self.assertEqual(train_input_channels.tolist(), [5, 5, 5])
            self.assertEqual(train_query_channel_indices.tolist(), [4, 4, 4])
            self.assertEqual(train_metadata["position_id_shape"], [5, 4, 5])
            self.assertTrue(np.all((train_inputs == 0) | (train_inputs >= config.token_offset)))
            self.assertTrue(np.all((train_labels == 0) | (train_labels >= config.token_offset)))

            decoded_input = unflatten_input_grid(
                train_inputs[0],
                image_height=4,
                image_width=5,
                counts=2,
                token_offset=config.token_offset,
                padded_height=config.final_image_shape[0],
                padded_width=config.final_image_shape[1],
                padded_channels=config.final_image_shape[2],
            )
            decoded_label = extract_label_grid(
                train_labels[0],
                image_height=4,
                image_width=5,
                counts=2,
                token_offset=config.token_offset,
                padded_height=config.final_image_shape[0],
                padded_width=config.final_image_shape[1],
                padded_channels=config.final_image_shape[2],
            )
            self.assertEqual(decoded_input.shape, (4, 5, 5))
            self.assertEqual(decoded_label.shape, (4, 5, 1))
            self.assertTrue(np.all(decoded_input <= config.resolved_out_colors - 1))
            self.assertTrue(np.all(decoded_input >= 0))
            self.assertTrue(np.all(decoded_label <= config.resolved_out_colors - 1))
            self.assertTrue(np.all(decoded_label >= 0))

            input_canvas = train_inputs[0].reshape(config.final_image_shape)
            label_canvas = train_labels[0].reshape(config.final_image_shape)
            self.assertTrue(np.all(label_canvas[:, :, :-1] == 0))
            self.assertTrue(np.all(label_canvas[:, :, -1] >= config.token_offset))
            self.assertTrue(np.all(input_canvas[:, :, :-1] >= config.token_offset))
            self.assertTrue(np.all(input_canvas[:, :, -1] >= config.token_offset))

            position_grid = train_position_ids[0].reshape(*config.final_image_shape, 3)
            np.testing.assert_array_equal(position_grid[0, 0, 0], np.array([0, 0, 0], dtype=np.int32))
            np.testing.assert_array_equal(position_grid[3, 4, 4], np.array([4, 3, 4], dtype=np.int32))
            np.testing.assert_array_equal(position_grid[2, 1, 3], np.array([3, 2, 1], dtype=np.int32))

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

    def test_build_dataset_supports_random_heights_widths_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "nca2d-variable"
            config = NCA2DDataConfig(
                output_dir=str(dataset_dir),
                train_size=18,
                test_size=8,
                seed=13,
                state_height=6,
                state_height_min=4,
                state_height_max=6,
                state_width=7,
                state_width_min=5,
                state_width_max=7,
                num_colors=4,
                patch_size=1,
                answer_steps=4,
                answer_steps_min=2,
                answer_steps_max=4,
                counts=3,
                counts_min=1,
                counts_max=3,
                time_span=1,
                gzip_threshold_low=None,
                gzip_threshold_high=None,
                batch_candidate_size=32,
                max_sampling_rounds=20,
                save_dtype="int32",
            )

            build_dataset(config)

            with open(dataset_dir / "config.json") as f:
                saved_config = json.load(f)

            self.assertEqual(saved_config["resolved_state_height_min"], 4)
            self.assertEqual(saved_config["resolved_state_height_max"], 6)
            self.assertEqual(saved_config["resolved_state_width_min"], 5)
            self.assertEqual(saved_config["resolved_state_width_max"], 7)
            self.assertEqual(saved_config["resolved_answer_steps_min"], 2)
            self.assertEqual(saved_config["resolved_answer_steps_max"], 4)
            self.assertEqual(saved_config["resolved_out_colors"], 4)
            self.assertEqual(saved_config["final_image_shape"], [6, 7, 7])
            self.assertEqual(saved_config["position_id_shape"], [7, 6, 7])
            self.assertEqual(saved_config["seq_len"], 294)

            train_inputs = np.load(dataset_dir / "train" / "all__inputs.npy")
            train_state_heights = np.load(dataset_dir / "train" / "all__state_heights.npy")
            train_state_widths = np.load(dataset_dir / "train" / "all__state_widths.npy")
            train_answer_steps = np.load(dataset_dir / "train" / "all__answer_steps.npy")
            train_counts = np.load(dataset_dir / "train" / "all__counts.npy")
            train_input_channels = np.load(dataset_dir / "train" / "all__input_channels.npy")

            self.assertEqual(train_inputs.shape, (18, 294))
            self.assertTrue(np.all(train_state_heights >= 4))
            self.assertTrue(np.all(train_state_heights <= 6))
            self.assertTrue(np.all(train_state_widths >= 5))
            self.assertTrue(np.all(train_state_widths <= 7))
            self.assertTrue(np.all(train_answer_steps >= 2))
            self.assertTrue(np.all(train_answer_steps <= 4))
            self.assertTrue(np.all(train_counts >= 1))
            self.assertTrue(np.all(train_counts <= 3))
            np.testing.assert_array_equal(train_input_channels, 2 * train_counts + 1)

            self.assertGreater(len(np.unique(train_state_heights)), 1)
            self.assertGreater(len(np.unique(train_state_widths)), 1)
            self.assertGreater(len(np.unique(train_answer_steps)), 1)
            self.assertGreater(len(np.unique(train_counts)), 1)

            decoded_input = unflatten_input_grid(
                train_inputs[0],
                image_height=int(train_state_heights[0]),
                image_width=int(train_state_widths[0]),
                counts=int(train_counts[0]),
                token_offset=config.token_offset,
                padded_height=config.final_image_shape[0],
                padded_width=config.final_image_shape[1],
                padded_channels=config.final_image_shape[2],
            )
            self.assertEqual(
                decoded_input.shape,
                (
                    int(train_state_heights[0]),
                    int(train_state_widths[0]),
                    int(train_input_channels[0]),
                ),
            )
            self.assertTrue(np.all(decoded_input <= config.resolved_out_colors - 1))
            self.assertTrue(np.all(decoded_input >= 0))

    def test_build_dataset_can_remap_internal_colors_into_larger_output_palette(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "nca2d-remap"
            config = NCA2DDataConfig(
                output_dir=str(dataset_dir),
                train_size=6,
                test_size=2,
                seed=19,
                state_height=6,
                state_height_min=6,
                state_height_max=6,
                state_width=6,
                state_width_min=6,
                state_width_max=6,
                num_colors=3,
                out_colors=10,
                patch_size=1,
                answer_steps=2,
                answer_steps_min=2,
                answer_steps_max=2,
                counts=2,
                counts_min=2,
                counts_max=2,
                gzip_threshold_low=None,
                gzip_threshold_high=None,
                batch_candidate_size=8,
                max_sampling_rounds=20,
                save_dtype="int32",
            )

            build_dataset(config)

            with open(dataset_dir / "config.json") as f:
                saved_config = json.load(f)
            self.assertEqual(saved_config["resolved_out_colors"], 10)
            self.assertEqual(saved_config["vocab_size"], 12)

            train_inputs = np.load(dataset_dir / "train" / "all__inputs.npy")
            train_labels = np.load(dataset_dir / "train" / "all__labels.npy")
            train_counts = np.load(dataset_dir / "train" / "all__counts.npy")

            saw_extended_palette = False
            for sample_idx in range(train_inputs.shape[0]):
                counts = int(train_counts[sample_idx])
                decoded_input = unflatten_input_grid(
                    train_inputs[sample_idx],
                    image_height=config.state_height,
                    image_width=config.state_width,
                    counts=counts,
                    token_offset=config.token_offset,
                    padded_height=config.final_image_shape[0],
                    padded_width=config.final_image_shape[1],
                    padded_channels=config.final_image_shape[2],
                )
                decoded_label = extract_label_grid(
                    train_labels[sample_idx],
                    image_height=config.state_height,
                    image_width=config.state_width,
                    counts=counts,
                    token_offset=config.token_offset,
                    padded_height=config.final_image_shape[0],
                    padded_width=config.final_image_shape[1],
                    padded_channels=config.final_image_shape[2],
                )

                combined = np.concatenate(
                    [decoded_input.reshape(-1), decoded_label.reshape(-1)],
                    axis=0,
                )
                unique_colors = np.unique(combined)
                self.assertLessEqual(unique_colors.size, config.num_colors)
                self.assertTrue(np.all(unique_colors >= 0))
                self.assertTrue(np.all(unique_colors < config.out_colors))
                if np.any(unique_colors >= config.num_colors):
                    saw_extended_palette = True

            self.assertTrue(saw_extended_palette)

    def test_counts_are_capped_to_respect_max_data_seq_len(self) -> None:
        config = NCA2DDataConfig(
            train_size=1,
            test_size=1,
            state_height=24,
            state_height_min=24,
            state_height_max=24,
            state_width=24,
            state_width_min=24,
            state_width_max=24,
            counts=20,
            counts_max=20,
            max_data_seq_len=9000,
        )

        self.assertEqual(config.resolved_counts_max, 7)
        self.assertEqual(config.final_image_shape, (24, 24, 15))
        self.assertEqual(config.seq_len, 8640)

    def test_pair_windows_follow_answer_steps_and_span_on_same_trajectory(self) -> None:
        config = NCA2DDataConfig(
            train_size=1,
            test_size=1,
            state_height=4,
            state_height_min=4,
            state_height_max=4,
            state_width=3,
            state_width_min=3,
            state_width_max=3,
            num_colors=4,
            patch_size=1,
            answer_steps=2,
            answer_steps_min=2,
            answer_steps_max=2,
            counts=2,
            counts_min=2,
            counts_max=2,
            time_start=10,
            time_span=1,
            gzip_threshold_low=None,
            gzip_threshold_high=None,
        )
        device = get_device()
        window_cfg = make_nca_config_for_sample(
            config,
            state_height=config.state_height,
            state_width=config.state_width,
            answer_steps=config.answer_steps,
        )
        total_windows = config.counts + 1
        full_cfg = make_nca_config_for_sample(
            config,
            state_height=config.state_height,
            state_width=config.state_width,
            answer_steps=(total_windows - 1) * (config.answer_steps + 1 + config.time_span) + config.answer_steps,
        )

        seed_everything(123)
        pair_windows = rollout_pair_windows_batched(
            window_cfg,
            sample_batch_size=2,
            example_count=config.counts,
            answer_steps=config.answer_steps,
            time_span=config.time_span,
            device=device,
        ).cpu().numpy()

        seed_everything(123)
        full_trajectories = rollout_trajectories_batched(
            full_cfg,
            batch_size=2,
            device=device,
        )
        expected = np.stack(
            [
                np.stack([full_trajectories[:, 0], full_trajectories[:, 2]], axis=1),
                np.stack([full_trajectories[:, 4], full_trajectories[:, 6]], axis=1),
                np.stack([full_trajectories[:, 8], full_trajectories[:, 10]], axis=1),
            ],
            axis=1,
        )

        np.testing.assert_array_equal(pair_windows, expected)


if __name__ == "__main__":
    unittest.main()
