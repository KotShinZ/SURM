from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if "argdantic" not in sys.modules:
    argdantic = types.ModuleType("argdantic")

    class ArgParser:
        def command(self, singleton: bool = False):
            def _decorator(fn):
                return fn

            return _decorator

    argdantic.ArgParser = ArgParser
    sys.modules["argdantic"] = argdantic


from data.build_arc_dataset import DataProcessConfig, convert_dataset  # noqa: E402


def _encode_fixed_canvas(grid: list[list[int]], include_eos: bool = True) -> np.ndarray:
    arr = np.array(grid, dtype=np.uint8)
    canvas = np.zeros((30, 30), dtype=np.uint8)
    h, w = arr.shape
    color_offset = 2 if include_eos else 1
    canvas[:h, :w] = arr + color_offset
    if include_eos and h < 30:
        canvas[h, :w] = 1
    if include_eos and w < 30:
        canvas[:h, w] = 1
    return canvas.reshape(-1)


def _encode_unpadded_canvas(
    grid: list[list[int]],
    canvas_shape: tuple[int, int],
    include_eos: bool = True,
) -> np.ndarray:
    arr = np.array(grid, dtype=np.uint8)
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    h, w = arr.shape
    color_offset = 2 if include_eos else 1
    canvas[:h, :w] = arr + color_offset
    if include_eos and h < canvas_shape[0]:
        canvas[h, :w] = 1
    if include_eos and w < canvas_shape[1]:
        canvas[:h, w] = 1
    return canvas.reshape(-1)


def _grid_canvas_shape(grid: list[list[int]], include_eos: bool = True) -> tuple[int, int]:
    arr = np.array(grid, dtype=np.uint8)
    height, width = arr.shape
    if include_eos and height < 30:
        height += 1
    if include_eos and width < 30:
        width += 1
    return height, width


class BuildARCDatasetTests(unittest.TestCase):
    def _write_subset(
        self,
        prefix: Path,
        subset_name: str,
        challenge_puzzles: dict,
        solution_puzzles: dict,
    ) -> None:
        with open(prefix.parent / f"{prefix.name}_{subset_name}-challenges.json", "w", encoding="utf-8") as f:
            json.dump(challenge_puzzles, f)
        with open(prefix.parent / f"{prefix.name}_{subset_name}-solutions.json", "w", encoding="utf-8") as f:
            json.dump(solution_puzzles, f)

    def test_num_aug_all_sets_source_defaults_and_allows_overrides(self) -> None:
        config = DataProcessConfig(
            input_file_prefix="arc",
            output_dir="dataset",
            subsets=["training"],
            test_set_name="evaluation",
            num_aug_all=7,
            num_aug={"ARC-AGI2": 3},
        )

        self.assertEqual(config.num_aug.ARC_AGI1, 7)
        self.assertEqual(config.num_aug.ARC_AGI2, 3)
        self.assertEqual(config.num_aug.ARC_GEN1, 7)
        self.assertEqual(config.num_aug.ARC_GEN2, 7)

    def test_arc_gen_examples_are_merged_into_training_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_prefix = tmp_path / "arc"
            output_dir = tmp_path / "dataset"
            arc_gen_dir = tmp_path / "arc-gen"
            arc_gen_dir.mkdir()

            training_puzzles = {
                "p_train": {
                    "train": [
                        {"input": [[1]], "output": [[2]]},
                        {"input": [[3]], "output": [[4]]},
                    ],
                    "test": [
                        {"input": [[5]]},
                    ],
                }
            }

            self._write_subset(
                input_prefix,
                "training",
                training_puzzles,
                {"p_train": [[[6]]]},
            )

            with open(arc_gen_dir / "p_train.json", "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {"input": [[7]], "output": [[8]]},
                        {"input": [[9]], "output": [[1]]},
                    ],
                    f,
                )
            with open(arc_gen_dir / "unused.json", "w", encoding="utf-8") as f:
                json.dump([{"input": [[0]], "output": [[0]]}], f)

            convert_dataset(
                DataProcessConfig(
                    input_file_prefix=str(input_prefix),
                    output_dir=str(output_dir),
                    subsets=["training"],
                    test_set_name="evaluation",
                    seed=0,
                    num_aug=0,
                    no_padding=False,
                    include_arc_gen=True,
                    arc_gen_dir=str(arc_gen_dir),
                )
            )

            inputs = np.load(output_dir / "train" / "all__inputs.npy")
            labels = np.load(output_dir / "train" / "all__labels.npy")
            puzzle_indices = np.load(output_dir / "train" / "all__puzzle_indices.npy")
            group_indices = np.load(output_dir / "train" / "all__group_indices.npy")

            with open(output_dir / "train" / "dataset.json", encoding="utf-8") as f:
                metadata = json.load(f)

            self.assertEqual(inputs.shape, (5, 900))
            self.assertEqual(labels.shape, (5, 900))
            np.testing.assert_array_equal(puzzle_indices, np.array([0, 5], dtype=np.int32))
            np.testing.assert_array_equal(group_indices, np.array([0, 1], dtype=np.int32))
            self.assertEqual(metadata["total_groups"], 1)
            self.assertEqual(metadata["mean_puzzle_examples"], 5.0)

    def test_arc_gen_examples_can_be_disabled_even_when_directory_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_prefix = tmp_path / "arc"
            output_dir = tmp_path / "dataset"
            arc_gen_dir = tmp_path / "arc-gen"
            arc_gen_dir.mkdir()

            training_puzzles = {
                "p_train": {
                    "train": [
                        {"input": [[1]], "output": [[2]]},
                        {"input": [[3]], "output": [[4]]},
                    ],
                    "test": [
                        {"input": [[5]]},
                    ],
                }
            }

            self._write_subset(
                input_prefix,
                "training",
                training_puzzles,
                {"p_train": [[[6]]]},
            )

            with open(arc_gen_dir / "p_train.json", "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {"input": [[7]], "output": [[8]]},
                        {"input": [[9]], "output": [[1]]},
                    ],
                    f,
                )

            convert_dataset(
                DataProcessConfig(
                    input_file_prefix=str(input_prefix),
                    output_dir=str(output_dir),
                    subsets=["training"],
                    test_set_name="evaluation",
                    seed=0,
                    num_aug=0,
                    no_padding=False,
                    include_arc_gen=False,
                    arc_gen_dir=str(arc_gen_dir),
                )
            )

            inputs = np.load(output_dir / "train" / "all__inputs.npy")
            labels = np.load(output_dir / "train" / "all__labels.npy")
            puzzle_indices = np.load(output_dir / "train" / "all__puzzle_indices.npy")
            group_indices = np.load(output_dir / "train" / "all__group_indices.npy")

            with open(output_dir / "train" / "dataset.json", encoding="utf-8") as f:
                metadata = json.load(f)

            self.assertEqual(inputs.shape, (3, 900))
            self.assertEqual(labels.shape, (3, 900))
            np.testing.assert_array_equal(puzzle_indices, np.array([0, 3], dtype=np.int32))
            np.testing.assert_array_equal(group_indices, np.array([0, 1], dtype=np.int32))
            self.assertEqual(metadata["total_groups"], 1)
            self.assertEqual(metadata["mean_puzzle_examples"], 3.0)

    def test_arc_gen_uses_source_augmentation_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_prefix = tmp_path / "arc"
            output_dir = tmp_path / "dataset"
            arc_gen_dir = tmp_path / "arc-gen"
            arc_gen_dir.mkdir()

            training_puzzles = {
                "p_train": {
                    "train": [
                        {"input": [[1]], "output": [[2]]},
                    ],
                    "test": [
                        {"input": [[5]]},
                    ],
                }
            }

            self._write_subset(
                input_prefix,
                "training",
                training_puzzles,
                {"p_train": [[[6]]]},
            )

            with open(arc_gen_dir / "p_train.json", "w", encoding="utf-8") as f:
                json.dump([{"input": [[7]], "output": [[8]]}], f)

            convert_dataset(
                DataProcessConfig(
                    input_file_prefix=str(input_prefix),
                    output_dir=str(output_dir),
                    subsets=["training"],
                    test_set_name="evaluation",
                    seed=0,
                    num_aug=0,
                    num_aug_gen=2,
                    no_padding=False,
                    include_arc_gen=True,
                    arc_gen_dir=str(arc_gen_dir),
                )
            )

            inputs = np.load(output_dir / "train" / "all__inputs.npy")
            puzzle_indices = np.load(output_dir / "train" / "all__puzzle_indices.npy")
            group_indices = np.load(output_dir / "train" / "all__group_indices.npy")

            self.assertEqual(inputs.shape, (3, 900))
            np.testing.assert_array_equal(puzzle_indices, np.array([0, 3], dtype=np.int32))
            np.testing.assert_array_equal(group_indices, np.array([0, 1], dtype=np.int32))

    def test_explicit_sources_share_groups_by_task_id_and_use_source_aug_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_prefix = tmp_path / "arc"
            output_dir = tmp_path / "dataset"
            arc_gen1_dir = tmp_path / "arc-gen1"
            arc_gen2_dir = tmp_path / "arc-gen2"
            arc_gen1_dir.mkdir()
            arc_gen2_dir.mkdir()

            self._write_subset(
                input_prefix,
                "training",
                {
                    "p_shared": {
                        "train": [{"input": [[1]], "output": [[2]]}],
                        "test": [{"input": [[3]]}],
                    }
                },
                {"p_shared": [[[4]]]},
            )
            self._write_subset(
                input_prefix,
                "training2",
                {
                    "p_shared": {
                        "train": [{"input": [[5]], "output": [[6]]}],
                        "test": [{"input": [[7]]}],
                    }
                },
                {"p_shared": [[[8]]]},
            )

            with open(arc_gen1_dir / "p_shared.json", "w", encoding="utf-8") as f:
                json.dump([{"input": [[1]], "output": [[3]]}], f)
            with open(arc_gen2_dir / "p_shared.json", "w", encoding="utf-8") as f:
                json.dump([{"input": [[2]], "output": [[4]]}], f)

            convert_dataset(
                DataProcessConfig(
                    input_file_prefix=str(input_prefix),
                    output_dir=str(output_dir),
                    subsets=["training"],
                    test_set_name="evaluation",
                    seed=0,
                    sources=["ARC-AGI1", "ARC-AGI2", "ARC-GEN1", "ARC-GEN2"],
                    num_aug={
                        "ARC-AGI1": 0,
                        "ARC-AGI2": 0,
                        "ARC-GEN1": 2,
                        "ARC-GEN2": 0,
                    },
                    no_padding=False,
                    arc_gen_dir=str(arc_gen1_dir),
                    arc_gen2_dir=str(arc_gen2_dir),
                )
            )

            inputs = np.load(output_dir / "train" / "all__inputs.npy")
            puzzle_indices = np.load(output_dir / "train" / "all__puzzle_indices.npy")
            group_indices = np.load(output_dir / "train" / "all__group_indices.npy")

            self.assertEqual(inputs.shape, (7, 900))
            np.testing.assert_array_equal(
                puzzle_indices,
                np.array([0, 2, 4, 5, 6, 7], dtype=np.int32),
            )
            np.testing.assert_array_equal(group_indices, np.array([0, 5], dtype=np.int32))

    def test_test_split_writes_attached_examples_only_for_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_prefix = tmp_path / "arc"
            output_dir = tmp_path / "dataset"

            evaluation_puzzles = {
                "p_eval": {
                    "train": [
                        {"input": [[1]], "output": [[2]]},
                        {"input": [[3, 3]], "output": [[4], [4]]},
                    ],
                    "test": [
                        {"input": [[5]]},
                    ],
                }
            }

            self._write_subset(
                input_prefix,
                "evaluation",
                evaluation_puzzles,
                {"p_eval": [[[6], [6]]]},
            )

            convert_dataset(
                DataProcessConfig(
                    input_file_prefix=str(input_prefix),
                    output_dir=str(output_dir),
                    subsets=["evaluation"],
                    test_set_name="evaluation",
                    seed=0,
                    num_aug=0,
                    no_padding=False,
                )
            )

            self.assertFalse((output_dir / "train" / "all__examples.npy").exists())
            self.assertFalse((output_dir / "train" / "all__example_shapes.npy").exists())
            examples = np.load(output_dir / "test" / "all__examples.npy", allow_pickle=True)
            example_shapes = np.load(output_dir / "test" / "all__example_shapes.npy", allow_pickle=True)

            self.assertEqual(examples.shape, (1,))
            self.assertEqual(example_shapes.shape, (1,))
            self.assertEqual(examples[0].shape, (2, 2, 900))
            np.testing.assert_array_equal(
                example_shapes[0],
                np.array([[30, 30], [30, 30]], dtype=np.int32),
            )
            np.testing.assert_array_equal(examples[0][0, 0], _encode_fixed_canvas([[1]]))
            np.testing.assert_array_equal(examples[0][0, 1], _encode_fixed_canvas([[2]]))
            np.testing.assert_array_equal(examples[0][1, 0], _encode_fixed_canvas([[3, 3]]))
            np.testing.assert_array_equal(examples[0][1, 1], _encode_fixed_canvas([[4], [4]]))

    def test_test_set_is_loaded_even_when_omitted_from_training_subsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_prefix = tmp_path / "arc"
            output_dir = tmp_path / "dataset"

            self._write_subset(
                input_prefix,
                "training",
                {
                    "p_train": {
                        "train": [{"input": [[1]], "output": [[2]]}],
                        "test": [{"input": [[3]]}],
                    }
                },
                {"p_train": [[[4]]]},
            )
            self._write_subset(
                input_prefix,
                "concept",
                {
                    "p_concept": {
                        "train": [{"input": [[5]], "output": [[6]]}],
                        "test": [{"input": [[7]]}],
                    }
                },
                {"p_concept": [[[8]]]},
            )
            self._write_subset(
                input_prefix,
                "evaluation",
                {
                    "p_eval": {
                        "train": [{"input": [[1, 1]], "output": [[2, 2]]}],
                        "test": [{"input": [[9]]}],
                    }
                },
                {"p_eval": [[[0]]]},
            )

            convert_dataset(
                DataProcessConfig(
                    input_file_prefix=str(input_prefix),
                    output_dir=str(output_dir),
                    subsets=["training", "concept"],
                    test_set_name="evaluation",
                    seed=0,
                    num_aug=0,
                    no_padding=False,
                )
            )

            train_inputs = np.load(output_dir / "train" / "all__inputs.npy")
            test_inputs = np.load(output_dir / "test" / "all__inputs.npy")
            examples = np.load(output_dir / "test" / "all__examples.npy", allow_pickle=True)

            self.assertEqual(train_inputs.shape, (4, 900))
            self.assertEqual(test_inputs.shape, (1, 900))
            self.assertEqual(examples.shape, (1,))
            np.testing.assert_array_equal(test_inputs[0], _encode_fixed_canvas([[9]]))
            np.testing.assert_array_equal(examples[0][0, 0], _encode_fixed_canvas([[1, 1]]))
            np.testing.assert_array_equal(examples[0][0, 1], _encode_fixed_canvas([[2, 2]]))

    def test_test_split_examples_support_no_padding_and_augmentation_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_prefix = tmp_path / "arc"
            output_dir = tmp_path / "dataset"

            evaluation_puzzles = {
                "p_eval": {
                    "train": [
                        {"input": [[1]], "output": [[2]]},
                        {"input": [[3, 3]], "output": [[4], [4]]},
                    ],
                    "test": [
                        {"input": [[5]]},
                    ],
                }
            }

            self._write_subset(
                input_prefix,
                "evaluation",
                evaluation_puzzles,
                {"p_eval": [[[6], [6]]]},
            )

            convert_dataset(
                DataProcessConfig(
                    input_file_prefix=str(input_prefix),
                    output_dir=str(output_dir),
                    subsets=["evaluation"],
                    test_set_name="evaluation",
                    seed=0,
                    num_aug=2,
                    no_padding=True,
                )
            )

            examples = np.load(output_dir / "test" / "all__examples.npy", allow_pickle=True)
            example_shapes = np.load(output_dir / "test" / "all__example_shapes.npy", allow_pickle=True)
            inputs = np.load(output_dir / "test" / "all__inputs.npy")
            labels = np.load(output_dir / "test" / "all__labels.npy")
            seq_offsets = np.load(output_dir / "test" / "all__seq_offsets.npy")
            label_seq_offsets = np.load(output_dir / "test" / "all__label_seq_offsets.npy")
            seq_shapes = np.load(output_dir / "test" / "all__seq_shapes.npy")
            label_seq_shapes = np.load(output_dir / "test" / "all__label_seq_shapes.npy")

            self.assertEqual(examples.shape, (2,))
            self.assertEqual(seq_offsets.shape, (3,))
            self.assertEqual(label_seq_offsets.shape, (3,))
            self.assertGreater(inputs.shape[0], 0)
            self.assertGreater(labels.shape[0], 0)
            self.assertFalse(np.array_equal(seq_offsets, label_seq_offsets))
            self.assertFalse(np.array_equal(seq_shapes, label_seq_shapes))

            first_examples = examples[0]
            first_example_shapes = example_shapes[0]
            self.assertEqual(first_examples.shape, (2, 2))

            first_input_shape = _grid_canvas_shape([[1]])
            first_label_shape = _grid_canvas_shape([[2]])
            second_input_shape = _grid_canvas_shape([[3, 3]])
            second_label_shape = _grid_canvas_shape([[4], [4]])
            np.testing.assert_array_equal(
                first_example_shapes,
                np.array(
                    [
                        [first_input_shape, first_label_shape],
                        [second_input_shape, second_label_shape],
                    ],
                    dtype=np.int32,
                ),
            )
            np.testing.assert_array_equal(first_examples[0, 0], _encode_unpadded_canvas([[1]], first_input_shape))
            np.testing.assert_array_equal(first_examples[0, 1], _encode_unpadded_canvas([[2]], first_label_shape))
            np.testing.assert_array_equal(first_examples[1, 0], _encode_unpadded_canvas([[3, 3]], second_input_shape))
            np.testing.assert_array_equal(first_examples[1, 1], _encode_unpadded_canvas([[4], [4]], second_label_shape))

    def test_no_eos_fixed_dataset_uses_compact_color_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_prefix = tmp_path / "arc"
            output_dir = tmp_path / "dataset"

            training_puzzles = {
                "p_train": {
                    "train": [
                        {"input": [[1, 0], [2, 9]], "output": [[4]]},
                    ],
                    "test": [
                        {"input": [[5]]},
                    ],
                }
            }

            self._write_subset(
                input_prefix,
                "training",
                training_puzzles,
                {"p_train": [[[6]]]},
            )

            convert_dataset(
                DataProcessConfig(
                    input_file_prefix=str(input_prefix),
                    output_dir=str(output_dir),
                    subsets=["training"],
                    test_set_name="evaluation",
                    seed=0,
                    num_aug=0,
                    no_padding=False,
                    no_eos=True,
                )
            )

            inputs = np.load(output_dir / "train" / "all__inputs.npy")
            labels = np.load(output_dir / "train" / "all__labels.npy")
            with open(output_dir / "train" / "dataset.json", encoding="utf-8") as f:
                metadata = json.load(f)

            self.assertEqual(inputs.shape, (2, 900))
            self.assertEqual(metadata["vocab_size"], 11)
            self.assertTrue(np.any(inputs == 1))
            self.assertTrue(np.any(inputs == 10))
            self.assertFalse(np.any(inputs == 11))
            self.assertFalse(np.any(labels == 11))
            np.testing.assert_array_equal(
                inputs[0],
                _encode_fixed_canvas([[1, 0], [2, 9]], include_eos=False),
            )
            np.testing.assert_array_equal(
                labels[0],
                _encode_fixed_canvas([[4]], include_eos=False),
            )

    def test_no_eos_no_padding_dataset_and_context_examples_use_compact_color_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_prefix = tmp_path / "arc"
            output_dir = tmp_path / "dataset"

            evaluation_puzzles = {
                "p_eval": {
                    "train": [
                        {"input": [[0]], "output": [[2]]},
                        {"input": [[3, 3]], "output": [[4], [4]]},
                    ],
                    "test": [
                        {"input": [[5]]},
                    ],
                }
            }

            self._write_subset(
                input_prefix,
                "evaluation",
                evaluation_puzzles,
                {"p_eval": [[[9], [6]]]},
            )

            convert_dataset(
                DataProcessConfig(
                    input_file_prefix=str(input_prefix),
                    output_dir=str(output_dir),
                    subsets=["evaluation"],
                    test_set_name="evaluation",
                    seed=0,
                    num_aug=0,
                    no_padding=True,
                    no_eos=True,
                )
            )

            examples = np.load(output_dir / "test" / "all__examples.npy", allow_pickle=True)
            example_shapes = np.load(output_dir / "test" / "all__example_shapes.npy", allow_pickle=True)
            inputs = np.load(output_dir / "test" / "all__inputs.npy")
            labels = np.load(output_dir / "test" / "all__labels.npy")
            seq_shapes = np.load(output_dir / "test" / "all__seq_shapes.npy")
            label_seq_shapes = np.load(output_dir / "test" / "all__label_seq_shapes.npy")
            with open(output_dir / "test" / "dataset.json", encoding="utf-8") as f:
                metadata = json.load(f)

            self.assertEqual(metadata["vocab_size"], 11)
            self.assertTrue(np.any(labels == 10))
            self.assertFalse(np.any(inputs == 11))
            self.assertFalse(np.any(labels == 11))
            np.testing.assert_array_equal(seq_shapes, np.array([[1, 1]], dtype=np.int32))
            np.testing.assert_array_equal(label_seq_shapes, np.array([[2, 1]], dtype=np.int32))
            np.testing.assert_array_equal(
                inputs,
                _encode_unpadded_canvas([[5]], (1, 1), include_eos=False),
            )
            np.testing.assert_array_equal(
                labels,
                _encode_unpadded_canvas([[9], [6]], (2, 1), include_eos=False),
            )

            first_examples = examples[0]
            first_example_shapes = example_shapes[0]
            np.testing.assert_array_equal(
                first_example_shapes,
                np.array(
                    [
                        [
                            _grid_canvas_shape([[0]], include_eos=False),
                            _grid_canvas_shape([[2]], include_eos=False),
                        ],
                        [
                            _grid_canvas_shape([[3, 3]], include_eos=False),
                            _grid_canvas_shape([[4], [4]], include_eos=False),
                        ],
                    ],
                    dtype=np.int32,
                ),
            )
            self.assertFalse(
                any(np.any(np.asarray(part) == 11) for part in first_examples.reshape(-1))
            )
            self.assertTrue(np.any(np.asarray(first_examples[0, 0]) == 1))
            np.testing.assert_array_equal(
                first_examples[0, 0],
                _encode_unpadded_canvas([[0]], (1, 1), include_eos=False),
            )
            np.testing.assert_array_equal(
                first_examples[0, 1],
                _encode_unpadded_canvas([[2]], (1, 1), include_eos=False),
            )


if __name__ == "__main__":
    unittest.main()
