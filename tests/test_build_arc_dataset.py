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


def _encode_fixed_canvas(grid: list[list[int]]) -> np.ndarray:
    arr = np.array(grid, dtype=np.uint8)
    canvas = np.zeros((30, 30), dtype=np.uint8)
    h, w = arr.shape
    canvas[:h, :w] = arr + 2
    if h < 30:
        canvas[h, :w] = 1
    if w < 30:
        canvas[:h, w] = 1
    return canvas.reshape(-1)


def _encode_unpadded_canvas(grid: list[list[int]], canvas_shape: tuple[int, int]) -> np.ndarray:
    arr = np.array(grid, dtype=np.uint8)
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    h, w = arr.shape
    canvas[:h, :w] = arr + 2
    if h < canvas_shape[0]:
        canvas[h, :w] = 1
    if w < canvas_shape[1]:
        canvas[:h, w] = 1
    return canvas.reshape(-1)


def _pair_canvas_shape(inp: list[list[int]], out: list[list[int]]) -> tuple[int, int]:
    inp_arr = np.array(inp, dtype=np.uint8)
    out_arr = np.array(out, dtype=np.uint8)
    height = max(inp_arr.shape[0], out_arr.shape[0])
    width = max(inp_arr.shape[1], out_arr.shape[1])
    if height < 30:
        height += 1
    if width < 30:
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
            np.testing.assert_array_equal(puzzle_indices, np.array([0, 3, 4, 5], dtype=np.int32))
            np.testing.assert_array_equal(group_indices, np.array([0, 3], dtype=np.int32))
            self.assertEqual(metadata["total_groups"], 1)
            self.assertEqual(metadata["mean_puzzle_examples"], 5 / 3)

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

    def test_arc_gen_uses_independent_augmentation_count(self) -> None:
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

            self.assertEqual(inputs.shape, (5, 900))
            np.testing.assert_array_equal(puzzle_indices, np.array([0, 2, 3, 4, 5], dtype=np.int32))
            np.testing.assert_array_equal(group_indices, np.array([0, 4], dtype=np.int32))

    def test_test_split_writes_attached_examples_only_for_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_prefix = tmp_path / "arc"
            output_dir = tmp_path / "dataset"

            evaluation_puzzles = {
                "p_eval": {
                    "train": [
                        {"input": [[1]], "output": [[2]]},
                        {"input": [[3, 3]], "output": [[4, 4]]},
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
                {"p_eval": [[[6]]]},
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
            np.testing.assert_array_equal(examples[0][1, 1], _encode_fixed_canvas([[4, 4]]))

    def test_test_split_examples_support_no_padding_and_augmentation_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_prefix = tmp_path / "arc"
            output_dir = tmp_path / "dataset"

            evaluation_puzzles = {
                "p_eval": {
                    "train": [
                        {"input": [[1]], "output": [[2]]},
                        {"input": [[3, 3]], "output": [[4, 4]]},
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
                {"p_eval": [[[6]]]},
            )

            convert_dataset(
                DataProcessConfig(
                    input_file_prefix=str(input_prefix),
                    output_dir=str(output_dir),
                    subsets=["evaluation"],
                    test_set_name="evaluation",
                    seed=0,
                    num_aug=1,
                    no_padding=True,
                )
            )

            examples = np.load(output_dir / "test" / "all__examples.npy", allow_pickle=True)
            example_shapes = np.load(output_dir / "test" / "all__example_shapes.npy", allow_pickle=True)
            inputs = np.load(output_dir / "test" / "all__inputs.npy")
            seq_offsets = np.load(output_dir / "test" / "all__seq_offsets.npy")

            self.assertEqual(examples.shape, (2,))
            self.assertEqual(seq_offsets.shape, (3,))
            self.assertGreater(inputs.shape[0], 0)

            first_examples = examples[0]
            first_example_shapes = example_shapes[0]
            self.assertEqual(first_examples.shape, (2, 2))

            first_shape = _pair_canvas_shape([[1]], [[2]])
            second_shape = _pair_canvas_shape([[3, 3]], [[4, 4]])
            np.testing.assert_array_equal(
                first_example_shapes,
                np.array([first_shape, second_shape], dtype=np.int32),
            )
            np.testing.assert_array_equal(first_examples[0, 0], _encode_unpadded_canvas([[1]], first_shape))
            np.testing.assert_array_equal(first_examples[0, 1], _encode_unpadded_canvas([[2]], first_shape))
            np.testing.assert_array_equal(first_examples[1, 0], _encode_unpadded_canvas([[3, 3]], second_shape))
            np.testing.assert_array_equal(first_examples[1, 1], _encode_unpadded_canvas([[4, 4]], second_shape))


if __name__ == "__main__":
    unittest.main()
