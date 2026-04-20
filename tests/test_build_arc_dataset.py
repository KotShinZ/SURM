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


if __name__ == "__main__":
    unittest.main()
