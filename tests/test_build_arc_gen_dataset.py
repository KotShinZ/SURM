from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

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


from data import build_arc_gen_dataset as build_arc_gen_dataset  # noqa: E402


def _make_generator():
    state = {"value": 0}

    def _generate():
        state["value"] += 1
        value = state["value"]
        return {
            "input": [[value]],
            "output": [[value + 1]],
        }

    return _generate


class BuildARCGenDatasetTests(unittest.TestCase):
    def test_generated_examples_follow_arc_dataset_output_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "dataset"

            registry = {
                "task_a": (
                    _make_generator(),
                    lambda: {
                        "train": [
                            {"input": [[1]], "output": [[2]]},
                        ],
                        "test": [
                            {"input": [[3]], "output": [[4]]},
                        ],
                    },
                )
            }

            with mock.patch.object(
                build_arc_gen_dataset,
                "load_arc_gen_task_registry",
                return_value=registry,
            ):
                build_arc_gen_dataset.convert_dataset(
                    build_arc_gen_dataset.DataProcessConfig(
                        output_dir=str(output_dir),
                        seed=0,
                        num_aug=0,
                        no_padding=False,
                        examples_per_task=2,
                        task_ids=["task_a"],
                    )
                )

            train_inputs = np.load(output_dir / "train" / "all__inputs.npy")
            train_labels = np.load(output_dir / "train" / "all__labels.npy")
            train_puzzle_indices = np.load(output_dir / "train" / "all__puzzle_indices.npy")
            train_group_indices = np.load(output_dir / "train" / "all__group_indices.npy")

            test_inputs = np.load(output_dir / "test" / "all__inputs.npy")
            test_labels = np.load(output_dir / "test" / "all__labels.npy")
            test_puzzle_indices = np.load(output_dir / "test" / "all__puzzle_indices.npy")
            test_group_indices = np.load(output_dir / "test" / "all__group_indices.npy")

            with open(output_dir / "train" / "dataset.json", encoding="utf-8") as f:
                train_metadata = json.load(f)
            with open(output_dir / "test" / "dataset.json", encoding="utf-8") as f:
                test_metadata = json.load(f)
            with open(output_dir / "identifiers.json", encoding="utf-8") as f:
                identifiers = json.load(f)
            with open(output_dir / "test_puzzles.json", encoding="utf-8") as f:
                test_puzzles = json.load(f)

            self.assertEqual(train_inputs.shape, (2, 900))
            self.assertEqual(train_labels.shape, (2, 900))
            np.testing.assert_array_equal(train_puzzle_indices, np.array([0, 2], dtype=np.int32))
            np.testing.assert_array_equal(train_group_indices, np.array([0, 1], dtype=np.int32))

            self.assertEqual(test_inputs.shape, (1, 900))
            self.assertEqual(test_labels.shape, (1, 900))
            np.testing.assert_array_equal(test_puzzle_indices, np.array([0, 1], dtype=np.int32))
            np.testing.assert_array_equal(test_group_indices, np.array([0, 1], dtype=np.int32))

            self.assertEqual(train_metadata["total_groups"], 1)
            self.assertEqual(train_metadata["mean_puzzle_examples"], 2.0)
            self.assertEqual(test_metadata["total_groups"], 1)
            self.assertEqual(test_metadata["mean_puzzle_examples"], 1.0)

            self.assertEqual(identifiers, ["<blank>", "task_a"])
            self.assertEqual(
                test_puzzles,
                {
                    "task_a": {
                        "train": [{"input": [[1]], "output": [[2]]}],
                        "test": [{"input": [[3]], "output": [[4]]}],
                    }
                },
            )


if __name__ == "__main__":
    unittest.main()
