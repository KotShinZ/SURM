from __future__ import annotations

import json
import sys
import tempfile
import time
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


def _make_flaky_generator():
    state = {"calls": 0}

    def _generate():
        state["calls"] += 1
        if state["calls"] == 1:
            raise ValueError("empty range in randrange(10, 10)")

        value = state["calls"]
        return {
            "input": [[value]],
            "output": [[value + 1]],
        }

    return _generate


class BuildARCGenDatasetRetryTests(unittest.TestCase):
    def test_generator_exceptions_are_retried_without_aborting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "dataset"

            registry = {
                "task_flaky": (
                    _make_flaky_generator(),
                    lambda: {
                        "train": [{"input": [[1]], "output": [[2]]}],
                        "test": [{"input": [[3]], "output": [[4]]}],
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
                        task_ids=["task_flaky"],
                    )
                )

            train_inputs = np.load(output_dir / "train" / "all__inputs.npy")
            train_labels = np.load(output_dir / "train" / "all__labels.npy")
            with open(output_dir / "train" / "dataset.json", encoding="utf-8") as f:
                train_metadata = json.load(f)

            self.assertEqual(train_inputs.shape, (2, 900))
            self.assertEqual(train_labels.shape, (2, 900))
            self.assertEqual(train_metadata["total_groups"], 1)
            self.assertEqual(train_metadata["mean_puzzle_examples"], 2.0)

    def test_generator_timeouts_are_retried_without_aborting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "dataset"

            state = {"calls": 0}

            def _slow_then_ok():
                state["calls"] += 1
                if state["calls"] == 1:
                    time.sleep(10)
                value = state["calls"]
                return {
                    "input": [[value]],
                    "output": [[value + 1]],
                }

            registry = {
                "task_slow": (
                    _slow_then_ok,
                    lambda: {
                        "train": [{"input": [[1]], "output": [[2]]}],
                        "test": [{"input": [[3]], "output": [[4]]}],
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
                        generator_timeout_sec=0.05,
                        task_ids=["task_slow"],
                    )
                )

            train_inputs = np.load(output_dir / "train" / "all__inputs.npy")
            train_labels = np.load(output_dir / "train" / "all__labels.npy")
            with open(output_dir / "train" / "dataset.json", encoding="utf-8") as f:
                train_metadata = json.load(f)

            self.assertEqual(train_inputs.shape, (2, 900))
            self.assertEqual(train_labels.shape, (2, 900))
            self.assertEqual(train_metadata["total_groups"], 1)
            self.assertEqual(train_metadata["mean_puzzle_examples"], 2.0)


if __name__ == "__main__":
    unittest.main()
