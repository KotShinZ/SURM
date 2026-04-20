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


from data import build_arc_dataset_full_2 as arc_full2  # noqa: E402


class BuildARCDatasetFull2Tests(unittest.TestCase):
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

    def test_make_pair_position_ids_emits_pair_io_row_col_axes(self) -> None:
        input_position_ids, output_position_ids = arc_full2._make_pair_position_ids((30, 30), example_index=12)

        self.assertEqual(input_position_ids.shape, (900, 4))
        self.assertEqual(output_position_ids.shape, (900, 4))
        np.testing.assert_array_equal(input_position_ids[0], np.array([12, 0, 0, 0], dtype=np.uint8))
        np.testing.assert_array_equal(input_position_ids[-1], np.array([12, 0, 29, 29], dtype=np.uint8))
        np.testing.assert_array_equal(output_position_ids[0], np.array([12, 1, 0, 0], dtype=np.uint8))
        np.testing.assert_array_equal(output_position_ids[-1], np.array([12, 1, 29, 29], dtype=np.uint8))

    def test_convert_dataset_writes_4d_position_shape_for_no_padding_arc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_prefix = tmp_path / "arc"
            output_dir = tmp_path / "dataset"

            puzzle = {
                "p_test": {
                    "train": [
                        {"input": [[1]], "output": [[2]]},
                        {"input": [[3, 3]], "output": [[4, 4]]},
                        {"input": [[5], [5]], "output": [[6], [6]]},
                    ],
                    "test": [
                        {"input": [[7, 7], [7, 7]]},
                    ],
                }
            }

            self._write_subset(
                input_prefix,
                "mini",
                puzzle,
                {"p_test": [[[8, 8], [8, 8]]]},
            )

            original_print_data = arc_full2.print_data
            arc_full2.print_data = lambda *args, **kwargs: None
            try:
                arc_full2.convert_dataset(
                    arc_full2.DataProcessConfig(
                        input_file_prefix=str(input_prefix),
                        output_dir=str(output_dir),
                        subsets=["mini"],
                        test_set_name="mini",
                        seed=0,
                        num_aug=0,
                        no_padding=True,
                        min_context_pairs=2,
                    )
                )
            finally:
                arc_full2.print_data = original_print_data

            with open(output_dir / "test" / "dataset.json", encoding="utf-8") as f:
                test_metadata = json.load(f)
            test_position_ids = np.load(output_dir / "test" / "all__position_ids.npy")

            self.assertEqual(test_metadata["position_id_shape"], [4, 2, 3, 3])
            self.assertEqual(test_position_ids.shape, (50, 4))
            np.testing.assert_array_equal(test_position_ids[0], np.array([0, 0, 0, 0], dtype=np.uint8))
            np.testing.assert_array_equal(test_position_ids[-1], np.array([3, 1, 2, 2], dtype=np.uint8))

            try:
                from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
            except ModuleNotFoundError as exc:
                self.skipTest(f"PuzzleDataset dependencies unavailable: {exc}")

            dataset = PuzzleDataset(
                PuzzleDatasetConfig(
                    seed=3,
                    dataset_path=str(output_dir),
                    global_batch_size=1,
                    test_set_mode=True,
                    epochs_per_iter=1,
                    rank=0,
                    num_replicas=1,
                ),
                split="test",
            )
            set_name, batch, effective_batch_size = next(iter(dataset))

            self.assertEqual(set_name, "all")
            self.assertEqual(effective_batch_size, 1)
            self.assertEqual(tuple(batch["position_ids"].shape), (50, 4))
            np.testing.assert_array_equal(batch["position_ids"][0].numpy(), np.array([0, 0, 0, 0], dtype=np.int32))
            np.testing.assert_array_equal(batch["position_ids"][-1].numpy(), np.array([3, 1, 2, 2], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
