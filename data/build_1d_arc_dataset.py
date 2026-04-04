from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import os
import json
import hashlib

import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel, model_validator

from data.common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    dataset_dir: str = "1D-ARC/dataset"
    output_dir: str = "data/1d-arc-aug"

    seed: int = 42
    num_aug: int = 1000

    test_ratio: float = 0.04
    min_test_per_task: int = 2

    @model_validator(mode="after")
    def _validate(self) -> "DataProcessConfig":
        if not (0.0 < self.test_ratio < 1.0):
            raise ValueError(f"test_ratio must be in (0, 1), got {self.test_ratio}")
        if self.min_test_per_task <= 0:
            raise ValueError(
                f"min_test_per_task must be > 0, got {self.min_test_per_task}"
            )
        if self.num_aug < 0:
            raise ValueError(f"num_aug must be >= 0, got {self.num_aug}")
        return self


ARC1DAugmentRetriesFactor = 10
PuzzleIdSeparator = "|||"
PadTokenId = 0
EosTokenId = 1
ColorTokenOffset = 2
ARCNumColors = 10


@dataclass
class ARC1DPuzzle:
    id: str
    task_name: str
    examples: List[Tuple[np.ndarray, np.ndarray]]


def arc1d_grid_to_np(grid: List[List[int]]) -> np.ndarray:
    arr = np.array(grid)

    assert arr.ndim == 2, f"Expected a 2D grid, got ndim={arr.ndim}"
    assert arr.shape[0] == 1, f"Expected a 1xN grid, got shape={arr.shape}"
    assert np.all((arr >= 0) & (arr <= 9)), "1D-ARC colors must be in [0, 9]"

    return arr.astype(np.uint8)


def np_grid_to_seq(grid: np.ndarray, seq_len: int) -> np.ndarray:
    assert grid.ndim == 2 and grid.shape[0] == 1
    width = grid.shape[1]
    assert width + 1 <= seq_len, f"Grid width {width} does not fit seq_len={seq_len}"

    seq = np.zeros((seq_len,), dtype=np.uint8)
    seq[:width] = grid[0] + ColorTokenOffset
    seq[width] = EosTokenId
    return seq


def grid_hash(grid: np.ndarray) -> str:
    assert grid.ndim == 2
    assert grid.dtype == np.uint8

    buffer = [dim.to_bytes(2, "big") for dim in grid.shape]
    buffer.append(grid.tobytes())
    return hashlib.sha256(b"".join(buffer)).hexdigest()


def puzzle_hash(puzzle: ARC1DPuzzle) -> str:
    hashes = []
    for inp, out in puzzle.examples:
        hashes.append(f"{grid_hash(inp)}|{grid_hash(out)}")

    hashes.sort()
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def aug(name: str):
    mapping = np.arange(ARCNumColors, dtype=np.uint8)
    mapping[1:] = np.random.permutation(mapping[1:])

    name_with_aug_repr = (
        f"{name}{PuzzleIdSeparator}c{''.join(str(x) for x in mapping.tolist())}"
    )

    def _map_grid(grid: np.ndarray) -> np.ndarray:
        return mapping[grid]

    return name_with_aug_repr, _map_grid


def _choose_num_test_puzzles(
    total_puzzles: int,
    test_ratio: float,
    min_test_per_task: int,
) -> int:
    if total_puzzles < 2:
        raise ValueError(
            f"Each 1D-ARC task directory must contain at least 2 puzzles, got {total_puzzles}"
        )

    num_test = max(min_test_per_task, int(round(total_puzzles * test_ratio)))
    num_test = min(num_test, total_puzzles - 1)
    return num_test


def _puzzle_max_width(puzzle: dict) -> int:
    max_width = 0
    for split_name in ("train", "test"):
        for example in puzzle.get(split_name, []):
            inp = arc1d_grid_to_np(example["input"])
            out = arc1d_grid_to_np(example["output"])
            max_width = max(max_width, inp.shape[1], out.shape[1])
    return max_width


def convert_single_1d_arc_puzzle(
    results: dict,
    name: str,
    task_name: str,
    puzzle: dict,
    aug_count: int,
    split_name: str,
) -> None:
    converted = ARC1DPuzzle(
        id=name,
        task_name=task_name,
        examples=[],
    )
    for example_type in ("train", "test"):
        converted.examples.extend(
            [
                (arc1d_grid_to_np(example["input"]), arc1d_grid_to_np(example["output"]))
                for example in puzzle.get(example_type, [])
            ]
        )

    group = [converted]

    if aug_count > 0:
        hashes = {puzzle_hash(converted)}

        for _trial in range(ARC1DAugmentRetriesFactor * aug_count):
            aug_name, map_grid = aug(name)

            augmented = ARC1DPuzzle(
                id=aug_name,
                task_name=task_name,
                examples=[(map_grid(inp), map_grid(out)) for inp, out in converted.examples],
            )
            h = puzzle_hash(augmented)
            if h not in hashes:
                hashes.add(h)
                group.append(augmented)

            if len(group) >= aug_count + 1:
                break

    results.setdefault(split_name, {})
    results[split_name].setdefault("all", [])
    results[split_name]["all"].append(group)


def load_puzzles_1d_arc(config: DataProcessConfig):
    dataset_dir = Path(config.dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    task_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    if not task_dirs:
        raise FileNotFoundError(f"No task directories found under {dataset_dir}")

    results: Dict[str, Dict[str, list]] = {}
    test_puzzles = {}
    task_names: List[str] = []
    max_grid_width = 0

    total_train_puzzles = 0
    total_test_puzzles = 0

    for task_idx, task_dir in enumerate(task_dirs):
        task_name = task_dir.name
        task_names.append(task_name)

        puzzle_files = sorted(task_dir.glob("*.json"))
        if not puzzle_files:
            continue

        num_test = _choose_num_test_puzzles(
            len(puzzle_files),
            test_ratio=config.test_ratio,
            min_test_per_task=config.min_test_per_task,
        )

        rng = np.random.default_rng(config.seed + task_idx)
        order = rng.permutation(len(puzzle_files))
        test_indices = set(order[:num_test].tolist())

        task_train = 0
        task_test = 0
        for file_idx, puzzle_file in enumerate(puzzle_files):
            with open(puzzle_file, "r") as f:
                puzzle = json.load(f)

            max_grid_width = max(max_grid_width, _puzzle_max_width(puzzle))

            split_name = "test" if file_idx in test_indices else "train"
            aug_count = config.num_aug if split_name == "train" else 0

            convert_single_1d_arc_puzzle(
                results=results,
                name=puzzle_file.stem,
                task_name=task_name,
                puzzle=puzzle,
                aug_count=aug_count,
                split_name=split_name,
            )

            if split_name == "test":
                task_test += 1
                test_puzzles[puzzle_file.stem] = {"task_name": task_name, **puzzle}
            else:
                task_train += 1

        total_train_puzzles += task_train
        total_test_puzzles += task_test
        print(
            f"[{task_name}] train puzzles: {task_train}, test puzzles: {task_test}, "
            f"num_aug(train only): {config.num_aug}"
        )

    print(f"Total task types: {len(task_names)}")
    print(f"Total train puzzles: {total_train_puzzles}")
    print(f"Total test puzzles: {total_test_puzzles}")
    print(f"Max grid width: {max_grid_width}")

    return results, test_puzzles, task_names, max_grid_width


def convert_dataset(config: DataProcessConfig) -> None:
    np.random.seed(config.seed)

    data, test_puzzles, task_names, max_grid_width = load_puzzles_1d_arc(config)

    seq_len = max_grid_width + 1
    task_identifier_map = {task_name: idx + 1 for idx, task_name in enumerate(task_names)}

    os.makedirs(config.output_dir, exist_ok=True)

    print(f"Sequence length: {seq_len}")
    print(f"Total task embeddings (including <blank>): {len(task_identifier_map) + 1}")

    for split_name, split in data.items():
        split_dir = os.path.join(config.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        total_examples = 0
        total_puzzles = 0
        total_groups = 0

        for subset_name, subset in split.items():
            results = {
                "inputs": [],
                "labels": [],
                "puzzle_identifiers": [],
                "puzzle_indices": [0],
                "group_indices": [0],
            }

            example_id = 0
            puzzle_id = 0

            for group in subset:
                for puzzle in group:
                    for inp, out in puzzle.examples:
                        results["inputs"].append(np_grid_to_seq(inp, seq_len))
                        results["labels"].append(np_grid_to_seq(out, seq_len))
                        example_id += 1
                        total_examples += 1

                    results["puzzle_indices"].append(example_id)
                    results["puzzle_identifiers"].append(
                        task_identifier_map[puzzle.task_name]
                    )
                    puzzle_id += 1
                    total_puzzles += 1

                results["group_indices"].append(puzzle_id)
                total_groups += 1

            serialized = {}
            for key, value in results.items():
                if key in {"inputs", "labels"}:
                    serialized[key] = np.stack(value, axis=0)
                else:
                    serialized[key] = np.array(value, dtype=np.int32)

            for key, value in serialized.items():
                np.save(os.path.join(split_dir, f"{subset_name}__{key}.npy"), value)

        metadata = PuzzleDatasetMetadata(
            seq_len=seq_len,
            vocab_size=ARCNumColors + ColorTokenOffset,
            pad_id=PadTokenId,
            ignore_label_id=PadTokenId,
            blank_identifier_id=0,
            num_puzzle_identifiers=len(task_identifier_map) + 1,
            total_groups=total_groups,
            mean_puzzle_examples=total_examples / total_puzzles,
            sets=list(split.keys()),
        )

        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)

        print(
            f"[{split_name}] puzzles: {total_puzzles}, examples: {total_examples}, "
            f"groups: {total_groups}, mean examples/puzzle: {metadata.mean_puzzle_examples:.2f}"
        )

    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>", *task_names], f)

    with open(os.path.join(config.output_dir, "test_puzzles.json"), "w") as f:
        json.dump(test_puzzles, f)

    with open(os.path.join(config.output_dir, "build_config.json"), "w") as f:
        json.dump(
            {
                **config.model_dump(),
                "task_names": task_names,
                "seq_len": seq_len,
                "num_task_embeddings": len(task_names),
                "max_grid_width": max_grid_width,
            },
            f,
            indent=2,
        )


@cli.command(singleton=True)
def main(config: DataProcessConfig) -> None:
    convert_dataset(config)


if __name__ == "__main__":
    cli()
