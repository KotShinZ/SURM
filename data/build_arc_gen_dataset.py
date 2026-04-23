from __future__ import annotations

from typing import Callable, Dict, List, Literal, Optional, Tuple
import hashlib
import importlib.util
import json
import os
import random
import signal
import sys

import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel, model_validator
from tqdm import tqdm

from data.build_arc_dataset import (
    ARCMaxGridSize,
    convert_single_arc_puzzle,
    filter_arc_examples_by_size,
    filter_puzzles_by_size,
    np_grid_to_fixed_seq_translational_augment,
    np_grids_to_unpadded_seq,
)
from data.common import PuzzleDatasetMetadata


cli = ArgParser()

ARC_GEN_V1_TASK_COUNT = 400


class DataProcessConfig(BaseModel):
    output_dir: str

    seed: int = 42

    # ARC-GEN already emits many examples per task, so keep family augmentation
    # disabled by default and make the per-task sample count configurable.
    num_aug: int = 0
    no_padding: bool = False

    arc_gen_root: str = "ARC-GEN"
    examples_per_task: int = 25000
    max_generation_attempts_per_task: Optional[int] = None
    generator_timeout_sec: Optional[float] = 1.0
    task_version: Literal["all", "v1", "v2"] = "all"
    task_ids: Optional[List[str]] = None

    @model_validator(mode="after")
    def _validate_ranges(self):
        if self.examples_per_task <= 0:
            raise ValueError(
                f"examples_per_task must be a positive integer, got {self.examples_per_task}"
            )
        if self.num_aug < 0:
            raise ValueError(f"num_aug must be >= 0, got {self.num_aug}")
        if (
            self.max_generation_attempts_per_task is not None
            and self.max_generation_attempts_per_task < self.examples_per_task
        ):
            raise ValueError(
                "max_generation_attempts_per_task must be >= examples_per_task when provided, "
                f"got {self.max_generation_attempts_per_task} and {self.examples_per_task}"
            )
        if self.generator_timeout_sec is not None and self.generator_timeout_sec <= 0:
            raise ValueError(
                "generator_timeout_sec must be > 0 when provided, "
                f"got {self.generator_timeout_sec}"
            )
        return self


TaskRegistry = Dict[str, Tuple[Callable[[], dict], Callable[[], dict]]]


def load_arc_gen_task_registry(arc_gen_root: str) -> TaskRegistry:
    arc_gen_root = os.path.abspath(arc_gen_root)
    task_list_path = os.path.join(arc_gen_root, "task_list.py")
    if not os.path.isfile(task_list_path):
        raise FileNotFoundError(
            f"ARC-GEN task list not found at {task_list_path}. "
            f"Set arc_gen_root to the ARC-GEN checkout directory."
        )

    if arc_gen_root not in sys.path:
        sys.path.insert(0, arc_gen_root)

    module_hash = hashlib.sha256(arc_gen_root.encode("utf-8")).hexdigest()[:12]
    module_name = f"_arc_gen_task_list_{module_hash}"

    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, task_list_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load ARC-GEN task list from {task_list_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    registry = module.task_list()
    return {task_id: (task_info[0], task_info[1]) for task_id, task_info in registry.items()}


def _normalize_arc_example(example: dict, *, task_id: str, source_name: str, index: int) -> dict:
    if not isinstance(example, dict):
        raise TypeError(
            f"{source_name} example for task {task_id} at index {index} must be an object, "
            f"got {type(example).__name__}"
        )
    if "input" not in example or "output" not in example:
        raise KeyError(
            f"{source_name} example for task {task_id} at index {index} must contain "
            "'input' and 'output'"
        )
    return {
        "input": example["input"],
        "output": example["output"],
    }


def _normalize_arc_puzzle(puzzle: dict, *, task_id: str, source_name: str) -> dict:
    if not isinstance(puzzle, dict):
        raise TypeError(
            f"{source_name} puzzle for task {task_id} must be an object, got {type(puzzle).__name__}"
        )

    normalized = {}
    for example_type in ["train", "test"]:
        examples = puzzle.get(example_type, [])
        if not isinstance(examples, list):
            raise TypeError(
                f"{source_name} puzzle field '{example_type}' for task {task_id} must be a list, "
                f"got {type(examples).__name__}"
            )
        normalized[example_type] = [
            _normalize_arc_example(
                example,
                task_id=task_id,
                source_name=source_name,
                index=index,
            )
            for index, example in enumerate(examples)
        ]

    return normalized


def _resolve_task_ids(config: DataProcessConfig, registry: TaskRegistry) -> List[str]:
    allowed_task_ids = _resolve_task_version_task_ids(registry, config.task_version)
    available_task_ids = [task_id for task_id in sorted(registry.keys()) if task_id in allowed_task_ids]
    if config.task_ids is None:
        return available_task_ids

    requested_task_ids = list(dict.fromkeys(config.task_ids))
    missing_task_ids = [task_id for task_id in requested_task_ids if task_id not in registry]
    if missing_task_ids:
        raise ValueError(
            "Unknown ARC-GEN task ids requested: "
            + ", ".join(sorted(missing_task_ids))
        )

    disallowed_task_ids = [task_id for task_id in requested_task_ids if task_id not in allowed_task_ids]
    if disallowed_task_ids:
        raise ValueError(
            f"Requested task ids are not part of task_version={config.task_version}: "
            + ", ".join(sorted(disallowed_task_ids))
        )
    return requested_task_ids


def _resolve_task_version_task_ids(
    registry: TaskRegistry,
    task_version: Literal["all", "v1", "v2"],
    *,
    v1_task_count: Optional[int] = None,
) -> set[str]:
    if v1_task_count is None:
        v1_task_count = ARC_GEN_V1_TASK_COUNT

    ordered_task_ids = list(registry.keys())
    if task_version == "all":
        return set(ordered_task_ids)

    if len(ordered_task_ids) < v1_task_count:
        raise ValueError(
            f"ARC-GEN registry has only {len(ordered_task_ids)} tasks, "
            f"which is smaller than the expected V1 boundary {v1_task_count}"
        )

    v1_task_ids = set(ordered_task_ids[:v1_task_count])
    if task_version == "v1":
        return v1_task_ids
    return set(ordered_task_ids[v1_task_count:])


def _generate_train_examples_for_task(
    config: DataProcessConfig,
    task_id: str,
    generator: Callable[[], dict],
) -> Tuple[List[dict], int, int, int, Optional[str]]:
    max_attempts = config.max_generation_attempts_per_task
    if max_attempts is None:
        max_attempts = max(config.examples_per_task, config.examples_per_task * 20)

    generated_examples: List[dict] = []
    removed_examples = 0
    attempts = 0
    failed_attempts = 0
    first_error_message: Optional[str] = None
    
    # print("examples_per_task:", config.examples_per_task)

    while len(generated_examples) < config.examples_per_task and attempts < max_attempts:
        # Seed per task/attempt so output does not depend on task iteration order.
        random.seed(f"{config.seed}:{task_id}:{attempts}")
        try:
            previous_handler = None
            timeout_sec = config.generator_timeout_sec
            if timeout_sec is not None and hasattr(signal, "SIGALRM"):
                def _timeout_handler(signum, frame):
                    raise TimeoutError(
                        f"ARC-GEN generator timed out after {timeout_sec:.3f}s "
                        f"(task={task_id}, attempt={attempts})"
                    )

                previous_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, timeout_sec)

            try:
                example = generator()
            finally:
                if timeout_sec is not None and hasattr(signal, "SIGALRM"):
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, previous_handler)
        except Exception as exc:
            failed_attempts += 1
            if first_error_message is None:
                first_error_message = f"{type(exc).__name__}: {exc}"
            attempts += 1
            continue

        normalized_example = _normalize_arc_example(
            example,
            task_id=task_id,
            source_name="ARC-GEN generator",
            index=attempts,
        )
        filtered_examples, removed = filter_arc_examples_by_size([normalized_example])
        generated_examples.extend(filtered_examples)
        removed_examples += removed
        attempts += 1

    return (
        generated_examples[: config.examples_per_task],
        attempts,
        removed_examples,
        failed_attempts,
        first_error_message,
    )


def load_puzzles_arc_gen(config: DataProcessConfig):
    train_examples_dest = ("train", "all")
    test_examples_dest = ("test", "all")

    task_registry = load_arc_gen_task_registry(config.arc_gen_root)
    task_ids = _resolve_task_ids(config, task_registry)

    print(f"Loaded ARC-GEN registry with {len(task_registry)} tasks from {config.arc_gen_root}")
    print(f"Selected {len(task_ids)} ARC-GEN tasks (task_version={config.task_version})")

    canonical_puzzles = {}
    for task_id in tqdm(task_ids, desc="Loading ARC-GEN validator puzzles"):
        _, validator = task_registry[task_id]
        canonical_puzzles[task_id] = _normalize_arc_puzzle(
            validator(),
            task_id=task_id,
            source_name="ARC-GEN validator",
        )

    canonical_puzzles = filter_puzzles_by_size(
        canonical_puzzles,
        source_name="arc-gen validator",
    )
    print(f"Kept {len(canonical_puzzles)} ARC-GEN validator puzzles after size filtering")

    generated_train_examples_by_task = {}
    total_generated_examples = 0
    total_attempts = 0
    total_removed_examples = 0
    total_failed_attempts = 0
    shortfall_tasks = {}
    failed_generation_tasks = {}

    for task_id in tqdm(task_ids, desc="Generating ARC-GEN train examples"):
        generator, _ = task_registry[task_id]
        (
            generated_examples,
            attempts,
            removed_examples,
            failed_attempts,
            first_error_message,
        ) = _generate_train_examples_for_task(config, task_id, generator)
        generated_train_examples_by_task[task_id] = generated_examples
        total_generated_examples += len(generated_examples)
        total_attempts += attempts
        total_removed_examples += removed_examples
        total_failed_attempts += failed_attempts
        if len(generated_examples) < config.examples_per_task:
            shortfall_tasks[task_id] = len(generated_examples)
        if failed_attempts > 0:
            failed_generation_tasks[task_id] = {
                "failed_attempts": failed_attempts,
                "first_error": first_error_message,
            }

    print(
        f"Generated {total_generated_examples} ARC-GEN train examples "
        f"from {total_attempts} attempts"
    )
    if total_failed_attempts > 0:
        print(
            f"Observed {total_failed_attempts} ARC-GEN generator exceptions across "
            f"{len(failed_generation_tasks)} tasks; skipped those attempts and continued"
        )
    if total_removed_examples > 0:
        print(
            f"Filtered out {total_removed_examples} generated examples with size >= "
            f"{ARCMaxGridSize}x{ARCMaxGridSize}"
        )
    if shortfall_tasks:
        print(
            f"{len(shortfall_tasks)} tasks fell short of the requested "
            f"{config.examples_per_task} examples"
        )
        preview = ", ".join(
            f"{task_id}={count}" for task_id, count in list(sorted(shortfall_tasks.items()))[:10]
        )
        if preview:
            print(f"Shortfall preview: {preview}")
    if failed_generation_tasks:
        preview = ", ".join(
            (
                f"{task_id}={info['failed_attempts']}"
                + (
                    f" ({info['first_error']})"
                    if info["first_error"] is not None
                    else ""
                )
            )
            for task_id, info in list(sorted(failed_generation_tasks.items()))[:10]
        )
        if preview:
            print(f"Generator failure preview: {preview}")

    puzzles = {}
    for task_id in task_ids:
        canonical_puzzle = canonical_puzzles.get(task_id, {"train": [], "test": []})
        generated_examples = generated_train_examples_by_task.get(task_id, [])

        if not generated_examples and not canonical_puzzle.get("test", []):
            continue

        puzzles[task_id] = {
            "train": generated_examples,
            "test": canonical_puzzle.get("test", []),
        }

    puzzles = list(puzzles.items())
    print(f"Converting {len(puzzles)} ARC-GEN tasks into dataset groups")
    np.random.shuffle(puzzles)

    results = {}
    test_puzzles = {}
    total_source_groups = 0
    output_group_counts = {}
    output_puzzle_counts = {}

    for task_id, puzzle in tqdm(puzzles, desc="Converting ARC-GEN tasks"):
        canonical_puzzle = canonical_puzzles.get(task_id)
        if canonical_puzzle and canonical_puzzle.get("test", []):
            test_puzzles[task_id] = canonical_puzzle

        (
            used_dests,
            _added_arc_gen_seed_puzzles,
            _added_arc_gen_total_puzzles,
            puzzle_counts_by_dest,
        ) = convert_single_arc_puzzle(
            results,
            task_id,
            puzzle,
            config.num_aug,
            {"train": train_examples_dest, "test": test_examples_dest},
        )
        if used_dests:
            total_source_groups += 1
        for dest, num_puzzles in puzzle_counts_by_dest.items():
            output_group_counts[dest] = output_group_counts.get(dest, 0) + 1
            output_puzzle_counts[dest] = output_puzzle_counts.get(dest, 0) + num_puzzles

    print(f"Total source task groups: {total_source_groups}")
    print(f"Source task groups with held-out test examples: {len(test_puzzles)}")
    print(f"Source task groups routed only to train: {total_source_groups - len(test_puzzles)}")
    for dest in sorted(output_group_counts):
        print(
            f"Output {dest[0]}/{dest[1]}: "
            f"groups={output_group_counts[dest]}, puzzles={output_puzzle_counts[dest]}"
        )

    return results, test_puzzles


def _save_split_dataset(
    split_name: str,
    split: dict,
    *,
    output_dir: str,
    identifier_map: Dict[str, int],
    no_padding: bool,
) -> Tuple[int, int, int]:
    split_output_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_output_dir, exist_ok=True)

    enable_translational_augment = split_name == "train"

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

        for group in tqdm(subset, desc=f"Processing {split_name}/{subset_name}"):
            for puzzle in group:
                no_aug_id = np.random.randint(0, len(puzzle.examples))
                for example_index, (inp, out) in enumerate(puzzle.examples):
                    if no_padding:
                        (inp, out), seq_shape = np_grids_to_unpadded_seq(inp, out)
                        results.setdefault("seq_shapes", []).append(seq_shape)
                    else:
                        inp, out = np_grid_to_fixed_seq_translational_augment(
                            inp,
                            out,
                            do_translation=enable_translational_augment and example_index != no_aug_id,
                        )

                    results["inputs"].append(inp)
                    results["labels"].append(out)
                    example_id += 1
                    total_examples += 1

                results["puzzle_indices"].append(example_id)
                results["puzzle_identifiers"].append(identifier_map[puzzle.id])
                puzzle_id += 1
                total_puzzles += 1

            results["group_indices"].append(puzzle_id)
            total_groups += 1

        print(
            f"{split_name}/{subset_name}: "
            f"groups={len(subset)}, puzzles={puzzle_id}, examples={example_id}"
        )

        for key, value in results.items():
            if no_padding and key in {"inputs", "labels"}:
                if value:
                    seq_lengths = np.array([seq.shape[0] for seq in value], dtype=np.int64)
                    seq_offsets = np.concatenate(
                        [np.array([0], dtype=np.int64), np.cumsum(seq_lengths, dtype=np.int64)]
                    )
                    flat_tokens = np.concatenate(value).astype(np.uint8, copy=False)
                else:
                    seq_offsets = np.array([0], dtype=np.int64)
                    flat_tokens = np.empty((0,), dtype=np.uint8)

                np.save(
                    os.path.join(split_output_dir, f"{subset_name}__{key}.npy"),
                    flat_tokens,
                )
                if key == "inputs":
                    np.save(
                        os.path.join(split_output_dir, f"{subset_name}__seq_offsets.npy"),
                        seq_offsets,
                    )
            elif key in {"inputs", "labels"}:
                if value:
                    array = np.stack(value, axis=0)
                else:
                    array = np.empty((0, ARCMaxGridSize * ARCMaxGridSize), dtype=np.uint8)
                np.save(os.path.join(split_output_dir, f"{subset_name}__{key}.npy"), array)
            elif key == "seq_shapes":
                if value:
                    array = np.array(value, dtype=np.int32)
                else:
                    array = np.empty((0, 2), dtype=np.int32)
                np.save(os.path.join(split_output_dir, f"{subset_name}__{key}.npy"), array)
            else:
                np.save(
                    os.path.join(split_output_dir, f"{subset_name}__{key}.npy"),
                    np.array(value, dtype=np.int32),
                )

    metadata = PuzzleDatasetMetadata(
        seq_len=ARCMaxGridSize * ARCMaxGridSize,
        vocab_size=12,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=max(identifier_map.values(), default=0) + 1,
        total_groups=total_groups,
        mean_puzzle_examples=(total_examples / total_puzzles) if total_puzzles > 0 else 0.0,
        sets=list(split.keys()),
        variable_seq_lengths=no_padding,
    )

    with open(os.path.join(split_output_dir, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f)

    print(f"{split_name}: total_puzzles={total_puzzles}, total_examples={total_examples}")
    return total_examples, total_puzzles, total_groups


def convert_dataset(config: DataProcessConfig):
    np.random.seed(config.seed)

    data, test_puzzles = load_puzzles_arc_gen(config)

    identifier_map = {}
    num_identifiers = 1  # 0 is blank

    print("Mapping puzzle IDs...")
    for split_name, split in data.items():
        print(f"split: {split_name}")
        for subset_name, subset in split.items():
            subset_group_count = len(subset)
            subset_puzzle_count = sum(len(group) for group in subset)
            print(
                f"  subset: {subset_name}, "
                f"groups={subset_group_count}, puzzles={subset_puzzle_count}"
            )
            for group in tqdm(
                subset,
                desc=f"Mapping IDs {split_name}/{subset_name}",
                leave=False,
            ):
                for puzzle in group:
                    if puzzle.id not in identifier_map:
                        identifier_map[puzzle.id] = num_identifiers
                        num_identifiers += 1

    print(f"Total puzzle IDs (including <blank>): {num_identifiers}")

    for split_name, split in data.items():
        _save_split_dataset(
            split_name,
            split,
            output_dir=config.output_dir,
            identifier_map=identifier_map,
            no_padding=config.no_padding,
        )

    os.makedirs(config.output_dir, exist_ok=True)

    with open(os.path.join(config.output_dir, "identifiers.json"), "w", encoding="utf-8") as f:
        ids_mapping = {value: key for key, value in identifier_map.items()}
        json.dump([ids_mapping.get(index, "<blank>") for index in range(num_identifiers)], f)

    with open(os.path.join(config.output_dir, "test_puzzles.json"), "w", encoding="utf-8") as f:
        json.dump(test_puzzles, f)


@cli.command(singleton=True)
def main(config: DataProcessConfig):
    convert_dataset(config)


if __name__ == "__main__":
    cli()
