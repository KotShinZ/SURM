from typing import Dict, List, Tuple
from dataclasses import dataclass
import os
import json
import hashlib

import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel

from data.common import PuzzleDatasetMetadata, dihedral_transform, inverse_dihedral_transform


cli = ArgParser()


class DataProcessConfig(BaseModel):
    input_file_prefix: str
    output_dir: str
    subsets: List[str]

    test_set_name: str

    seed: int = 42
    num_aug: int = 1000
    no_padding: bool = True
    min_context_pairs: int = 2


ARCMaxGridSize = 30
ARCAugmentRetriesFactor = 5

PuzzleIdSeparator = "|||"
DummyPuzzleIdentifier = 1


@dataclass
class ARCFullPuzzle:
    id: str
    pairs: List[Tuple[np.ndarray, np.ndarray]]
    target_indices: List[int]


def arc_grid_to_np(grid: List[List[int]]):
    arr = np.array(grid)

    # Shape check
    assert arr.ndim == 2
    assert arr.shape[0] <= ARCMaxGridSize and arr.shape[1] <= ARCMaxGridSize
    # Element check
    assert np.all((arr >= 0) & (arr <= 9))
    return arr.astype(np.uint8)


def np_grid_to_fixed_seq_translational_augment(
    inp: np.ndarray,
    out: np.ndarray,
    do_translation: bool,
):
    # PAD: 0, <eos>: 1, digits: 2 ... 11
    # Compute random top-left pad
    if do_translation:
        pad_r = np.random.randint(0, ARCMaxGridSize - max(inp.shape[0], out.shape[0]) + 1)
        pad_c = np.random.randint(0, ARCMaxGridSize - max(inp.shape[1], out.shape[1]) + 1)
    else:
        pad_r = pad_c = 0

    # Pad grid
    result = []
    for grid in [inp, out]:
        nrow, ncol = grid.shape
        grid = np.pad(
            grid + 2,
            ((pad_r, ARCMaxGridSize - pad_r - nrow), (pad_c, ARCMaxGridSize - pad_c - ncol)),
            constant_values=0,
        )

        # Add <eos>
        eos_row, eos_col = pad_r + nrow, pad_c + ncol
        if eos_row < ARCMaxGridSize:
            grid[eos_row, pad_c:eos_col] = 1
        if eos_col < ARCMaxGridSize:
            grid[pad_r:eos_row, eos_col] = 1

        result.append(grid.flatten())

    return result


def np_grids_to_unpadded_seq(inp: np.ndarray, out: np.ndarray):
    # PAD: 0, <eos>: 1, digits: 2 ... 11
    #
    # The full-context dataset concatenates multiple grid-pairs into a 1D sequence.
    # Each individual pair still uses the smallest shared canvas that can contain
    # both grids plus the EOS border when the ARC 30x30 limit leaves room for it.
    canvas_h = max(inp.shape[0], out.shape[0])
    canvas_w = max(inp.shape[1], out.shape[1])
    if canvas_h < ARCMaxGridSize:
        canvas_h += 1
    if canvas_w < ARCMaxGridSize:
        canvas_w += 1

    result = []
    for grid in [inp, out]:
        nrow, ncol = grid.shape
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        canvas[:nrow, :ncol] = grid + 2

        if nrow < canvas_h:
            canvas[nrow, :ncol] = 1
        if ncol < canvas_w:
            canvas[:nrow, ncol] = 1

        result.append(canvas.flatten())

    return result, (canvas_h, canvas_w)


def grid_hash(grid: np.ndarray):
    assert grid.ndim == 2
    assert grid.dtype == np.uint8

    buffer = [x.to_bytes(1, "big") for x in grid.shape]
    buffer.append(grid.tobytes())

    return hashlib.sha256(b"".join(buffer)).hexdigest()


def puzzle_hash(puzzle: ARCFullPuzzle):
    hashes = []
    for inp, out in puzzle.pairs:
        hashes.append(f"{grid_hash(inp)}|{grid_hash(out)}")

    hashes.sort()
    target_repr = ",".join(str(x) for x in puzzle.target_indices)
    return hashlib.sha256(f"{target_repr}|{'|'.join(hashes)}".encode()).hexdigest()


def aug(name: str):
    # Augment plan
    trans_id = np.random.randint(0, 8)
    mapping = np.concatenate(
        [np.arange(0, 1, dtype=np.uint8), np.random.permutation(np.arange(1, 10, dtype=np.uint8))]
    )  # Permute colors, Excluding "0" (black)

    name_with_aug_repr = (
        f"{name}{PuzzleIdSeparator}t{trans_id}{PuzzleIdSeparator}{''.join(str(x) for x in mapping)}"
    )

    def _map_grid(grid: np.ndarray):
        return dihedral_transform(mapping[grid], trans_id)

    return name_with_aug_repr, _map_grid


def inverse_aug(name: str):
    # Inverse the "aug" function
    if PuzzleIdSeparator not in name:
        return name, lambda x: x

    trans_id, perm = name.split(PuzzleIdSeparator)[-2:]
    trans_id = int(trans_id[1:])  # Remove "t" letter
    inv_perm = np.argsort(list(perm)).astype(np.uint8)

    def _map_grid(grid: np.ndarray):
        return inv_perm[inverse_dihedral_transform(grid, trans_id)]

    return name.split(PuzzleIdSeparator)[0], _map_grid


def _build_train_template(name: str, train_pairs: List[Tuple[np.ndarray, np.ndarray]]):
    return ARCFullPuzzle(
        id=name,
        pairs=list(train_pairs),
        target_indices=list(range(len(train_pairs))),
    )


def _build_joint_template(
    name: str,
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    test_pairs: List[Tuple[np.ndarray, np.ndarray]],
    target_indices: List[int],
):
    return ARCFullPuzzle(
        id=name,
        pairs=[*train_pairs, *test_pairs],
        target_indices=target_indices,
    )


def convert_single_arc_puzzle(
    results: dict,
    name: str,
    puzzle: dict,
    aug_count: int,
    min_context_pairs: int,
    dest_mapping: Dict[str, Tuple[str, str]],
):
    train_pairs = [
        (arc_grid_to_np(example["input"]), arc_grid_to_np(example["output"]))
        for example in puzzle.get("train", [])
    ]
    test_pairs = [
        (arc_grid_to_np(example["input"]), arc_grid_to_np(example["output"]))
        for example in puzzle.get("test", [])
    ]
    # print(f"Puzzle {name}: {len(train_pairs)} train pairs, {len(test_pairs)} test pairs")

    train_dest = dest_mapping["train"]
    test_dest = dest_mapping["test"]

    converted: Dict[Tuple[str, str], ARCFullPuzzle] = {}
    if train_dest == test_dest:
        all_pairs = [*train_pairs, *test_pairs]
        converted[train_dest] = ARCFullPuzzle(
            id=name,
            pairs=all_pairs,
            target_indices=list(range(len(all_pairs))),
        )
    else:
        converted[train_dest] = _build_train_template(name, train_pairs)
        test_target_indices = [len(train_pairs) + len(test_pairs) - 1] if test_pairs else []
        converted[test_dest] = _build_joint_template(
            name,
            train_pairs=train_pairs,
            test_pairs=test_pairs,
            target_indices=test_target_indices,
        )

    # Keep only templates that can actually emit at least one full-context sample.
    converted = {
        dest: template
        for dest, template in converted.items()
        if len(template.pairs) >= min_context_pairs + 1 and len(template.target_indices) > 0
    }
    if not converted:
        return False

    group = [converted]

    # Augment
    if aug_count > 0:
        hashes = {"|".join(sorted(puzzle_hash(template) for template in converted.values()))}

        for _trial in range(ARCAugmentRetriesFactor * aug_count):
            aug_name, map_grid = aug(name)
            augmented = {
                dest: ARCFullPuzzle(
                    id=aug_name,
                    pairs=[(map_grid(inp), map_grid(out)) for inp, out in template.pairs],
                    target_indices=list(template.target_indices),
                )
                for dest, template in converted.items()
            }

            h = "|".join(sorted(puzzle_hash(template) for template in augmented.values()))
            if h not in hashes:
                hashes.add(h)
                group.append(augmented)

            if len(group) >= aug_count + 1:
                break

    # Append
    for dest in converted.keys():
        dest_split, dest_set = dest

        results.setdefault(dest_split, {})
        results[dest_split].setdefault(dest_set, [])
        results[dest_split][dest_set].append([converted_map[dest] for converted_map in group])

    return True


def load_puzzles_arcagi(config: DataProcessConfig):
    if config.min_context_pairs < 2:
        raise ValueError(f"min_context_pairs must be >= 2, got {config.min_context_pairs}")

    train_examples_dest = ("train", "all")
    test_examples_map = {
        config.test_set_name: [(1.0, ("test", "all"))],
        "_default": [(1.0, ("train", "all"))],
    }

    test_puzzles = {}
    results = {}

    total_puzzles = 0
    skipped_puzzles = 0
    for subset_name in config.subsets:
        # Load all puzzles in this subset
        with open(f"{config.input_file_prefix}_{subset_name}-challenges.json", "r") as f:
            puzzles = json.load(f)
            print(f"Loaded {len(puzzles)} puzzles from {subset_name} challenges")

        sols_filename = f"{config.input_file_prefix}_{subset_name}-solutions.json"
        if os.path.isfile(sols_filename):
            with open(sols_filename, "r") as f:
                sols = json.load(f)
                print(f"Loaded {len(sols)} solutions from {subset_name} solutions")

                for puzzle_id in puzzles.keys():
                    for idx, sol_grid in enumerate(sols[puzzle_id]):
                        puzzles[puzzle_id]["test"][idx]["output"] = sol_grid
        else:
            print(f"{subset_name} solutions not found, filling with dummy")
            for puzzle_id, puzzle in puzzles.items():
                for example in puzzle["test"]:
                    example.setdefault("output", [[0]])

        # Shuffle puzzles
        puzzles = list(puzzles.items())
        print(f"Shuffling {len(puzzles)} puzzles...")
        np.random.shuffle(puzzles)

        # Assign by fraction
        for idx, (name, puzzle) in enumerate(puzzles):
            fraction = idx / len(puzzles)
            test_examples_dest = None
            for f, dest in test_examples_map.get(subset_name, test_examples_map["_default"]):
                if fraction < f:
                    test_examples_dest = dest
                    break

            assert test_examples_dest is not None

            converted = convert_single_arc_puzzle(
                results,
                name,
                puzzle,
                config.num_aug,
                config.min_context_pairs,
                {"train": train_examples_dest, "test": test_examples_dest},
            )
            if not converted:
                skipped_puzzles += 1
                continue

            if test_examples_dest[0] == "test":
                test_puzzles[name] = puzzle

            total_puzzles += 1

    print(f"Total convertible puzzles: {total_puzzles}")
    print(f"Skipped puzzles (not enough solved pairs): {skipped_puzzles}")
    print("results keys:", results.keys())
    return results, test_puzzles


def _sample_context_indices(
    num_pairs: int,
    target_idx: int,
    min_context_pairs: int,
):
    candidate_indices = [idx for idx in range(num_pairs) if idx != target_idx]
    if len(candidate_indices) < min_context_pairs:
        return None

    num_context = np.random.randint(min_context_pairs, len(candidate_indices) + 1)
    context_indices = np.random.choice(candidate_indices, size=num_context, replace=False).tolist()
    return context_indices


def _pair_seq_upper_bound(
    inp: np.ndarray,
    out: np.ndarray,
    no_padding: bool,
):
    if not no_padding:
        return 2 * ARCMaxGridSize * ARCMaxGridSize

    canvas_h = max(inp.shape[0], out.shape[0])
    canvas_w = max(inp.shape[1], out.shape[1])
    if canvas_h < ARCMaxGridSize:
        canvas_h += 1
    if canvas_w < ARCMaxGridSize:
        canvas_w += 1
    return 2 * canvas_h * canvas_w


def _full_sample_upper_bound(group: List[ARCFullPuzzle], no_padding: bool):
    if not group:
        return 0
    return max(
        sum(_pair_seq_upper_bound(inp, out, no_padding) for inp, out in puzzle.pairs)
        for puzzle in group
    )


def _make_pair_sequences(
    inp: np.ndarray,
    out: np.ndarray,
    do_translation: bool,
    no_padding: bool,
):
    if no_padding:
        (inp_seq, out_seq), pair_shape = np_grids_to_unpadded_seq(inp, out)
    else:
        inp_seq, out_seq = np_grid_to_fixed_seq_translational_augment(
            inp,
            out,
            do_translation=do_translation,
        )
        pair_shape = (ARCMaxGridSize, ARCMaxGridSize)

    return (
        inp_seq.astype(np.uint8, copy=False),
        out_seq.astype(np.uint8, copy=False),
        pair_shape,
    )


def _make_pair_position_ids(pair_shape: Tuple[int, int], example_index: int):
    pair_h, pair_w = pair_shape
    rows = np.repeat(np.arange(pair_h, dtype=np.uint8), pair_w)
    cols = np.tile(np.arange(pair_w, dtype=np.uint8), pair_h)
    depth = np.full((pair_h * pair_w,), example_index, dtype=np.uint8)
    input_io = np.zeros((pair_h * pair_w,), dtype=np.uint8)
    output_io = np.ones((pair_h * pair_w,), dtype=np.uint8)

    input_position_ids = np.stack([depth, input_io, rows, cols], axis=-1)
    output_position_ids = np.stack([depth, output_io, rows, cols], axis=-1)
    return input_position_ids, output_position_ids


def _build_full_context_example(
    puzzle: ARCFullPuzzle,
    target_idx: int,
    min_context_pairs: int,
    enable_translational_augment: bool,
    no_padding: bool,
    use_all_pairs_in_order: bool = False,
):
    if use_all_pairs_in_order:
        if target_idx != len(puzzle.pairs) - 1:
            raise ValueError(
                "Full-context evaluation requires the target pair to be the final pair."
            )
        ordered_indices = list(range(len(puzzle.pairs)))
    else:
        context_indices = _sample_context_indices(len(puzzle.pairs), target_idx, min_context_pairs)
        if context_indices is None:
            return None

        ordered_indices = [*context_indices, target_idx]
        np.random.shuffle(ordered_indices)

    no_aug_pair_pos = np.random.randint(0, len(ordered_indices))

    input_parts = []
    label_parts = []
    position_parts = []

    for pair_pos, pair_idx in enumerate(ordered_indices):
        inp, out = puzzle.pairs[pair_idx]
        do_translation = enable_translational_augment and pair_pos != no_aug_pair_pos
        inp_seq, out_seq, pair_shape = _make_pair_sequences(inp, out, do_translation, no_padding)

        zero_inp = np.zeros_like(inp_seq, dtype=np.uint8)
        zero_out = np.zeros_like(out_seq, dtype=np.uint8)
        inp_pos_ids, out_pos_ids = _make_pair_position_ids(pair_shape, pair_pos)

        if pair_idx == target_idx:
            input_parts.extend([inp_seq, zero_out])
            label_parts.extend([zero_inp, out_seq])
            position_parts.extend([inp_pos_ids, out_pos_ids])
        else:
            input_parts.extend([inp_seq, out_seq])
            label_parts.extend([zero_inp, zero_out])
            position_parts.extend([inp_pos_ids, out_pos_ids])

    sample_input = np.concatenate(input_parts).astype(np.uint8, copy=False)
    sample_label = np.concatenate(label_parts).astype(np.uint8, copy=False)
    sample_position_ids = np.concatenate(position_parts, axis=0).astype(np.uint8, copy=False)
    seq_shape = (1, int(sample_input.shape[0]))
    return sample_input, sample_label, seq_shape, sample_position_ids


def _display_window(length: int, max_items: int):
    if length <= max_items:
        return list(range(length))

    head = max_items // 2
    tail = max_items - head
    return [*range(head), None, *range(length - tail, length)]


def _format_arc_token(token: int):
    if token == 0:
        return "."
    if token == 1:
        return "E"
    if 2 <= token <= 11:
        return str(token - 2)
    return f"?{token}"


def _extract_pair_grid(
    data: np.ndarray,
    pos_id: np.ndarray,
    pair_idx: int,
    is_output: bool,
):
    pair_mask = pos_id[:, 0] == pair_idx
    if is_output:
        part_mask = pair_mask & (pos_id[:, 1] == 1)
        rows = pos_id[part_mask, 2]
    else:
        part_mask = pair_mask & (pos_id[:, 1] == 0)
        rows = pos_id[part_mask, 2]

    cols = pos_id[part_mask, 3]
    tokens = data[part_mask]

    if tokens.size == 0:
        return np.empty((0, 0), dtype=np.uint8)

    height = int(rows.max()) + 1
    width = int(cols.max()) + 1
    expected_size = height * width
    if tokens.size != expected_size:
        raise ValueError(
            f"Failed to reconstruct pair {pair_idx}: expected {expected_size} tokens, got {tokens.size}"
        )

    return tokens.reshape(height, width)


def _format_grid_lines(grid: np.ndarray, max_rows: int, max_cols: int):
    if grid.size == 0:
        return ["<empty>"]

    row_window = _display_window(grid.shape[0], max_rows)
    col_window = _display_window(grid.shape[1], max_cols)
    lines = []

    for row_idx in row_window:
        if row_idx is None:
            lines.append("... | ...")
            continue

        cells = []
        for col_idx in col_window:
            if col_idx is None:
                cells.append("...")
            else:
                cells.append(f"{_format_arc_token(int(grid[row_idx, col_idx])):>3}")
        lines.append(f"r{row_idx:02d} |{''.join(cells)}")

    return lines


def print_data(
    data: np.ndarray,
    pos_id: np.ndarray,
    title: str = "",
    max_rows: int = 12,
    max_cols: int = 12,
):
    if data.ndim != 1:
        raise ValueError(f"Expected 1D token sequence, got shape={data.shape}")
    if pos_id.ndim != 2 or pos_id.shape[1] != 4:
        raise ValueError(f"Expected position_ids with shape (N, 4), got shape={pos_id.shape}")
    if data.shape[0] != pos_id.shape[0]:
        raise ValueError(
            f"Token length and position_ids length must match, got {data.shape[0]} and {pos_id.shape[0]}"
        )
    if data.size == 0:
        print(f"{title or 'sample'}: <empty>")
        return

    num_pairs = int(pos_id[:, 0].max()) + 1
    sample_name = title or "sample"
    print(f"{sample_name}: tokens={data.shape[0]}, pairs={num_pairs}")
    print("legend: . = blank/pad, E = eos, 0-9 = ARC colors")

    for pair_idx in range(num_pairs):
        input_grid = _extract_pair_grid(data, pos_id, pair_idx, is_output=False)
        output_grid = _extract_pair_grid(data, pos_id, pair_idx, is_output=True)
        if input_grid.shape != output_grid.shape:
            raise ValueError(
                f"Input/output canvas mismatch at pair {pair_idx}: {input_grid.shape} vs {output_grid.shape}"
            )

        input_lines = _format_grid_lines(input_grid, max_rows=max_rows, max_cols=max_cols)
        output_lines = _format_grid_lines(output_grid, max_rows=max_rows, max_cols=max_cols)
        left_width = max(len(line) for line in input_lines)
        pair_shape = f"{input_grid.shape[0]}x{input_grid.shape[1]}"
        cropped = input_grid.shape[0] > max_rows or input_grid.shape[1] > max_cols
        crop_suffix = " (cropped)" if cropped else ""

        print(f"  Pair {pair_idx} | canvas={pair_shape}{crop_suffix}")
        print(f"    {'input':<{left_width}}    output")
        for line_idx in range(max(len(input_lines), len(output_lines))):
            left = input_lines[line_idx] if line_idx < len(input_lines) else ""
            right = output_lines[line_idx] if line_idx < len(output_lines) else ""
            print(f"    {left:<{left_width}}    {right}")
        print()


def convert_dataset(config: DataProcessConfig):
    np.random.seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    # Read dataset
    data, test_puzzles = load_puzzles_arcagi(config)

    # All real examples share a single dummy identifier so the model cannot
    # memorize task-specific embeddings.
    num_identifiers = 2  # 0 is blank, 1 is shared dummy

    global_seq_len = 0
    for split in data.values():
        for subset in split.values():
            for group in subset:
                global_seq_len = max(global_seq_len, _full_sample_upper_bound(group, config.no_padding))

    if global_seq_len <= 0:
        raise ValueError("No full-context ARC samples were generated.")

    # Save
    for split_name, split in data.items():
        print("split: ", split_name)
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)

        enable_translational_augment = split_name == "train"
        use_all_pairs_in_order = split_name == "test"

        total_examples = 0
        total_puzzles = 0
        total_groups = 0
        split_max_position_id = np.zeros((4,), dtype=np.int32)

        for subset_name, subset in split.items():
            results = {
                "inputs": [],
                "labels": [],
                "puzzle_identifiers": [],
                "puzzle_indices": [0],
                "group_indices": [0],
            }
            if config.no_padding:
                results["seq_shapes"] = []
                results["position_ids"] = []

            example_id = 0
            puzzle_id = 0

            for group in subset:
                for puzzle in group:
                    for target_idx in puzzle.target_indices:
                        built = _build_full_context_example(
                            puzzle=puzzle,
                            target_idx=target_idx,
                            min_context_pairs=config.min_context_pairs,
                            enable_translational_augment=enable_translational_augment,
                            no_padding=config.no_padding,
                            use_all_pairs_in_order=use_all_pairs_in_order,
                        )
                        if built is None:
                            continue

                        inp, out, seq_shape, position_ids = built
                        results["inputs"].append(inp)
                        results["labels"].append(out)
                        if config.no_padding:
                            results["seq_shapes"].append(seq_shape)
                            results["position_ids"].append(position_ids)
                            split_max_position_id = np.maximum(
                                split_max_position_id,
                                position_ids.max(axis=0).astype(np.int32) + 1,
                            )

                        example_id += 1
                        total_examples += 1

                    results["puzzle_indices"].append(example_id)
                    results["puzzle_identifiers"].append(DummyPuzzleIdentifier)

                    puzzle_id += 1
                    total_puzzles += 1

                results["group_indices"].append(puzzle_id)
                total_groups += 1

            for key, value in results.items():
                if key in {"inputs", "labels"}:
                    if config.no_padding:
                        if value:
                            target_id = 20
                            print_data(
                                value[target_id],
                                results["position_ids"][target_id],
                                title=f"{split_name}/{subset_name} {key}[{target_id}]",
                            )
                        seq_lengths = np.array([seq.shape[0] for seq in value], dtype=np.int64)
                        seq_offsets = np.concatenate(
                            [np.array([0], dtype=np.int64), np.cumsum(seq_lengths, dtype=np.int64)]
                        )
                        flat_tokens = (
                            np.concatenate(value).astype(np.uint8, copy=False)
                            if value
                            else np.empty((0,), dtype=np.uint8)
                        )
                        np.save(
                            os.path.join(config.output_dir, split_name, f"{subset_name}__{key}.npy"),
                            flat_tokens,
                        )
                        if key == "inputs":
                            np.save(
                                os.path.join(config.output_dir, split_name, f"{subset_name}__seq_offsets.npy"),
                                seq_offsets,
                            )
                    else:
                        padded = [
                            np.pad(seq, (0, global_seq_len - seq.shape[0]), constant_values=0)
                            for seq in value
                        ]
                        array = (
                            np.stack(padded, 0).astype(np.uint8, copy=False)
                            if padded
                            else np.empty((0, global_seq_len), dtype=np.uint8)
                        )
                        np.save(
                            os.path.join(config.output_dir, split_name, f"{subset_name}__{key}.npy"),
                            array,
                        )
                elif key == "position_ids":
                    flat_positions = (
                        np.concatenate(value, axis=0).astype(np.uint8, copy=False)
                        if value
                        else np.empty((0, 4), dtype=np.uint8)
                    )
                    np.save(
                        os.path.join(config.output_dir, split_name, f"{subset_name}__{key}.npy"),
                        flat_positions,
                    )
                elif key == "seq_shapes":
                    np.save(
                        os.path.join(config.output_dir, split_name, f"{subset_name}__{key}.npy"),
                        np.array(value, dtype=np.int32),
                    )
                else:
                    np.save(
                        os.path.join(config.output_dir, split_name, f"{subset_name}__{key}.npy"),
                        np.array(value, dtype=np.int32),
                    )

        metadata = PuzzleDatasetMetadata(
            seq_len=global_seq_len,
            vocab_size=10 + 2,  # PAD + EOS + "0" ... "9"
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=total_examples / max(total_puzzles, 1),
            sets=list(split.keys()),
            variable_seq_lengths=config.no_padding,
            position_id_shape=split_max_position_id.tolist() if config.no_padding and total_examples > 0 else None,
        )
        print(f"  Total puzzles: {total_puzzles}")
        print(f"  Total examples: {total_examples}")
        print(f"  Total groups: {total_groups}")
        print(f"  Mean examples per puzzle: {metadata.mean_puzzle_examples:.2f}")
        print(f"  Sequence length upper bound: {metadata.seq_len}")
        if metadata.position_id_shape is not None:
            print(f"  Position ID shape: {metadata.position_id_shape}")

        with open(os.path.join(config.output_dir, split_name, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)
            print(f"  Saved metadata to {f.name}")

    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>", "<shared_dummy>"], f)

    with open(os.path.join(config.output_dir, "test_puzzles.json"), "w") as f:
        json.dump(test_puzzles, f)


@cli.command(singleton=True)
def main(config: DataProcessConfig):
    convert_dataset(config)


if __name__ == "__main__":
    cli()
