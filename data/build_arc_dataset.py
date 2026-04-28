from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import os
import json
import hashlib
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from data.common import PuzzleDatasetMetadata, dihedral_transform, inverse_dihedral_transform


cli = ArgParser()


class DataProcessConfig(BaseModel):
    input_file_prefix: str
    output_dir: str
    subsets: List[str]

    test_set_name: str

    seed: int = 42
    num_aug: int = 1000
    num_aug_gen: Optional[int] = None
    no_padding: bool = False
    include_arc_gen: bool = False
    arc_gen_dir: Optional[str] = "data/arc-gen"
    
    
ARCMaxGridSize = 30
ARCAugmentRetriesFactor = 5

PuzzleIdSeparator = "|||"

_ARC_TOKEN_BG_COLORS = {
    2: 16,   # ARC color 0: black
    3: 21,   # ARC color 1: blue
    4: 196,  # ARC color 2: red
    5: 46,   # ARC color 3: green
    6: 226,  # ARC color 4: yellow
    7: 244,  # ARC color 5: gray
    8: 201,  # ARC color 6: magenta
    9: 208,  # ARC color 7: orange
    10: 51,  # ARC color 8: cyan
    11: 88,  # ARC color 9: dark red
}
    

@dataclass
class ARCPuzzle:
    id: str

    examples: List[Tuple[np.ndarray, np.ndarray]]
    context_examples: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None

    
def arc_grid_to_np(grid: List[List[int]]):
    arr = np.array(grid)

    # Shape check
    assert arr.ndim == 2
    assert arr.shape[0] <= ARCMaxGridSize and arr.shape[1] <= ARCMaxGridSize
    # Element check
    assert np.all((arr >= 0) & (arr <= 9))
    return arr.astype(np.uint8)


def arc_grid_shape(grid: List[List[int]]):
    return len(grid), max((len(row) for row in grid), default=0)


def is_supported_arc_grid(grid: List[List[int]]):
    height, width = arc_grid_shape(grid)
    return height <= ARCMaxGridSize and width <= ARCMaxGridSize


def filter_arc_examples_by_size(examples: List[dict]):
    filtered_examples = []
    removed_examples = 0

    for example in examples:
        if is_supported_arc_grid(example["input"]) and is_supported_arc_grid(example["output"]):
            filtered_examples.append(example)
        else:
            removed_examples += 1

    return filtered_examples, removed_examples


def filter_puzzles_by_size(
    puzzles: Dict[str, dict],
    *,
    source_name: str,
    extra_train_examples_by_puzzle: Optional[Dict[str, List[dict]]] = None,
):
    filtered_puzzles = {}
    removed_by_example_type = {}
    removed_puzzles = 0

    for puzzle_id, puzzle in puzzles.items():
        filtered_puzzle = {}
        kept_any_examples = False

        for example_type, examples in puzzle.items():
            filtered_examples, removed_examples = filter_arc_examples_by_size(examples)
            filtered_puzzle[example_type] = filtered_examples
            kept_any_examples = kept_any_examples or bool(filtered_examples)
            removed_by_example_type[example_type] = (
                removed_by_example_type.get(example_type, 0) + removed_examples
            )

        # ARC-GEN examples are merged into the matching source task group before
        # augmentation. They still keep the original task group alive after size
        # filtering when all ARC-AGI examples for that task are filtered out.
        if extra_train_examples_by_puzzle and extra_train_examples_by_puzzle.get(puzzle_id):
            kept_any_examples = True

        if kept_any_examples:
            filtered_puzzles[puzzle_id] = filtered_puzzle
        else:
            removed_puzzles += 1

    removed_summary = ", ".join(
        f"{example_type}={count}"
        for example_type, count in sorted(removed_by_example_type.items())
        if count > 0
    )
    if removed_summary:
        print(
            f"Filtered out {source_name} examples with size >= {ARCMaxGridSize}x{ARCMaxGridSize}: "
            f"{removed_summary}"
        )
    if removed_puzzles > 0:
        print(
            f"Dropped {removed_puzzles} {source_name} puzzles after size filtering because "
            f"no examples remained"
        )

    return filtered_puzzles


def np_grid_to_fixed_seq_translational_augment(inp: np.ndarray, out: np.ndarray, do_translation: bool):
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
        grid = np.pad(grid + 2, ((pad_r, ARCMaxGridSize - pad_r - nrow), (pad_c, ARCMaxGridSize - pad_c - ncol)), constant_values=0)

        # Add <eos>
        eos_row, eos_col = pad_r + nrow, pad_c + ncol
        if eos_row < ARCMaxGridSize:
            grid[eos_row, pad_c:eos_col] = 1
        if eos_col < ARCMaxGridSize:
            grid[pad_r:eos_row, eos_col] = 1

        result.append(grid.flatten())

    return result


def _np_grid_to_unpadded_seq(grid: np.ndarray):
    # PAD: 0, <eos>: 1, digits: 2 ... 11
    canvas_h, canvas_w = grid.shape
    if canvas_h < ARCMaxGridSize:
        canvas_h += 1
    if canvas_w < ARCMaxGridSize:
        canvas_w += 1

    nrow, ncol = grid.shape
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas[:nrow, :ncol] = grid + 2

    if nrow < canvas_h:
        canvas[nrow, :ncol] = 1
    if ncol < canvas_w:
        canvas[:nrow, ncol] = 1

    return canvas.flatten(), (canvas_h, canvas_w)


def np_grids_to_unpadded_seq(inp: np.ndarray, out: np.ndarray):
    inp_seq, inp_shape = _np_grid_to_unpadded_seq(inp)
    out_seq, out_shape = _np_grid_to_unpadded_seq(out)
    return [inp_seq, out_seq], (inp_shape, out_shape)


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


def _color_cell(token: int) -> str:
    if token == 0:
        return "\033[48;5;235m  \033[0m"
    if token == 1:
        return "\033[48;5;15;30mEE\033[0m"
    if token in _ARC_TOKEN_BG_COLORS:
        return f"\033[48;5;{_ARC_TOKEN_BG_COLORS[token]}m  \033[0m"
    return "\033[48;5;15;31m??\033[0m"


def _format_color_grid_lines(grid: np.ndarray, max_rows: int, max_cols: int):
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
                cells.append(_color_cell(int(grid[row_idx, col_idx])))
        lines.append(f"r{row_idx:02d} |{''.join(cells)}")

    return lines


def _visible_len(text: str) -> int:
    visible = 0
    in_escape = False
    for char in text:
        if char == "\033":
            in_escape = True
        elif in_escape and char == "m":
            in_escape = False
        elif not in_escape:
            visible += 1
    return visible


def print_data(
    data: np.ndarray,
    seq_shape: Tuple[int, int],
    title: str = "",
    max_rows: int = 31,
    max_cols: int = 31,
):
    if data.ndim != 1:
        raise ValueError(f"Expected 1D token sequence, got shape={data.shape}")

    height, width = int(seq_shape[0]), int(seq_shape[1])
    expected_size = height * width
    if data.shape[0] != expected_size:
        raise ValueError(
            f"Token length and seq_shape must match, got {data.shape[0]} and {seq_shape}"
        )

    if data.size == 0:
        print(f"{title or 'sample'}: <empty>")
        return

    grid = data.reshape(height, width)
    cropped = grid.shape[0] > max_rows or grid.shape[1] > max_cols
    crop_suffix = " (cropped)" if cropped else ""

    print(f"{title or 'sample'}: canvas={height}x{width}{crop_suffix}")
    print("legend: dark gray = blank/pad, EE = eos, colored blocks = ARC colors")
    for line in _format_color_grid_lines(grid, max_rows=max_rows, max_cols=max_cols):
        print(line)
    print()


def _extract_pair_grid(data: np.ndarray, position_ids: np.ndarray, pair_idx: int, io_id: int) -> np.ndarray:
    part_mask = (position_ids[:, 0] == pair_idx) & (position_ids[:, 1] == io_id)
    rows = position_ids[part_mask, 2]
    cols = position_ids[part_mask, 3]
    tokens = data[part_mask]

    if tokens.size == 0:
        return np.empty((0, 0), dtype=np.int32)

    height = int(rows.max()) + 1
    width = int(cols.max()) + 1
    expected_size = height * width
    if tokens.size != expected_size:
        raise ValueError(f"Failed to reconstruct pair {pair_idx}: expected {expected_size} tokens, got {tokens.size}")

    return tokens.reshape(height, width)


def _print_color_data(
    data: np.ndarray,
    position_ids: np.ndarray,
    title: str,
    max_rows: int = 21,
    max_cols: int = 21,
) -> None:
    if data.ndim != 1:
        raise ValueError(f"Expected 1D token sequence, got shape={data.shape}")
    if position_ids.ndim != 2 or position_ids.shape[1] != 4:
        raise ValueError(f"Expected position_ids with shape (N, 4), got shape={position_ids.shape}")
    if data.shape[0] != position_ids.shape[0]:
        raise ValueError(
            f"Token length and position_ids length must match, got {data.shape[0]} and {position_ids.shape[0]}"
        )
    if data.size == 0:
        print(f"{title}: <empty>")
        return

    num_pairs = int(position_ids[:, 0].max()) + 1
    print(f"{title}: tokens={data.shape[0]}, pairs={num_pairs}")
    print("legend: dark gray = blank/pad, EE = eos, colored blocks = ARC colors")

    for pair_idx in range(num_pairs):
        input_grid = _extract_pair_grid(data, position_ids, pair_idx, io_id=0)
        output_grid = _extract_pair_grid(data, position_ids, pair_idx, io_id=1)

        input_lines = _format_color_grid_lines(input_grid, max_rows=max_rows, max_cols=max_cols)
        output_lines = _format_color_grid_lines(output_grid, max_rows=max_rows, max_cols=max_cols)
        left_width = max(_visible_len(line) for line in input_lines)
        pair_shape = (
            f"input={input_grid.shape[0]}x{input_grid.shape[1]} "
            f"output={output_grid.shape[0]}x{output_grid.shape[1]}"
        )
        cropped = (
            input_grid.shape[0] > max_rows
            or input_grid.shape[1] > max_cols
            or output_grid.shape[0] > max_rows
            or output_grid.shape[1] > max_cols
        )
        crop_suffix = " (cropped)" if cropped else ""

        print(f"  Pair {pair_idx} | canvas={pair_shape}{crop_suffix}")
        print(f"    {'input':<{left_width}}    output")
        for line_idx in range(max(len(input_lines), len(output_lines))):
            left = input_lines[line_idx] if line_idx < len(input_lines) else ""
            right = output_lines[line_idx] if line_idx < len(output_lines) else ""
            padding = " " * (left_width - _visible_len(left))
            print(f"    {left}{padding}    {right}")
        print()


def encode_arc_example_pair(
    inp: np.ndarray,
    out: np.ndarray,
    *,
    no_padding: bool,
    do_translation: bool,
):
    if no_padding:
        return np_grids_to_unpadded_seq(inp, out)

    return np_grid_to_fixed_seq_translational_augment(
        inp,
        out,
        do_translation=do_translation,
    ), None


def encode_context_examples(
    examples: Optional[List[Tuple[np.ndarray, np.ndarray]]],
    *,
    no_padding: bool,
):
    examples = examples or []
    shape_dims = (2, 2) if no_padding else (2,)
    example_shapes = np.empty((len(examples), *shape_dims), dtype=np.int32)

    if no_padding:
        encoded_examples = np.empty((len(examples), 2), dtype=object)
        for example_idx, (inp, out) in enumerate(examples):
            (encoded_inp, encoded_out), seq_shape = encode_arc_example_pair(
                inp,
                out,
                no_padding=True,
                do_translation=False,
            )
            encoded_examples[example_idx, 0] = encoded_inp.astype(np.uint8, copy=False)
            encoded_examples[example_idx, 1] = encoded_out.astype(np.uint8, copy=False)
            example_shapes[example_idx] = np.asarray(seq_shape, dtype=np.int32)
        return encoded_examples, example_shapes

    encoded_pairs = []
    for example_idx, (inp, out) in enumerate(examples):
        (encoded_inp, encoded_out), _seq_shape = encode_arc_example_pair(
            inp,
            out,
            no_padding=False,
            do_translation=False,
        )
        encoded_pairs.append(
            np.stack(
                [
                    encoded_inp.astype(np.uint8, copy=False),
                    encoded_out.astype(np.uint8, copy=False),
                ],
                axis=0,
            )
        )
        example_shapes[example_idx] = (ARCMaxGridSize, ARCMaxGridSize)

    if not encoded_pairs:
        return np.empty((0, 2, ARCMaxGridSize * ARCMaxGridSize), dtype=np.uint8), example_shapes

    return np.stack(encoded_pairs, axis=0).astype(np.uint8, copy=False), example_shapes


def grid_hash(grid: np.ndarray):
    assert grid.ndim == 2
    assert grid.dtype == np.uint8

    buffer = [x.to_bytes(1, "big") for x in grid.shape]
    buffer.append(grid.tobytes())
    
    return hashlib.sha256(b"".join(buffer)).hexdigest()


def puzzle_hash(puzzle: dict):
    # Hash the puzzle for checking equivalence
    hashes = []
    for example_type, example in puzzle.items():
        for input, label in example.examples:
            hashes.append(f"{grid_hash(input)}|{grid_hash(label)}")
            
    hashes.sort()
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def load_arc_gen_puzzles(arc_gen_dir: str):
    arc_gen_puzzles = {}
    total_examples = 0
    removed_examples = 0

    for file_name in tqdm(
        sorted(os.listdir(arc_gen_dir)),
        desc="Loading arc-gen puzzles",
    ):
        if not file_name.endswith(".json"):
            continue

        puzzle_id = os.path.splitext(file_name)[0]
        with open(os.path.join(arc_gen_dir, file_name), "r") as f:
            examples = json.load(f)

        assert isinstance(examples, list), f"{file_name} must contain a list of examples"

        normalized_examples = []
        for idx, example in enumerate(examples):
            assert isinstance(example, dict), f"{file_name}[{idx}] must be an object"
            assert "input" in example and "output" in example, (
                f"{file_name}[{idx}] must contain both 'input' and 'output'"
            )
            normalized_examples.append(
                {
                    "input": example["input"],
                    "output": example["output"],
                }
            )

        normalized_examples, removed = filter_arc_examples_by_size(normalized_examples)

        arc_gen_puzzles[puzzle_id] = normalized_examples
        total_examples += len(normalized_examples)
        removed_examples += removed

    print(
        f"Loaded {len(arc_gen_puzzles)} arc-gen puzzles with "
        f"{total_examples} generated train examples from {arc_gen_dir}"
    )
    if removed_examples > 0:
        print(
            f"Filtered out {removed_examples} arc-gen examples with size >= "
            f"{ARCMaxGridSize}x{ARCMaxGridSize}"
        )
    return arc_gen_puzzles


def aug(name: str):
    # Augment plan
    trans_id = np.random.randint(0, 8)
    mapping = np.concatenate([np.arange(0, 1, dtype=np.uint8), np.random.permutation(np.arange(1, 10, dtype=np.uint8))])  # Permute colors, Excluding "0" (black)
    
    name_with_aug_repr = f"{name}{PuzzleIdSeparator}t{trans_id}{PuzzleIdSeparator}{''.join(str(x) for x in mapping)}"

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


def build_augmented_puzzle_family(
    base_name: str,
    converted: Dict[Tuple[str, str], ARCPuzzle],
    aug_count: int,
):
    if not converted:
        return {}

    family_by_dest = {dest: [converted_puzzle] for dest, converted_puzzle in converted.items()}
    target_family_size = max(1, aug_count)
    if target_family_size <= 1:
        return family_by_dest

    hashes = {puzzle_hash(converted)}

    for _trial in range(ARCAugmentRetriesFactor * target_family_size):
        aug_name, map_grid = aug(base_name)

        augmented = {
            dest: ARCPuzzle(
                aug_name,
                [(map_grid(input), map_grid(label)) for (input, label) in puzzle.examples],
                (
                    [(map_grid(input), map_grid(label)) for (input, label) in puzzle.context_examples]
                    if puzzle.context_examples is not None
                    else None
                ),
            )
            for dest, puzzle in converted.items()
        }
        h = puzzle_hash(augmented)
        if h not in hashes:
            hashes.add(h)
            for dest, augmented_puzzle in augmented.items():
                family_by_dest.setdefault(dest, []).append(augmented_puzzle)

        if all(len(group) >= target_family_size for group in family_by_dest.values()):
            break

    return family_by_dest


def convert_single_arc_puzzle(
    results: dict,
    name: str,
    puzzle: dict,
    aug_count: int,
    dest_mapping: Dict[str, Tuple[str, str]],
    arc_gen_examples: Optional[List[dict]] = None,
    arc_gen_aug_count: Optional[int] = None,
):
    # Convert examples already present in the source ARC puzzle.
    dests = set(dest_mapping.values())
    converted = {dest: ARCPuzzle(name, []) for dest in dests}
    source_train_examples = [
        (arc_grid_to_np(example["input"]), arc_grid_to_np(example["output"]))
        for example in puzzle.get("train", [])
    ]
    for example_type, examples in puzzle.items():
        if len(examples) == 0:
            continue
        # Map to target split
        dest = dest_mapping[example_type]
        converted[dest].examples.extend([(arc_grid_to_np(example["input"]), arc_grid_to_np(example["output"])) for example in examples])
        if dest[0] == "test":
            converted[dest].context_examples = list(source_train_examples)

    # ARC-GEN examples belong to the same task id as the source ARC-AGI puzzle.
    # Keep them in the same augmentation puzzle instead of creating additional
    # puzzle families; this preserves group and puzzle counts and increases only
    # the number of examples per puzzle.
    added_arc_gen_examples = 0
    train_dest = dest_mapping.get("train")
    if arc_gen_examples and train_dest is not None:
        converted.setdefault(train_dest, ARCPuzzle(name, []))
        converted[train_dest].examples.extend(
            [
                (arc_grid_to_np(example["input"]), arc_grid_to_np(example["output"]))
                for example in arc_gen_examples
            ]
        )
        added_arc_gen_examples = len(arc_gen_examples)

    converted = {dest: converted_puzzle for dest, converted_puzzle in converted.items() if converted_puzzle.examples}
    groups_by_dest = build_augmented_puzzle_family(name, converted, aug_count)

    if not groups_by_dest:
        return set(), 0, 0, {}

    # Append
    for dest, group in groups_by_dest.items():
        # Convert the examples
        dest_split, dest_set = dest

        results.setdefault(dest_split, {})
        results[dest_split].setdefault(dest_set, [])
        results[dest_split][dest_set].append(group)

    return (
        set(groups_by_dest.keys()),
        added_arc_gen_examples,
        0,
        {dest: len(group) for dest, group in groups_by_dest.items()},
    )


def load_puzzles_arcagi(config: DataProcessConfig):
    train_examples_dest = ("train", "all")
    test_examples_map = {
        config.test_set_name: [(1.0, ("test", "all"))],
        "_default": [(1.0, ("train", "all"))]
    }
    
    test_puzzles = {}
    results = {}

    total_source_groups = 0
    output_group_counts = {}
    output_puzzle_counts = {}

    arc_gen_puzzles = {}
    if "training" in config.subsets:
        if not config.include_arc_gen:
            print("ARC-GEN integration disabled by config, skipping")
        elif config.arc_gen_dir:
            if os.path.isdir(config.arc_gen_dir):
                arc_gen_puzzles = load_arc_gen_puzzles(config.arc_gen_dir)
            else:
                print(f"arc-gen directory not found at {config.arc_gen_dir}, skipping")
        else:
            print("ARC-GEN integration enabled but no arc_gen_dir was provided, skipping")

    for subset_name in config.subsets:
        # Load all puzzles in this subset
        with open(f"{config.input_file_prefix}_{subset_name}-challenges.json", "r") as f:
            puzzles = json.load(f)
            print (f"Loaded {len(puzzles)} puzzles from {subset_name} challenges")

        sols_filename = f"{config.input_file_prefix}_{subset_name}-solutions.json"
        if os.path.isfile(sols_filename):
            with open(sols_filename, "r") as f:
                sols = json.load(f)
                print (f"Loaded {len(sols)} solutions from {subset_name} solutions")
                
                for puzzle_id in puzzles.keys():
                    for idx, sol_grid in enumerate(sols[puzzle_id]):
                        puzzles[puzzle_id]["test"][idx]["output"] = sol_grid
        else:
            # Fill with dummy
            print (f"{subset_name} solutions not found, filling with dummy")

            for puzzle_id, puzzle in puzzles.items():
                for example in puzzle["test"]:
                    example.setdefault("output", [[0]])

        arc_gen_enabled_for_subset = subset_name == "training" and bool(arc_gen_puzzles)
        if arc_gen_enabled_for_subset:
            matched_arc_gen_groups = sum(
                1 for puzzle_id in puzzles.keys() if arc_gen_puzzles.get(puzzle_id)
            )
            matched_arc_gen_seed_puzzles = sum(
                len(arc_gen_puzzles[puzzle_id]) for puzzle_id in puzzles.keys() if arc_gen_puzzles.get(puzzle_id)
            )
            missing_arc_gen_groups = sum(
                1 for puzzle_id in arc_gen_puzzles.keys() if puzzle_id not in puzzles
            )
            print(
                f"ARC-GEN matched {matched_arc_gen_groups} training groups "
                f"with {matched_arc_gen_seed_puzzles} seed puzzles before augmentation "
                f"({missing_arc_gen_groups} unmatched puzzles)"
            )
            if config.num_aug_gen is not None and config.num_aug_gen != config.num_aug:
                print(
                    "num_aug_gen is ignored when merging ARC-GEN examples into ARC-AGI "
                    "augmentation puzzles; using num_aug for puzzle count"
                )

        puzzles = filter_puzzles_by_size(
            puzzles,
            source_name=subset_name,
            extra_train_examples_by_puzzle=arc_gen_puzzles if arc_gen_enabled_for_subset else None,
        )

        # Shuffle puzzles
        puzzles = list(puzzles.items())
        print (f"Shuffling {len(puzzles)} puzzles...")
        np.random.shuffle(puzzles)

        added_arc_gen_groups = 0
        added_arc_gen_examples = 0
        
        # Assign by fraction
        for idx, (name, puzzle) in tqdm(
            enumerate(puzzles),
            total=len(puzzles),
            desc=f"Converting {subset_name}",
        ):
            fraction = idx / len(puzzles)
            test_examples_dest = None
            for f, dest in test_examples_map.get(subset_name, test_examples_map["_default"]):
                if fraction < f:
                    test_examples_dest = dest
                    break
                    
            assert test_examples_dest is not None
            
            if test_examples_dest[0] == "test" and len(puzzle.get("test", [])) > 0:
                test_puzzles[name] = puzzle

            (
                used_dests,
                added_arc_gen_example_count,
                _added_arc_gen_total_puzzle_count,
                puzzle_counts_by_dest,
            ) = convert_single_arc_puzzle(
                results,
                name,
                puzzle,
                config.num_aug,
                {"train": train_examples_dest, "test": test_examples_dest},
                arc_gen_examples=arc_gen_puzzles.get(name) if arc_gen_enabled_for_subset else None,
                arc_gen_aug_count=(
                    config.num_aug_gen
                    if config.num_aug_gen is not None
                    else config.num_aug
                ),
            )
            if used_dests:
                total_source_groups += 1
            for dest, num_puzzles in puzzle_counts_by_dest.items():
                output_group_counts[dest] = output_group_counts.get(dest, 0) + 1
                output_puzzle_counts[dest] = output_puzzle_counts.get(dest, 0) + num_puzzles
            if added_arc_gen_example_count > 0:
                added_arc_gen_groups += 1
                added_arc_gen_examples += added_arc_gen_example_count

        if arc_gen_enabled_for_subset:
            print(
                f"Merged {added_arc_gen_examples} ARC-GEN examples into "
                f"{added_arc_gen_groups} training groups in subset '{subset_name}'"
            )

    print (f"Total source task groups: {total_source_groups}")
    print (f"Source task groups with held-out test examples: {len(test_puzzles)}")
    print (f"Source task groups routed only to train: {total_source_groups - len(test_puzzles)}")
    for dest in sorted(output_group_counts):
        print(
            f"Output {dest[0]}/{dest[1]}: "
            f"groups={output_group_counts[dest]}, puzzles={output_puzzle_counts[dest]}"
        )
    print("results keys:", results.keys())
    return results, test_puzzles


def convert_dataset(config: DataProcessConfig):
    np.random.seed(config.seed)
    
    # Read dataset
    data, test_puzzles = load_puzzles_arcagi(config)
    
    # Map global puzzle identifiers
    num_identifiers = 1  # 0 is blank
    identifier_map = {}
    print("Mapping puzzle IDs...")
    for split_name, split in data.items():
        print("split: ", split_name)
        print("subset_len: ", len(split))
        for subset_name, subset in split.items():
            subset_group_count = len(subset)
            subset_puzzle_count = sum(len(group) for group in subset)
            print(" subset: ", subset_name)
            print("  num_groups: ", subset_group_count)
            print("  num_puzzles: ", subset_puzzle_count)
            for group in tqdm(
                subset,
                desc=f"Mapping IDs {split_name}/{subset_name}",
                leave=False,
            ):
                for puzzle in group:
                    if puzzle.id not in identifier_map:
                        identifier_map[puzzle.id] = num_identifiers
                        num_identifiers += 1

    print (f"Total puzzle IDs (including <blank>): {num_identifiers}")

    # Save
    for split_name, split in data.items():
        print("split: ", split_name)
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)
        
        # Translational augmentations
        enable_translational_augment = split_name == "train"

        # Statistics
        total_examples = 0
        total_puzzles = 0
        total_groups = 0
        
        for subset_name, subset in split.items():
            # Construct subset
            results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
            if split_name == "test":
                results["examples"] = []
                results["example_shapes"] = []
            results["puzzle_indices"].append(0)
            results["group_indices"].append(0)
            
            example_id = 0
            puzzle_id = 0
            
            for group in tqdm(subset, desc=f"Processing {split_name}/{subset_name}"):
                for puzzle in group:
                    # Push puzzle
                    no_aug_id = np.random.randint(0, len(puzzle.examples))
                    for _idx_ex, (inp, out) in enumerate(puzzle.examples):
                        if config.no_padding:
                            (inp, out), pair_seq_shape = encode_arc_example_pair(
                                inp,
                                out,
                                no_padding=True,
                                do_translation=False,
                            )
                            input_seq_shape, label_seq_shape = pair_seq_shape
                            results.setdefault("seq_shapes", []).append(input_seq_shape)
                            results.setdefault("label_seq_shapes", []).append(label_seq_shape)
                        else:
                            (inp, out), _seq_shape = encode_arc_example_pair(
                                inp,
                                out,
                                no_padding=False,
                                do_translation=enable_translational_augment and _idx_ex != no_aug_id,
                            )
                            
                        results["inputs"].append(inp)
                        results["labels"].append(out)
                        if split_name == "test":
                            context_examples, context_shapes = encode_context_examples(
                                puzzle.context_examples,
                                no_padding=config.no_padding,
                            )
                            results["examples"].append(context_examples)
                            results["example_shapes"].append(context_shapes)
                        example_id += 1
                        
                        total_examples += 1

                    results["puzzle_indices"].append(example_id)
                    results["puzzle_identifiers"].append(identifier_map[puzzle.id])
                    
                    puzzle_id += 1
                    
                    total_puzzles += 1
                    
                # Push group
                results["group_indices"].append(puzzle_id)
                total_groups += 1
                
            for key, value in results.items():
                print(f"    {key}: {len(value)} items")
                for i in range(min(3, len(value))):
                    print(f"      {key}[{i}]: shape={value[i].shape if isinstance(value[i], np.ndarray) else 'list'}, dtype={value[i].dtype if isinstance(value[i], np.ndarray) else 'N/A'}")
                    # print(f"        Value: {value[i]}")
            # print(results["group_indices"])
            
            target_train_id = np.random.randint(0, len(results["inputs"]))
            target_test_id = np.random.randint(0, len(results["inputs"]))
            
            for k, v in results.items():
                if config.no_padding and k in {"inputs", "labels"}:
                    if v:
                        target_id = target_test_id if split_name == "test" else target_train_id
                        shape_key = "seq_shapes" if k == "inputs" else "label_seq_shapes"
                        print_data(
                            v[target_id],
                            results[shape_key][target_id],
                            title=f"{split_name}/{subset_name} {k}[{target_id}]",
                        )
                        if split_name == "test" and k == "inputs" and "examples" in results:
                            context_examples = results["examples"][target_id]
                            context_shapes = results["example_shapes"][target_id]
                            for example_idx in range(len(context_examples)):
                                print_data(
                                    np.asarray(context_examples[example_idx][0], dtype=np.uint8),
                                    tuple(int(v) for v in context_shapes[example_idx][0]),
                                    title=(
                                        f"{split_name}/{subset_name} examples[{target_id}]"
                                        f"[{example_idx}].input"
                                    ),
                                )
                                print_data(
                                    np.asarray(context_examples[example_idx][1], dtype=np.uint8),
                                    tuple(int(v) for v in context_shapes[example_idx][1]),
                                    title=(
                                        f"{split_name}/{subset_name} examples[{target_id}]"
                                        f"[{example_idx}].output"
                                    ),
                                )
                    seq_lengths = np.array([seq.shape[0] for seq in v], dtype=np.int64)
                    seq_offsets = np.concatenate(
                        [np.array([0], dtype=np.int64), np.cumsum(seq_lengths, dtype=np.int64)]
                    )
                    flat_tokens = np.concatenate(v).astype(np.uint8, copy=False)
                    np.save(os.path.join(config.output_dir, split_name, f"{subset_name}__{k}.npy"), flat_tokens)
                    if k == "inputs":
                        np.save(
                            os.path.join(config.output_dir, split_name, f"{subset_name}__seq_offsets.npy"),
                            seq_offsets,
                        )
                    else:
                        np.save(
                            os.path.join(config.output_dir, split_name, f"{subset_name}__label_seq_offsets.npy"),
                            seq_offsets,
                        )
                elif k in {"inputs", "labels"}:
                    np.save(os.path.join(config.output_dir, split_name, f"{subset_name}__{k}.npy"), np.stack(v, 0))
                elif k == "seq_shapes":
                    np.save(
                        os.path.join(config.output_dir, split_name, f"{subset_name}__{k}.npy"),
                        np.array(v, dtype=np.int32),
                    )
                elif k == "label_seq_shapes":
                    np.save(
                        os.path.join(config.output_dir, split_name, f"{subset_name}__{k}.npy"),
                        np.array(v, dtype=np.int32),
                    )
                elif k == "examples":
                    examples_array = np.empty((len(v),), dtype=object)
                    for example_idx, context_examples in enumerate(v):
                        examples_array[example_idx] = context_examples
                    np.save(
                        os.path.join(config.output_dir, split_name, f"{subset_name}__{k}.npy"),
                        examples_array,
                    )
                elif k == "example_shapes":
                    example_shapes_array = np.empty((len(v),), dtype=object)
                    for example_idx, context_shapes in enumerate(v):
                        example_shapes_array[example_idx] = context_shapes
                    np.save(
                        os.path.join(config.output_dir, split_name, f"{subset_name}__{k}.npy"),
                        example_shapes_array,
                    )
                else:
                    np.save(
                        os.path.join(config.output_dir, split_name, f"{subset_name}__{k}.npy"),
                        np.array(v, dtype=np.int32),
                    )
        
        # Metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=ARCMaxGridSize * ARCMaxGridSize,
            vocab_size=10 + 2,  # PAD + EOS + "0" ... "9"
            
            pad_id=0,
            ignore_label_id=0,
            
            blank_identifier_id=0,
            num_puzzle_identifiers=num_identifiers,
            
            total_groups=total_groups,
            mean_puzzle_examples=total_examples / total_puzzles,
            sets=list(split.keys()),
            variable_seq_lengths=config.no_padding,
        )
        print(f"  Total puzzles: {total_puzzles}")
        print(f"  Total examples: {total_examples}")
        print(f"  Total groups: {total_groups}")
        print(f"  Mean examples per puzzle: {metadata.mean_puzzle_examples:.2f}")

        # Save metadata as JSON.
        with open(os.path.join(config.output_dir, split_name, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)
            print(f"  Saved metadata to {f.name}")
            
    # Save IDs mapping
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        ids_mapping = {v: k for k, v in identifier_map.items()}
        
        json.dump([ids_mapping.get(i, "<blank>") for i in range(num_identifiers)], f)
    
    # Save Test Puzzles
    with open(os.path.join(config.output_dir, "test_puzzles.json"), "w") as f:
        json.dump(test_puzzles, f)


@cli.command(singleton=True)
def main(config: DataProcessConfig):
    convert_dataset(config)


if __name__ == "__main__":
    cli()
