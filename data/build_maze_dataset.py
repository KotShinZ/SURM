from typing import Optional
from collections import deque
import math
import os
import csv
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from data.common import PuzzleDatasetMetadata, dihedral_transform


CHARSET = "# SGo"


cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/maze-30x30-hard-1k"
    output_dir: str = "data/maze-30x30-hard-1k"

    subsample_size: Optional[int] = None
    aug: bool = False
    num_aug: int = 0
    rebuild: bool = False


def add_random_walls(inp: np.ndarray, out: np.ndarray, max_added_walls: int = 10):
    aug_inp = inp.copy()
    aug_out = out.copy()

    # Only add walls to open cells that are not part of the solution path.
    candidate_mask = (aug_inp == ord(" ")) & (aug_out == ord(" "))
    candidates = np.argwhere(candidate_mask)

    if len(candidates) == 0 or max_added_walls <= 0:
        return aug_inp, aug_out

    num_walls = np.random.randint(1, min(max_added_walls, len(candidates)) + 1)
    chosen = candidates[np.random.choice(len(candidates), size=num_walls, replace=False)]

    aug_inp[chosen[:, 0], chosen[:, 1]] = ord("#")
    aug_out[chosen[:, 0], chosen[:, 1]] = ord("#")
    return aug_inp, aug_out


def swap_start_goal(inp: np.ndarray, out: np.ndarray):
    aug_inp = inp.copy()
    aug_out = out.copy()

    for arr in (aug_inp, aug_out):
        start_mask = arr == ord("S")
        goal_mask = arr == ord("G")
        arr[start_mask] = ord("G")
        arr[goal_mask] = ord("S")

    return aug_inp, aug_out


def solve_maze_bfs(grid: np.ndarray):
    start_pos = np.argwhere(grid == ord("S"))
    goal_pos = np.argwhere(grid == ord("G"))

    if len(start_pos) != 1 or len(goal_pos) != 1:
        return None

    start = tuple(start_pos[0])
    goal = tuple(goal_pos[0])
    queue = deque([start])
    parents = {start: None}

    while queue:
        row, col = queue.popleft()
        if (row, col) == goal:
            break

        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n_row = row + d_row
            n_col = col + d_col

            if not (0 <= n_row < grid.shape[0] and 0 <= n_col < grid.shape[1]):
                continue
            if grid[n_row, n_col] == ord("#"):
                continue

            nxt = (n_row, n_col)
            if nxt in parents:
                continue

            parents[nxt] = (row, col)
            queue.append(nxt)

    if goal not in parents:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parents[cur]

    path.reverse()
    return path


def render_maze_label(inp: np.ndarray, path):
    out = inp.copy()

    for row, col in path[1:-1]:
        out[row, col] = ord("o")

    return out


def relocate_start_goal(grid: np.ndarray):
    open_cells = np.argwhere(grid == ord(" "))
    if len(open_cells) < 2:
        return None

    selected = open_cells[np.random.choice(len(open_cells), size=2, replace=False)]
    rebuilt = grid.copy()
    rebuilt[selected[0, 0], selected[0, 1]] = ord("S")
    rebuilt[selected[1, 0], selected[1, 1]] = ord("G")
    return rebuilt


def balance_random_walls(grid: np.ndarray, max_wall_changes: int = 10):
    rebuilt = grid.copy()

    wall_cells = np.argwhere(rebuilt == ord("#"))
    open_cells = np.argwhere(rebuilt == ord(" "))
    max_changes = min(max_wall_changes, len(wall_cells), len(open_cells))

    if max_changes <= 0:
        return rebuilt

    num_changes = np.random.randint(0, max_changes + 1)
    if num_changes == 0:
        return rebuilt

    removed_walls = wall_cells[np.random.choice(len(wall_cells), size=num_changes, replace=False)]
    rebuilt[removed_walls[:, 0], removed_walls[:, 1]] = ord(" ")

    add_candidate_mask = rebuilt == ord(" ")
    add_candidate_mask[removed_walls[:, 0], removed_walls[:, 1]] = False
    add_candidates = np.argwhere(add_candidate_mask)
    if len(add_candidates) < num_changes:
        return grid.copy()

    added_walls = add_candidates[np.random.choice(len(add_candidates), size=num_changes, replace=False)]
    rebuilt[added_walls[:, 0], added_walls[:, 1]] = ord("#")
    return rebuilt


def rebuild_maze(inp: np.ndarray, max_wall_changes: int = 10, max_retries: int = 24):
    for _ in range(max_retries):
        rebuilt = dihedral_transform(inp, np.random.randint(0, 8)).copy()
        rebuilt[(rebuilt == ord("S")) | (rebuilt == ord("G"))] = ord(" ")
        rebuilt = balance_random_walls(rebuilt, max_wall_changes=max_wall_changes)

        rebuilt = relocate_start_goal(rebuilt)
        if rebuilt is None:
            continue

        path = solve_maze_bfs(rebuilt)
        if path is None:
            continue

        label = render_maze_label(rebuilt, path)
        if not np.array_equal(rebuilt, inp):
            return rebuilt, label

    return None


def augment_maze(inp: np.ndarray, out: np.ndarray):
    aug_inp = inp.copy()
    aug_out = out.copy()

    for _ in range(8):
        trans_id = np.random.randint(0, 8)
        aug_inp = dihedral_transform(inp, trans_id)
        aug_out = dihedral_transform(out, trans_id)

        wall_budget = np.random.randint(0, 11)
        aug_inp, aug_out = add_random_walls(aug_inp, aug_out, max_added_walls=wall_budget)

        if np.random.rand() < 0.5:
            aug_inp, aug_out = swap_start_goal(aug_inp, aug_out)

        if not (np.array_equal(aug_inp, inp) and np.array_equal(aug_out, out)):
            return aug_inp, aug_out

    return aug_inp, aug_out


def convert_subset(set_name: str, config: DataProcessConfig):
    # Read CSV
    all_chars = set()
    grid_size = None
    inputs = []
    labels = []
    
    with open(hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset"), newline="") as csvfile:  # type: ignore
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for source, q, a, rating in reader:
            all_chars.update(q)
            all_chars.update(a)

            if grid_size is None:
                n = int(len(q) ** 0.5)
                grid_size = (n, n)
                
            inputs.append(np.frombuffer(q.encode(), dtype=np.uint8).reshape(grid_size))
            labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(grid_size))

    # If subsample_size is specified for the training set,
    # randomly sample the desired number of examples.
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)
        if config.subsample_size < total_samples:
            indices = np.random.choice(total_samples, size=config.subsample_size, replace=False)
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    # Generate dataset
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    num_augments = 0
    use_legacy_dihedral_aug = set_name == "train" and config.aug and config.num_aug == 0 and config.rebuild == False
    if set_name == "train":
        num_augments = config.num_aug if config.num_aug > 0 else (7 if use_legacy_dihedral_aug else 0)

    print(f"Number of augmentations per puzzle: {num_augments}")
    print(f"Using legacy dihedral augmentation: {use_legacy_dihedral_aug}")
    print(f"Using rebuild-based augmentation: {config.rebuild and not use_legacy_dihedral_aug}")

    for orig_inp, orig_out in zip(tqdm(inputs), labels):
        for aug_idx in range(1 + num_augments):
            if aug_idx == 0 and config.rebuild == False:
                inp, out = orig_inp, orig_out
            elif use_legacy_dihedral_aug:
                inp = dihedral_transform(orig_inp, aug_idx)
                out = dihedral_transform(orig_out, aug_idx)
            else:
                rebuilt = rebuild_maze(orig_inp) if config.rebuild else None
                if rebuilt is not None:
                    inp, out = rebuilt
                else:
                    # print("Rebuild failed, falling back to augmentation.")
                    inp, out = augment_maze(orig_inp, orig_out)

            results["inputs"].append(inp)
            results["labels"].append(out)
            example_id += 1
            puzzle_id += 1
            
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)
            
        # Push group
        results["group_indices"].append(puzzle_id)
            
    # Char mappings
    assert len(all_chars - set(CHARSET)) == 0
    
    char2id = np.zeros(256, np.uint8)
    char2id[np.array(list(map(ord, CHARSET)))] = np.arange(len(CHARSET)) + 1

    # To Numpy
    def _seq_to_numpy(seq):
        arr = np.vstack([char2id[s.reshape(-1)] for s in seq])
        
        return arr
    
    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=int(math.prod(grid_size)),  # type: ignore
        vocab_size=len(CHARSET) + 1,  # PAD + Charset
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"]
    )

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
        
    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
        
    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
