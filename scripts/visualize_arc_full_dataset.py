#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


ARC_TOKEN_PAD = 0
ARC_TOKEN_EOS = 1
ARC_TOKEN_COLOR_OFFSET = 2
ARC_NUM_COLORS = 10

ARC_COLORS = [
    "#000000",
    "#0074D9",
    "#FF4136",
    "#2ECC40",
    "#FFDC00",
    "#AAAAAA",
    "#F012BE",
    "#FF851B",
    "#7FDBFF",
    "#870C25",
]
ARC_CMAP = ListedColormap(ARC_COLORS)
ARC_NORM = BoundaryNorm(np.arange(-0.5, ARC_NUM_COLORS + 0.5, 1.0), ARC_CMAP.N)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize ARC full-context datasets stored in the variable-length no-padding format."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Dataset directory such as data/arc1concept-full-aug-1000-nopadding-13_2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <dataset_dir>/visualizations.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to visualize.",
    )
    parser.add_argument(
        "--representative-count",
        type=int,
        default=4,
        help="How many evenly spaced representative samples to render per split.",
    )
    return parser.parse_args()


def load_split(dataset_dir: Path, split: str) -> dict[str, Any]:
    split_dir = dataset_dir / split
    metadata = json.loads((split_dir / "dataset.json").read_text())

    offsets = np.load(split_dir / "all__seq_offsets.npy")
    shapes = np.load(split_dir / "all__seq_shapes.npy")
    inputs = np.load(split_dir / "all__inputs.npy", mmap_mode="r")
    labels = np.load(split_dir / "all__labels.npy", mmap_mode="r")
    position_ids_path = split_dir / "all__position_ids.npy"
    position_ids = np.load(position_ids_path, mmap_mode="r") if position_ids_path.exists() else None

    lengths = np.diff(offsets).astype(np.int64, copy=False)

    return {
        "split": split,
        "split_dir": split_dir,
        "metadata": metadata,
        "sequence_layout": metadata.get("sequence_layout") or "sample",
        "offsets": offsets,
        "shapes": shapes,
        "inputs": inputs,
        "labels": labels,
        "position_ids": position_ids,
        "lengths": lengths,
    }


def choose_sample_indices(num_samples: int, count: int) -> list[int]:
    if num_samples <= 0 or count <= 0:
        return []
    if num_samples <= count:
        return list(range(num_samples))
    return sorted(set(np.linspace(0, num_samples - 1, num=count, dtype=int).tolist()))


def decode_arc_canvas(canvas: np.ndarray) -> np.ndarray:
    canvas = np.asarray(canvas)
    if canvas.ndim != 2:
        raise ValueError(f"Expected a 2D canvas, got shape {canvas.shape}")

    nonzero_rows = np.flatnonzero(np.any(canvas != ARC_TOKEN_PAD, axis=1))
    nonzero_cols = np.flatnonzero(np.any(canvas != ARC_TOKEN_PAD, axis=0))
    if nonzero_rows.size == 0 or nonzero_cols.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    active_h = int(nonzero_rows[-1]) + 1
    active_w = int(nonzero_cols[-1]) + 1
    active = canvas[:active_h, :active_w]

    # EOS は「その行/列に実データ色がなく、境界だけが 1 で埋まる」形で置かれる。
    eos_rows = np.flatnonzero(np.any(active == ARC_TOKEN_EOS, axis=1) & ~np.any(active >= ARC_TOKEN_COLOR_OFFSET, axis=1))
    eos_cols = np.flatnonzero(np.any(active == ARC_TOKEN_EOS, axis=0) & ~np.any(active >= ARC_TOKEN_COLOR_OFFSET, axis=0))

    grid_h = int(eos_rows[0]) if eos_rows.size else active_h
    grid_w = int(eos_cols[0]) if eos_cols.size else active_w

    cropped = active[:grid_h, :grid_w]
    if cropped.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    if np.any(cropped < ARC_TOKEN_COLOR_OFFSET):
        raise ValueError("Encountered PAD/EOS tokens inside the cropped ARC grid.")

    decoded = cropped.astype(np.int16, copy=False) - ARC_TOKEN_COLOR_OFFSET
    if np.any(decoded < 0) or np.any(decoded >= ARC_NUM_COLORS):
        raise ValueError("Decoded ARC colors fell outside the expected [0, 9] range.")
    return decoded.astype(np.uint8, copy=False)


def decode_arc_canvas_without_eos(canvas: np.ndarray) -> np.ndarray:
    canvas = np.asarray(canvas)
    if canvas.ndim != 2:
        raise ValueError(f"Expected a 2D canvas, got shape {canvas.shape}")

    nonzero_rows = np.flatnonzero(np.any(canvas != ARC_TOKEN_PAD, axis=1))
    nonzero_cols = np.flatnonzero(np.any(canvas != ARC_TOKEN_PAD, axis=0))
    if nonzero_rows.size == 0 or nonzero_cols.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    active_h = int(nonzero_rows[-1]) + 1
    active_w = int(nonzero_cols[-1]) + 1
    cropped = canvas[:active_h, :active_w]
    if np.any(cropped < ARC_TOKEN_COLOR_OFFSET):
        raise ValueError("Encountered PAD tokens inside a cropped ARC canvas without EOS.")

    decoded = cropped.astype(np.int16, copy=False) - ARC_TOKEN_COLOR_OFFSET
    if np.any(decoded < 0) or np.any(decoded >= ARC_NUM_COLORS):
        raise ValueError("Decoded ARC colors fell outside the expected [0, 9] range.")
    return decoded.astype(np.uint8, copy=False)


def decode_canvas_for_layout(canvas: np.ndarray, sequence_layout: str) -> np.ndarray:
    if sequence_layout == "pair_no_eos":
        return decode_arc_canvas_without_eos(canvas)
    return decode_arc_canvas(canvas)


def _reconstruct_canvas_from_positions(tokens: np.ndarray, positions: np.ndarray) -> np.ndarray:
    if tokens.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    rows = positions[:, 2].astype(np.int64, copy=False)
    cols = positions[:, 3].astype(np.int64, copy=False)
    canvas = np.zeros((int(rows.max()) + 1, int(cols.max()) + 1), dtype=np.uint8)
    canvas[rows, cols] = tokens.astype(np.uint8, copy=False)
    return canvas


def load_sample(split_data: dict[str, Any], sample_index: int) -> tuple[list[dict[str, Any]], tuple[int, ...], int]:
    offsets = split_data["offsets"]
    shapes = split_data["shapes"]
    start = int(offsets[sample_index])
    end = int(offsets[sample_index + 1])
    sample_shape = tuple(int(x) for x in shapes[sample_index].tolist())
    sequence_layout = split_data["sequence_layout"]

    if sequence_layout in {"fixed", "sample"}:
        inputs = np.asarray(split_data["inputs"][start:end]).reshape(sample_shape)
        labels = np.asarray(split_data["labels"][start:end]).reshape(sample_shape)
        pair_entries = []
        for pair_index in range(inputs.shape[0]):
            pair_entries.append(
                {
                    "pair_index": pair_index,
                    "input_canvas": inputs[pair_index, 0],
                    "output_canvas": inputs[pair_index, 1],
                    "label_canvas": labels[pair_index, 1],
                }
            )
        return pair_entries, sample_shape, end - start

    if split_data["position_ids"] is None:
        raise ValueError(f"sequence_layout={sequence_layout} requires explicit position_ids.")

    input_tokens = np.asarray(split_data["inputs"][start:end])
    label_tokens = np.asarray(split_data["labels"][start:end])
    position_ids = np.asarray(split_data["position_ids"][start:end])

    pair_entries = []
    for pair_index in sorted(np.unique(position_ids[:, 0]).tolist()):
        input_mask = (position_ids[:, 0] == pair_index) & (position_ids[:, 1] == 0)
        output_mask = (position_ids[:, 0] == pair_index) & (position_ids[:, 1] == 1)
        pair_entries.append(
            {
                "pair_index": int(pair_index),
                "input_canvas": _reconstruct_canvas_from_positions(input_tokens[input_mask], position_ids[input_mask]),
                "output_canvas": _reconstruct_canvas_from_positions(input_tokens[output_mask], position_ids[output_mask]),
                "label_canvas": _reconstruct_canvas_from_positions(label_tokens[output_mask], position_ids[output_mask]),
            }
        )
    return pair_entries, sample_shape, end - start


def render_arc_grid(ax: Any, grid: np.ndarray, title: str, highlight: bool = False) -> None:
    ax.imshow(grid, cmap=ARC_CMAP, norm=ARC_NORM, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#222222", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_aspect("equal")

    border_color = "#D7263D" if highlight else "#777777"
    border_width = 2.4 if highlight else 1.0
    for spine in ax.spines.values():
        spine.set_color(border_color)
        spine.set_linewidth(border_width)


def render_sample(split_data: dict[str, Any], sample_index: int, output_path: Path, heading: str) -> dict[str, Any]:
    pair_entries, sample_shape, packed_length = load_sample(split_data, sample_index)
    sequence_layout = split_data["sequence_layout"]

    fig, axes = plt.subplots(
        nrows=len(pair_entries),
        ncols=2,
        figsize=(7.0, max(2.8, 2.8 * len(pair_entries))),
        squeeze=False,
        constrained_layout=True,
    )

    target_pair_index: int | None = None

    for row_index, pair_entry in enumerate(pair_entries):
        pair_index = int(pair_entry["pair_index"])
        label_canvas = pair_entry["label_canvas"]
        is_target = bool(np.any(label_canvas != ARC_TOKEN_PAD))
        if is_target:
            target_pair_index = pair_index

        input_grid = decode_canvas_for_layout(pair_entry["input_canvas"], sequence_layout)
        output_source = label_canvas if is_target else pair_entry["output_canvas"]
        output_grid = decode_canvas_for_layout(output_source, sequence_layout)

        role = "Target" if is_target else "Context"
        input_title = f"{role} {pair_index + 1} Input\n{input_grid.shape[0]}x{input_grid.shape[1]}"
        output_title = f"{role} {pair_index + 1} Output\n{output_grid.shape[0]}x{output_grid.shape[1]}"

        render_arc_grid(axes[row_index, 0], input_grid, input_title, highlight=is_target)
        render_arc_grid(axes[row_index, 1], output_grid, output_title, highlight=is_target)

    split = split_data["split"]
    title = (
        f"{heading}\n"
        f"split={split} sample_index={sample_index} layout={sequence_layout} seq_shape_hint={sample_shape} seq_len={packed_length}"
    )
    if target_pair_index is not None:
        title += f" target_pair={target_pair_index + 1}"
    fig.suptitle(title, fontsize=12)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "sample_index": sample_index,
        "seq_len": packed_length,
        "shape": list(sample_shape),
        "file": output_path.name,
    }


def render_length_histogram(split_data_map: dict[str, dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = {"train": "#1f77b4", "test": "#ff7f0e"}

    for split, split_data in split_data_map.items():
        lengths = split_data["lengths"]
        if lengths.size == 0:
            continue
        color = colors.get(split, None)
        ax.hist(
            lengths,
            bins=80,
            histtype="step",
            linewidth=2.0,
            color=color,
            label=f"{split} (n={len(lengths):,}, max={int(lengths.max()):,})",
        )

    ax.set_title("Packed Sequence Length Distribution")
    ax.set_xlabel("seq_len")
    ax.set_ylabel("sample count")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_split_summary(split_data: dict[str, Any]) -> dict[str, Any]:
    lengths = split_data["lengths"]
    shapes = split_data["shapes"]
    max_idx = int(np.argmax(lengths))
    pair_slot_counts, pair_slot_freqs = np.unique(shapes[:, 0], return_counts=True)

    return {
        "num_samples": int(lengths.size),
        "min_seq_len": int(lengths.min()),
        "max_seq_len": int(lengths.max()),
        "mean_seq_len": float(lengths.mean()),
        "median_seq_len": float(np.median(lengths)),
        "max_seq_len_sample_index": max_idx,
        "max_seq_shape": [int(x) for x in shapes[max_idx].tolist()],
        "pair_slot_distribution": {
            str(int(slot_count)): int(freq)
            for slot_count, freq in zip(pair_slot_counts.tolist(), pair_slot_freqs.tolist(), strict=True)
        },
        "metadata": split_data["metadata"],
    }


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else dataset_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_data_map: dict[str, dict[str, Any]] = {}
    for split in args.splits:
        split_dir = dataset_dir / split
        if split_dir.is_dir():
            split_data_map[split] = load_split(dataset_dir, split)

    if not split_data_map:
        raise FileNotFoundError(f"No requested splits were found under {dataset_dir}")

    render_length_histogram(split_data_map, output_dir / "sequence_lengths.png")

    summary: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "summary": {},
        "samples": {},
        "longest_samples": {},
    }

    overall_longest: tuple[str, int, int] | None = None

    for split, split_data in split_data_map.items():
        lengths = split_data["lengths"]
        split_summary = build_split_summary(split_data)
        summary["summary"][split] = split_summary

        sample_entries: list[dict[str, Any]] = []
        sample_indices = choose_sample_indices(len(lengths), args.representative_count)
        for sample_index in sample_indices:
            output_path = output_dir / f"{split}_sample_{sample_index:06d}.png"
            sample_entries.append(
                render_sample(
                    split_data=split_data,
                    sample_index=sample_index,
                    output_path=output_path,
                    heading="Representative Sample",
                )
            )
        summary["samples"][split] = sample_entries

        longest_index = split_summary["max_seq_len_sample_index"]
        longest_output = output_dir / f"{split}_longest_sample_{longest_index:06d}.png"
        longest_entry = render_sample(
            split_data=split_data,
            sample_index=longest_index,
            output_path=longest_output,
            heading="Longest seq_len Sample",
        )
        summary["longest_samples"][split] = longest_entry

        longest_len = int(lengths[longest_index])
        if overall_longest is None or longest_len > overall_longest[2]:
            overall_longest = (split, longest_index, longest_len)

    if overall_longest is not None:
        split, sample_index, seq_len = overall_longest
        summary["overall_longest"] = {
            "split": split,
            "sample_index": sample_index,
            "seq_len": seq_len,
            "file": summary["longest_samples"][split]["file"],
        }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved visualizations to {output_dir}")
    if "overall_longest" in summary:
        overall = summary["overall_longest"]
        print(
            "Overall longest sample:",
            f"split={overall['split']}",
            f"sample_index={overall['sample_index']}",
            f"seq_len={overall['seq_len']}",
        )


if __name__ == "__main__":
    main()
