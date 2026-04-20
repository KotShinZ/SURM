from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle


ARC_LIMIT = 30


def grid_size(grid: list[list[int]]) -> tuple[int, int]:
    height = len(grid)
    width = max((len(row) for row in grid), default=0)
    return height, width


def build_matrix(counter: Counter[tuple[int, int]], max_height: int, max_width: int) -> np.ndarray:
    matrix = np.zeros((max_height, max_width), dtype=np.int32)
    for (height, width), count in counter.items():
        if height == 0 or width == 0:
            continue
        matrix[height - 1, width - 1] = count
    return matrix


def draw_heatmap(ax: plt.Axes, matrix: np.ndarray, title: str, vmax: int) -> None:
    masked = np.ma.masked_where(matrix == 0, matrix)
    image = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        norm=LogNorm(vmin=1, vmax=max(vmax, 1)),
    )

    ax.set_title(title)
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")

    max_height, max_width = matrix.shape
    tick_candidates = sorted({1, 5, 10, 20, 30, 40, 50, 60, 70, max_width, max_height})
    x_ticks = [tick - 1 for tick in tick_candidates if 1 <= tick <= max_width]
    y_ticks = [tick - 1 for tick in tick_candidates if 1 <= tick <= max_height]
    ax.set_xticks(x_ticks, [tick + 1 for tick in x_ticks], rotation=45, ha="right")
    ax.set_yticks(y_ticks, [tick + 1 for tick in y_ticks])

    if max_width >= ARC_LIMIT and max_height >= ARC_LIMIT:
        boundary = Rectangle(
            (-0.5, -0.5),
            ARC_LIMIT,
            ARC_LIMIT,
            fill=False,
            edgecolor="white",
            linewidth=1.5,
            linestyle="--",
        )
        ax.add_patch(boundary)

    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Count")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot width/height distribution heatmaps for data/arc-gen.")
    parser.add_argument("--arc-gen-dir", type=Path, default=Path("data/arc-gen"))
    parser.add_argument("--output", type=Path, default=Path("data/arc-gen-size-heatmap.png"))
    args = parser.parse_args()

    input_counter: Counter[tuple[int, int]] = Counter()
    output_counter: Counter[tuple[int, int]] = Counter()
    combined_counter: Counter[tuple[int, int]] = Counter()

    file_count = 0
    example_count = 0
    max_height = 0
    max_width = 0
    over_limit_input = 0
    over_limit_output = 0

    for path in sorted(args.arc_gen_dir.glob("*.json")):
        file_count += 1
        with path.open("r") as f:
            examples = json.load(f)

        if not isinstance(examples, list):
            raise ValueError(f"{path} must contain a list of examples")

        for idx, example in enumerate(examples):
            if not isinstance(example, dict):
                raise ValueError(f"{path}[{idx}] must be an object")

            example_count += 1

            input_size = grid_size(example["input"])
            output_size = grid_size(example["output"])

            input_counter[input_size] += 1
            output_counter[output_size] += 1
            combined_counter[input_size] += 1
            combined_counter[output_size] += 1

            max_height = max(max_height, input_size[0], output_size[0])
            max_width = max(max_width, input_size[1], output_size[1])

            if input_size[0] > ARC_LIMIT or input_size[1] > ARC_LIMIT:
                over_limit_input += 1
            if output_size[0] > ARC_LIMIT or output_size[1] > ARC_LIMIT:
                over_limit_output += 1

    input_matrix = build_matrix(input_counter, max_height, max_width)
    output_matrix = build_matrix(output_counter, max_height, max_width)
    combined_matrix = build_matrix(combined_counter, max_height, max_width)
    vmax = int(max(input_matrix.max(), output_matrix.max(), combined_matrix.max()))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    draw_heatmap(
        axes[0],
        input_matrix,
        f"Input grids\ncount={int(input_matrix.sum())}, >30={over_limit_input}",
        vmax,
    )
    draw_heatmap(
        axes[1],
        output_matrix,
        f"Output grids\ncount={int(output_matrix.sum())}, >30={over_limit_output}",
        vmax,
    )
    draw_heatmap(
        axes[2],
        combined_matrix,
        f"Combined grids\ncount={int(combined_matrix.sum())}, >30={over_limit_input + over_limit_output}",
        vmax,
    )

    fig.suptitle(
        f"ARC-Gen Grid Size Distribution ({file_count} files, {example_count} examples)",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved heatmap to {args.output}")
    print(f"Files: {file_count}")
    print(f"Examples: {example_count}")
    print(f"Max size: {max_height}x{max_width}")
    print(f"Input grids over 30x30: {over_limit_input}")
    print(f"Output grids over 30x30: {over_limit_output}")


if __name__ == "__main__":
    main()
