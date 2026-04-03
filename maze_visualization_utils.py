from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from evaluate_trained_model import (
    _hidden_diff_norm,
    _index_structure,
    _power_of_two_loop_checkpoints,
    _resolve_checkpoint_path,
    _supports_hidden_pruning,
    _valid_example_mask,
    create_model_for_evaluation,
    create_test_dataloader,
    evaluate_model,
    load_config_from_checkpoint_path,
    load_model_weights,
)


COMMAND = (
    "python evaluate_trained_model.py "
    "--checkpoint checkpoints/URM-maze "
    "--max_problems 4096 "
    "--loops 32 "
    "--batch_size 1024 "
    "--hidden_diff_threshold 0.1"
)

MAZE_TOKEN_NAMES = {
    0: "PAD",
    1: "#",
    2: "space",
    3: "S",
    4: "G",
    5: "o",
}

MAZE_TOKEN_COLORS = {
    0: "#f4f4f4",
    1: "#202124",
    2: "#ffffff",
    3: "#2f6fed",
    4: "#fbbc04",
    5: "#34a853",
}

PATH_ERROR_NAMES = {
    0: "background",
    1: "correct path",
    2: "missed path",
    3: "extra path",
}

PATH_ERROR_COLORS = {
    0: "#f5f5f5",
    1: "#2e7d32",
    2: "#d32f2f",
    3: "#ef6c00",
}

IGNORE_LABEL_ID = -100

MAZE_CMAP = ListedColormap([MAZE_TOKEN_COLORS[idx] for idx in sorted(MAZE_TOKEN_COLORS)])
MAZE_NORM = BoundaryNorm(np.arange(-0.5, len(MAZE_TOKEN_COLORS) + 0.5, 1.0), MAZE_CMAP.N)

PATH_ERROR_CMAP = ListedColormap([PATH_ERROR_COLORS[idx] for idx in sorted(PATH_ERROR_COLORS)])
PATH_ERROR_NORM = BoundaryNorm(np.arange(-0.5, len(PATH_ERROR_COLORS) + 0.5, 1.0), PATH_ERROR_CMAP.N)


def resolve_repo_root(start: Optional[Path] = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "evaluate_trained_model.py").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not find the repository root from the current working directory.")


def ensure_repo_root_on_syspath(repo_root: Path) -> None:
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def default_paths(repo_root: Path) -> Dict[str, Path]:
    checkpoint_dir = repo_root / "checkpoints" / "URM-maze"
    metrics_path = (
        checkpoint_dir
        / "step_39060_evaluation_test_max_problems_4096_loops_32_hidden_diff_threshold_0_1.json"
    )
    detailed_path = metrics_path.with_name(f"{metrics_path.stem}_detailed.pt")
    return {
        "checkpoint_dir": checkpoint_dir,
        "metrics_path": metrics_path,
        "detailed_path": detailed_path,
    }


def set_plot_style() -> None:
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.18


def load_payload(path: os.PathLike[str] | str) -> Dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return torch.load(path, map_location="cpu")


def _loop_sort_key(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return value


def _to_numpy(value: torch.Tensor | np.ndarray | Sequence[Any]) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _reshape_maze(flat_tokens: np.ndarray) -> np.ndarray:
    side = int(round(math.sqrt(flat_tokens.size)))
    if side * side != flat_tokens.size:
        raise ValueError(f"Expected a square maze, got flattened length {flat_tokens.size}.")
    return flat_tokens.reshape(side, side)


def format_metrics_table(metrics_payload: Mapping[str, Any]) -> str:
    rows = []
    header = f"{'set':<10} {'count':>8} {'token_acc':>12} {'exact_acc':>12} {'lm_loss':>12} {'avg_steps':>12}"
    rows.append(header)
    rows.append("-" * len(header))
    for set_name, values in metrics_payload["metrics"].items():
        rows.append(
            f"{set_name:<10} "
            f"{int(values.get('count', 0)):>8d} "
            f"{values.get('accuracy', float('nan')):>12.6f} "
            f"{values.get('exact_accuracy', float('nan')):>12.6f} "
            f"{values.get('lm_loss', float('nan')):>12.6f} "
            f"{values.get('steps', float('nan')):>12.6f}"
        )
    return "\n".join(rows)


def build_loop_summary(metrics_payload: Mapping[str, Any], set_name: str = "overall") -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for loop, per_set in sorted(
        metrics_payload.get("power_of_two_loop_metrics", {}).items(),
        key=lambda item: _loop_sort_key(item[0]),
    ):
        if set_name not in per_set:
            continue
        row = {"loop": float(loop)}
        row.update({key: float(value) for key, value in per_set[set_name].items()})
        rows.append(row)
    return rows


def collect_final_steps(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    max_problems: Optional[int],
    hidden_diff_threshold: Optional[float],
) -> tuple[Dict[str, torch.Tensor], Dict[int, torch.Tensor]]:
    model.eval()

    hidden_pruning_enabled = (
        hidden_diff_threshold is not None
        and hidden_diff_threshold > 0
        and _supports_hidden_pruning(model)
    )

    final_steps_by_set: Dict[str, List[torch.Tensor]] = defaultdict(list)
    hidden_diffs_by_step: Dict[int, List[float]] = defaultdict(list)
    processed = 0

    with torch.inference_mode():
        for set_name, batch, _global_batch_size in dataloader:
            if max_problems is not None and processed >= max_problems:
                break

            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            valid_indices = torch.nonzero(_valid_example_mask(batch["labels"]), as_tuple=False).squeeze(-1)
            if valid_indices.numel() == 0:
                continue

            if max_problems is not None:
                remaining = max_problems - processed
                if remaining <= 0:
                    break
                valid_indices = valid_indices[:remaining]

            batch = {key: value.index_select(0, valid_indices) for key, value in batch.items()}
            batch_size = batch["inputs"].shape[0]
            processed += batch_size

            with torch.device(str(device)):
                carry = model.initial_carry(batch)

            active_batch = batch
            active_indices = torch.arange(batch_size, device=device)
            final_steps = torch.empty((batch_size,), dtype=torch.int64, device=device)

            while True:
                carry_before_step = None
                if hidden_pruning_enabled and active_indices.numel() > 0 and hasattr(carry, "current_hidden"):
                    carry_before_step = carry

                carry, _loss, _metrics, _preds, all_finish = model(
                    carry=carry,
                    batch=active_batch,
                    return_keys={"preds"},
                )

                if hidden_pruning_enabled and carry_before_step is not None and not all_finish:
                    hidden_diff_norm = _hidden_diff_norm(
                        carry.current_hidden.detach(),
                        carry_before_step.current_hidden.detach(),
                    )

                    step_values = carry.steps.detach().to(torch.int64).cpu().tolist()
                    diff_values = hidden_diff_norm.detach().to(torch.float32).cpu().tolist()
                    for step_value, diff_value in zip(step_values, diff_values):
                        hidden_diffs_by_step[int(step_value)].append(float(diff_value))

                    pruned_mask = hidden_diff_norm <= hidden_diff_threshold
                    if torch.any(pruned_mask):
                        final_steps[active_indices[pruned_mask]] = carry.steps[pruned_mask].to(torch.int64)

                        keep_indices = torch.nonzero(~pruned_mask, as_tuple=False).squeeze(-1)
                        if keep_indices.numel() == 0:
                            active_indices = active_indices[:0]
                            break

                        active_indices = active_indices[keep_indices]
                        active_batch = {
                            key: value.index_select(0, keep_indices)
                            for key, value in active_batch.items()
                        }
                        carry = _index_structure(carry, keep_indices)

                if all_finish:
                    if active_indices.numel() > 0:
                        final_steps[active_indices] = carry.steps.to(torch.int64)
                    break

            final_steps_by_set[set_name].append(final_steps.cpu())

    finalized_steps = {
        set_name: torch.cat(chunks, dim=0) if chunks else torch.empty(0, dtype=torch.int64)
        for set_name, chunks in final_steps_by_set.items()
    }
    finalized_hidden_diffs = {
        step: torch.tensor(values, dtype=torch.float32)
        for step, values in sorted(hidden_diffs_by_step.items())
    }
    return finalized_steps, finalized_hidden_diffs


def run_detailed_evaluation(
    checkpoint_dir: os.PathLike[str] | str,
    split: str = "test",
    batch_size: int = 1024,
    max_problems: Optional[int] = 4096,
    loops: Optional[int] = 32,
    hidden_diff_threshold: Optional[float] = 0.1,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = load_config_from_checkpoint_path(str(checkpoint_dir))
    config.global_batch_size = batch_size
    if config.arch.__pydantic_extra__ is None:
        config.arch.__pydantic_extra__ = {}
    if loops is not None:
        config.arch.__pydantic_extra__["loops"] = loops

    resolved_checkpoint_path = _resolve_checkpoint_path(str(checkpoint_dir))
    if resolved_checkpoint_path is None:
        raise FileNotFoundError(f"Could not resolve a checkpoint under {checkpoint_dir}.")

    dataloader = create_test_dataloader(config, split, config.global_batch_size)
    metadata = dataloader.dataset.metadata

    model = create_model_for_evaluation(config, metadata, device=device_obj)
    checkpoint_step = load_model_weights(
        model,
        resolved_checkpoint_path,
        device=device_obj,
        strict=True,
    )

    loop_checkpoints = _power_of_two_loop_checkpoints(loops)
    metrics_by_set, loop_metrics_by_step, saved_outputs, processed_batches, processed_problems = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device_obj,
        max_problems=max_problems,
        max_batches=None,
        save_predictions=True,
        hidden_diff_threshold=hidden_diff_threshold,
        loop_checkpoints=loop_checkpoints,
    )

    filtered_outputs = {}
    for set_name, per_set_outputs in saved_outputs.items():
        filtered_outputs[set_name] = {
            key: value
            for key, value in per_set_outputs.items()
            if key in {"inputs", "labels", "preds"}
        }

    final_steps, hidden_diff_by_step = collect_final_steps(
        model=model,
        dataloader=create_test_dataloader(config, split, config.global_batch_size),
        device=device_obj,
        max_problems=max_problems,
        hidden_diff_threshold=hidden_diff_threshold,
    )

    return {
        "command": COMMAND,
        "checkpoint": resolved_checkpoint_path,
        "checkpoint_step": checkpoint_step,
        "data_path": config.data_path,
        "split": split,
        "batch_size": config.global_batch_size,
        "loops": loops,
        "hidden_diff_threshold": hidden_diff_threshold,
        "processed_batches": processed_batches,
        "processed_problems": processed_problems,
        "metrics": metrics_by_set,
        "power_of_two_loop_metrics": loop_metrics_by_step,
        "outputs": filtered_outputs,
        "final_steps": final_steps,
        "hidden_diff_by_step": hidden_diff_by_step,
    }


def load_or_run_detailed_evaluation(
    checkpoint_dir: os.PathLike[str] | str,
    detailed_cache_path: os.PathLike[str] | str,
    split: str = "test",
    batch_size: int = 1024,
    max_problems: Optional[int] = 4096,
    loops: Optional[int] = 32,
    hidden_diff_threshold: Optional[float] = 0.1,
    device: Optional[str] = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    detailed_cache_path = Path(detailed_cache_path)
    if detailed_cache_path.exists() and not force_rerun:
        return torch.load(detailed_cache_path, map_location="cpu")

    detailed = run_detailed_evaluation(
        checkpoint_dir=checkpoint_dir,
        split=split,
        batch_size=batch_size,
        max_problems=max_problems,
        loops=loops,
        hidden_diff_threshold=hidden_diff_threshold,
        device=device,
    )

    detailed_cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(detailed, detailed_cache_path)
    return detailed


def compute_example_stats(detailed_payload: Mapping[str, Any], set_name: str = "all") -> Dict[str, np.ndarray]:
    outputs = detailed_payload["outputs"][set_name]

    inputs = _to_numpy(outputs["inputs"]).astype(np.int16, copy=False)
    labels = _to_numpy(outputs["labels"]).astype(np.int16, copy=False)
    preds = _to_numpy(outputs["preds"]).astype(np.int16, copy=False)
    steps = _to_numpy(detailed_payload["final_steps"][set_name]).astype(np.int16, copy=False)

    valid_mask = labels != IGNORE_LABEL_ID
    correct_mask = (preds == labels) | (~valid_mask)
    exact = correct_mask.all(axis=1)

    token_accuracy = np.divide(
        ((preds == labels) & valid_mask).sum(axis=1),
        np.maximum(valid_mask.sum(axis=1), 1),
        dtype=np.float64,
    )

    true_path = (labels == 5) & valid_mask
    pred_path = preds == 5
    path_intersection = (true_path & pred_path).sum(axis=1)
    true_path_count = true_path.sum(axis=1)
    pred_path_count = pred_path.sum(axis=1)

    path_precision = np.divide(
        path_intersection,
        np.maximum(pred_path_count, 1),
        dtype=np.float64,
    )
    path_recall = np.divide(
        path_intersection,
        np.maximum(true_path_count, 1),
        dtype=np.float64,
    )
    path_f1 = np.divide(
        2.0 * path_precision * path_recall,
        np.maximum(path_precision + path_recall, 1e-8),
        dtype=np.float64,
    )

    path_exact = (true_path == pred_path).all(axis=1)
    missed_path = (true_path & ~pred_path).sum(axis=1).astype(np.int16, copy=False)
    extra_path = (~true_path & pred_path).sum(axis=1).astype(np.int16, copy=False)
    path_length = true_path_count.astype(np.int16, copy=False)

    return {
        "inputs": inputs,
        "labels": labels,
        "preds": preds,
        "steps": steps,
        "valid_mask": valid_mask,
        "exact": exact,
        "token_accuracy": token_accuracy,
        "true_path": true_path,
        "pred_path": pred_path,
        "path_precision": path_precision,
        "path_recall": path_recall,
        "path_f1": path_f1,
        "path_exact": path_exact,
        "missed_path": missed_path,
        "extra_path": extra_path,
        "path_length": path_length,
    }


def hidden_diff_summary(detailed_payload: Mapping[str, Any]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    threshold = float(detailed_payload.get("hidden_diff_threshold") or 0.0)

    for step, values in sorted(detailed_payload.get("hidden_diff_by_step", {}).items()):
        values_np = _to_numpy(values).astype(np.float64, copy=False)
        if values_np.size == 0:
            continue
        rows.append(
            {
                "step": float(step),
                "count": float(values_np.size),
                "mean": float(values_np.mean()),
                "median": float(np.median(values_np)),
                "q90": float(np.quantile(values_np, 0.9)),
                "fraction_below_threshold": float(np.mean(values_np <= threshold)),
            }
        )
    return rows


def build_analysis_lines(
    metrics_payload: Mapping[str, Any],
    stats: Mapping[str, np.ndarray],
    set_name: str = "overall",
) -> List[str]:
    metrics = metrics_payload["metrics"][set_name]
    loop_rows = build_loop_summary(metrics_payload, set_name=set_name)

    exact = _to_numpy(stats["exact"]).astype(bool)
    path_exact = _to_numpy(stats["path_exact"]).astype(bool)
    path_f1 = _to_numpy(stats["path_f1"]).astype(np.float64)
    steps = _to_numpy(stats["steps"]).astype(np.int64)
    path_length = _to_numpy(stats["path_length"]).astype(np.int64)
    missed_path = _to_numpy(stats["missed_path"]).astype(np.int64)
    extra_path = _to_numpy(stats["extra_path"]).astype(np.int64)

    step_counter = Counter(int(step) for step in steps.tolist())
    total_examples = max(len(steps), 1)
    step_distribution = ", ".join(
        f"{step}: {count} ({count / total_examples:.1%})"
        for step, count in sorted(step_counter.items())
    )

    lines = [
        f"Processed {int(metrics['count'])} mazes from the `{metrics_payload['split']}` split.",
        (
            f"Overall exact accuracy is {metrics['exact_accuracy']:.1%}, while token accuracy is "
            f"{metrics['accuracy']:.2%}. Because walls and empty space dominate the grid, the path-only "
            f"view is more revealing: path exact accuracy is {path_exact.mean():.1%} and mean path F1 is "
            f"{path_f1.mean():.3f}."
        ),
    ]

    if loop_rows:
        loop_by_value = {int(row["loop"]): row for row in loop_rows}
        first_row = loop_rows[0]
        best_row = max(loop_rows, key=lambda row: row["exact_accuracy"])
        final_row = loop_rows[-1]
        lines.append(
            (
                f"Loop scaling saturates early: exact accuracy goes from "
                f"{first_row['exact_accuracy']:.1%} at {int(first_row['loop'])} loops to "
                f"{best_row['exact_accuracy']:.1%} at {int(best_row['loop'])} loops, and stays at "
                f"{final_row['exact_accuracy']:.1%} by {int(final_row['loop'])} loops."
            )
        )

    lines.append(
        f"Final step distribution is concentrated very early: {step_distribution}."
    )

    unique_steps = sorted(step_counter)
    step_quality = []
    for step in unique_steps:
        mask = steps == step
        if not np.any(mask):
            continue
        step_quality.append(f"{step}: {exact[mask].mean():.1%} exact")
    lines.append(
        "Faster exits are usually cleaner: " + ", ".join(step_quality) + "."
    )

    unsolved_mask = ~exact
    if np.any(unsolved_mask):
        lines.append(
            (
                f"Unsolved mazes miss {missed_path[unsolved_mask].mean():.1f} true path cells on average and "
                f"hallucinate {extra_path[unsolved_mask].mean():.1f} extra path cells. Their mean path length "
                f"({path_length[unsolved_mask].mean():.1f}) is close to solved mazes "
                f"({path_length[exact].mean():.1f}), so path length alone does not explain the failures."
            )
        )

    return lines


def plot_loop_metrics(metrics_payload: Mapping[str, Any], set_name: str = "overall"):
    rows = build_loop_summary(metrics_payload, set_name=set_name)
    if not rows:
        raise ValueError("No power-of-two loop metrics were found in the payload.")

    loops = np.array([row["loop"] for row in rows], dtype=np.float64)
    exact_accuracy = np.array([row["exact_accuracy"] for row in rows], dtype=np.float64)
    token_accuracy = np.array([row["accuracy"] for row in rows], dtype=np.float64)
    avg_steps = np.array([row["steps"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plots = [
        (exact_accuracy, "Exact Accuracy", "accuracy"),
        (token_accuracy, "Token Accuracy", "accuracy"),
        (avg_steps, "Average Final Step", "step"),
    ]

    for ax, (values, title, ylabel) in zip(axes, plots):
        ax.plot(loops, values, marker="o", linewidth=2.0, color="#1a73e8")
        ax.set_title(title)
        ax.set_xlabel("loop budget")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_xticks(loops)
        ax.set_xticklabels([str(int(loop)) for loop in loops])

    fig.tight_layout()
    return fig, axes


def plot_runtime_diagnostics(stats: Mapping[str, np.ndarray], detailed_payload: Mapping[str, Any]):
    steps = _to_numpy(stats["steps"]).astype(np.int64)
    exact = _to_numpy(stats["exact"]).astype(bool)

    step_values, step_counts = np.unique(steps, return_counts=True)
    step_solve_rates = np.array(
        [exact[steps == step].mean() for step in step_values],
        dtype=np.float64,
    )

    hidden_rows = hidden_diff_summary(detailed_payload)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.bar(step_values, step_counts, color="#4c8bf5", width=0.7)
    ax.set_title("Final Step Distribution")
    ax.set_xlabel("final step")
    ax.set_ylabel("count")
    ax.set_xticks(step_values)

    twin = ax.twinx()
    twin.plot(step_values, step_solve_rates, color="#d93025", marker="o", linewidth=2.0)
    twin.set_ylabel("exact accuracy")
    twin.set_ylim(0.0, 1.05)
    twin.grid(False)

    ax = axes[1]
    if hidden_rows:
        hidden_steps = np.array([row["step"] for row in hidden_rows], dtype=np.float64)
        below = np.array([row["fraction_below_threshold"] for row in hidden_rows], dtype=np.float64)
        median = np.array([row["median"] for row in hidden_rows], dtype=np.float64)
        threshold = float(detailed_payload.get("hidden_diff_threshold") or 0.0)

        ax.bar(hidden_steps, below, color="#34a853", width=0.7)
        ax.set_title("Hidden-Diff Pruning Signal")
        ax.set_xlabel("step")
        ax.set_ylabel("fraction <= threshold")
        ax.set_xticks(hidden_steps)
        ax.set_ylim(0.0, 1.05)

        twin = ax.twinx()
        twin.plot(hidden_steps, median, color="#f29900", marker="o", linewidth=2.0)
        twin.axhline(threshold, color="#d93025", linestyle="--", linewidth=1.5)
        twin.set_ylabel("median hidden diff")
        twin.grid(False)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Hidden-diff statistics unavailable", ha="center", va="center")

    fig.tight_layout()
    return fig, axes


def plot_path_distributions(stats: Mapping[str, np.ndarray]):
    path_f1 = _to_numpy(stats["path_f1"]).astype(np.float64)
    path_length = _to_numpy(stats["path_length"]).astype(np.float64)
    exact = _to_numpy(stats["exact"]).astype(bool)
    steps = _to_numpy(stats["steps"]).astype(np.float64)
    missed_path = _to_numpy(stats["missed_path"]).astype(np.float64)
    extra_path = _to_numpy(stats["extra_path"]).astype(np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    ax = axes[0]
    ax.hist(path_f1, bins=20, color="#5f6368", edgecolor="white")
    ax.set_title("Path F1 Distribution")
    ax.set_xlabel("path F1")
    ax.set_ylabel("count")

    ax = axes[1]
    solved_lengths = path_length[exact]
    unsolved_lengths = path_length[~exact]
    bins = np.arange(path_length.min(), path_length.max() + 2)
    ax.hist(solved_lengths, bins=bins, alpha=0.75, label="solved", color="#34a853")
    if unsolved_lengths.size:
        ax.hist(unsolved_lengths, bins=bins, alpha=0.75, label="unsolved", color="#d93025")
    ax.set_title("Path Length by Outcome")
    ax.set_xlabel("true path length")
    ax.set_ylabel("count")
    ax.legend()

    ax = axes[2]
    scatter = ax.scatter(
        path_length,
        path_f1,
        c=steps,
        cmap="viridis",
        s=26,
        alpha=0.85,
        linewidths=0.2,
        edgecolors="#202124",
    )
    ax.set_title("Path F1 vs Path Length")
    ax.set_xlabel("true path length")
    ax.set_ylabel("path F1")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("final step")

    fig.tight_layout()

    summary = {
        "solved_missed_mean": float(missed_path[exact].mean()) if np.any(exact) else 0.0,
        "solved_extra_mean": float(extra_path[exact].mean()) if np.any(exact) else 0.0,
        "unsolved_missed_mean": float(missed_path[~exact].mean()) if np.any(~exact) else 0.0,
        "unsolved_extra_mean": float(extra_path[~exact].mean()) if np.any(~exact) else 0.0,
    }
    return fig, axes, summary


def render_token_grid(ax, flat_tokens: np.ndarray, title: str) -> None:
    maze = _reshape_maze(_to_numpy(flat_tokens))
    ax.imshow(maze, cmap=MAZE_CMAP, norm=MAZE_NORM, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def render_path_error_grid(ax, flat_labels: np.ndarray, flat_preds: np.ndarray, title: str) -> None:
    labels = _to_numpy(flat_labels)
    preds = _to_numpy(flat_preds)

    true_path = labels == 5
    pred_path = preds == 5
    error_map = np.zeros_like(labels, dtype=np.int8)
    error_map[true_path & pred_path] = 1
    error_map[true_path & ~pred_path] = 2
    error_map[~true_path & pred_path] = 3

    ax.imshow(_reshape_maze(error_map), cmap=PATH_ERROR_CMAP, norm=PATH_ERROR_NORM, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def plot_legends():
    token_handles = [
        Patch(facecolor=MAZE_TOKEN_COLORS[idx], edgecolor="#202124", label=f"{idx}: {MAZE_TOKEN_NAMES[idx]}")
        for idx in sorted(MAZE_TOKEN_NAMES)
    ]
    error_handles = [
        Patch(facecolor=PATH_ERROR_COLORS[idx], edgecolor="#202124", label=f"{idx}: {PATH_ERROR_NAMES[idx]}")
        for idx in sorted(PATH_ERROR_NAMES)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 2.2))
    axes[0].axis("off")
    axes[0].legend(handles=token_handles, loc="center", ncol=3, frameon=False)
    axes[0].set_title("Maze token colors")

    axes[1].axis("off")
    axes[1].legend(handles=error_handles, loc="center", ncol=2, frameon=False)
    axes[1].set_title("Path error colors")

    fig.tight_layout()
    return fig, axes


def show_examples(
    stats: Mapping[str, np.ndarray],
    indices: Sequence[int],
    title: str,
) -> None:
    indices = list(indices)
    if not indices:
        raise ValueError("Expected at least one example index.")

    inputs = stats["inputs"]
    labels = stats["labels"]
    preds = stats["preds"]
    steps = stats["steps"]
    exact = stats["exact"]
    path_f1 = stats["path_f1"]
    missed_path = stats["missed_path"]
    extra_path = stats["extra_path"]

    fig, axes = plt.subplots(len(indices), 4, figsize=(15, 3.6 * len(indices)), squeeze=False)
    fig.suptitle(title, fontsize=14, y=1.01)

    for row_idx, sample_idx in enumerate(indices):
        input_title = (
            f"#{sample_idx} input\n"
            f"step={int(steps[sample_idx])} | exact={bool(exact[sample_idx])} | path F1={path_f1[sample_idx]:.3f}"
        )
        render_token_grid(axes[row_idx, 0], inputs[sample_idx], input_title)
        render_token_grid(axes[row_idx, 1], labels[sample_idx], "label")
        render_token_grid(axes[row_idx, 2], preds[sample_idx], "prediction")
        render_path_error_grid(
            axes[row_idx, 3],
            labels[sample_idx],
            preds[sample_idx],
            f"path diff\nmissed={int(missed_path[sample_idx])}, extra={int(extra_path[sample_idx])}",
        )

    fig.tight_layout()
    plt.show()
