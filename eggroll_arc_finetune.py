from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pydantic
import torch
import wandb
import yaml
from tqdm import tqdm

from data.common import PuzzleDatasetMetadata
from eggroll_utils import (
    PopulationPerturbations,
    apply_updates_,
    collect_evolvable_parameters,
    extract_parameter_state,
    load_parameter_state_,
    normalize_fitness,
)
from evaluate_trained_model import (
    _act_early_stop_enabled,
    _append_pruned_chunk,
    _compute_batch_metric_sums,
    _index_structure,
    _merge_pruned_predictions,
    _power_of_two_loop_checkpoints,
    _resolve_checkpoint_path,
    _supports_hidden_pruning,
    _unwrap_eval_model,
    _valid_example_mask,
    create_model_for_evaluation,
    create_test_dataloader,
    evaluate_model,
    load_config_from_checkpoint_path,
    load_model_weights,
)
from evaluators.arc_majority_vote import (
    default_arc_submission_dir,
    maybe_create_arc_majority_vote_evaluator,
)
from models.losses import IGNORE_LABEL_ID


FULL_CONFIRM_DEFAULT_MAX_PROBLEMS = 4096
FULL_CONFIRM_DEFAULT_BATCH_SIZE = 2048


class WandbConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    enabled: bool = True
    project: str = "eggroll-urm"
    name: Optional[str] = None
    mode: str = "offline"
    directory: Optional[str] = None
    tags: List[str] = pydantic.Field(default_factory=list)
    resume: bool = True


class ShortEvalConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    enabled: bool = True
    every_generations: int = 5
    max_problems: int = 512
    batch_size: int = 512
    loops: Optional[int] = None
    hidden_diff_threshold: Optional[float] = None


class EggrollFinetuneConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    checkpoint: str
    output_dir: str
    seed: int = 0
    population: int = 32
    rank: int = 4
    sigma: float = 1e-4
    lr: float = 3e-3
    proxy_loops: int = 2
    confirm_loops: int = 256
    hidden_diff_threshold: float = 0.1
    hard_group_pool: int = 128
    es_train_groups: int = 96
    es_val_groups: int = 32
    max_generations: int = 200
    patience: int = 10

    pilot_generations: int = 5
    pilot_sigmas: List[float] = pydantic.Field(default_factory=lambda: [5e-5, 1e-4, 2e-4])
    pilot_lrs: List[float] = pydantic.Field(default_factory=lambda: [1e-3, 3e-3, 1e-2])
    confirm_every_generations: int = 25
    confirm_improvement_delta: float = 0.005
    widen_after_generations: int = 100
    acceptance_exact_accuracy: Optional[float] = None

    wandb: WandbConfig = pydantic.Field(default_factory=WandbConfig)
    short_eval: ShortEvalConfig = pydantic.Field(default_factory=ShortEvalConfig)

    @pydantic.model_validator(mode="after")
    def _validate_ranges(self):
        if self.population <= 0 or self.population % 2 != 0:
            raise ValueError("population must be a positive even integer")
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.hard_group_pool < self.es_train_groups + self.es_val_groups:
            raise ValueError("hard_group_pool must be >= es_train_groups + es_val_groups")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.pilot_generations <= 0:
            raise ValueError("pilot_generations must be positive")
        if self.confirm_every_generations <= 0:
            raise ValueError("confirm_every_generations must be positive")
        if self.confirm_improvement_delta < 0:
            raise ValueError("confirm_improvement_delta must be non-negative")
        if self.widen_after_generations <= 0:
            raise ValueError("widen_after_generations must be positive")
        if self.short_eval.enabled:
            if self.short_eval.every_generations <= 0:
                raise ValueError("short_eval.every_generations must be positive")
            if self.short_eval.max_problems <= 0:
                raise ValueError("short_eval.max_problems must be positive")
            if self.short_eval.batch_size <= 0:
                raise ValueError("short_eval.batch_size must be positive")
        return self


@dataclass
class PreparedBatch:
    batch: Dict[str, torch.Tensor]
    effective_size: int


@dataclass
class GroupScore:
    group_id: int
    failure_rate: float
    exact_accuracy: float
    accuracy: float
    lm_loss: float
    steps: float
    count: int


@dataclass
class BaselineReference:
    path: Path
    exact_accuracy: float
    batch_size: int
    max_problems: int
    loops: int
    hidden_diff_threshold: float
    payload: Dict[str, Any]


def _floatify_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {key: float(value) for key, value in metrics.items()}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=_json_default) + "\n")


def flatten_metrics(prefix: str, payload: Dict[str, Any]) -> Dict[str, float]:
    flattened: Dict[str, float] = {}
    for key, value in payload.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_metrics(full_key, value))
        elif isinstance(value, (int, float, np.integer, np.floating)):
            flattened[full_key] = float(value)
    return flattened


def maybe_log_wandb(run: Optional[wandb.sdk.wandb_run.Run], payload: Dict[str, Any], *, step: Optional[int] = None) -> None:
    if run is None:
        return
    metrics = flatten_metrics("", payload)
    if metrics:
        run.log(metrics, step=step)


def threshold_suffix(hidden_diff_threshold: Optional[float]) -> str:
    if hidden_diff_threshold is None or hidden_diff_threshold <= 0:
        return ""
    threshold_value = f"{hidden_diff_threshold:g}".replace("-", "m").replace(".", "_")
    return f"_hidden_diff_threshold_{threshold_value}"


def set_model_loops(model: torch.nn.Module, loops: int) -> None:
    base_model = _unwrap_eval_model(model)
    config = getattr(base_model, "config", None)
    if config is None:
        raise ValueError("Model does not expose a config object with loop settings.")
    if hasattr(config, "loops"):
        config.loops = int(loops)
    elif hasattr(config, "halt_max_steps"):
        config.halt_max_steps = int(loops)
    else:
        raise ValueError("Model config does not expose loops or halt_max_steps.")

    inner = getattr(base_model, "inner", None)
    if inner is not None and hasattr(inner, "config") and hasattr(inner.config, "loops"):
        inner.config.loops = int(loops)


def get_model_loops(model: torch.nn.Module) -> Optional[int]:
    base_model = _unwrap_eval_model(model)
    config = getattr(base_model, "config", None)
    if config is None:
        return None
    if hasattr(config, "loops"):
        return int(config.loops)
    if hasattr(config, "halt_max_steps"):
        return int(config.halt_max_steps)
    return None


class IndexedSplitData:
    def __init__(self, dataset_root: str, split: str) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        split_root = self.dataset_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Dataset split not found: {split_root}")

        with (split_root / "dataset.json").open("r", encoding="utf-8") as handle:
            self.metadata = PuzzleDatasetMetadata(**json.load(handle))

        if self.metadata.sets != ["all"]:
            raise ValueError(f"Eggroll finetuning currently expects a single 'all' set, found {self.metadata.sets}")

        prefix = split_root / "all__"
        self.inputs = np.load(f"{prefix}inputs.npy", mmap_mode="r")
        self.labels = np.load(f"{prefix}labels.npy", mmap_mode="r")
        self.puzzle_identifiers = np.load(f"{prefix}puzzle_identifiers.npy", mmap_mode=None)
        self.puzzle_indices = np.load(f"{prefix}puzzle_indices.npy", mmap_mode=None)
        self.group_indices = np.load(f"{prefix}group_indices.npy", mmap_mode=None)

    @lru_cache(maxsize=None)
    def example_indices_for_group(self, group_id: int) -> np.ndarray:
        group_start = int(self.group_indices[group_id])
        group_end = int(self.group_indices[group_id + 1])
        chunks: List[np.ndarray] = []
        for puzzle_id in range(group_start, group_end):
            example_start = int(self.puzzle_indices[puzzle_id])
            example_end = int(self.puzzle_indices[puzzle_id + 1])
            if example_end > example_start:
                chunks.append(np.arange(example_start, example_end, dtype=np.int64))
        if not chunks:
            return np.empty((0,), dtype=np.int64)
        return np.concatenate(chunks)

    def build_batches_for_group(
        self,
        group_id: int,
        batch_size: int,
        *,
        pin_memory: bool,
    ) -> List[PreparedBatch]:
        return self.build_batches_from_example_indices(
            self.example_indices_for_group(group_id),
            batch_size,
            pin_memory=pin_memory,
        )

    def build_batches_from_group_ids(
        self,
        group_ids: Sequence[int],
        batch_size: int,
        *,
        pin_memory: bool,
    ) -> List[PreparedBatch]:
        chunks = [self.example_indices_for_group(int(group_id)) for group_id in group_ids]
        non_empty = [chunk for chunk in chunks if chunk.size > 0]
        if not non_empty:
            return []
        return self.build_batches_from_example_indices(
            np.concatenate(non_empty),
            batch_size,
            pin_memory=pin_memory,
        )

    def build_batches_from_example_indices(
        self,
        example_indices: np.ndarray,
        batch_size: int,
        *,
        pin_memory: bool,
    ) -> List[PreparedBatch]:
        batches: List[PreparedBatch] = []
        if example_indices.size == 0:
            return batches

        for start in range(0, int(example_indices.size), batch_size):
            current_indices = np.asarray(example_indices[start : start + batch_size], dtype=np.int64)
            puzzle_indices = np.searchsorted(self.puzzle_indices, current_indices, side="right") - 1

            inputs = np.asarray(self.inputs[current_indices], dtype=np.int32)
            labels = np.asarray(self.labels[current_indices], dtype=np.int32)
            puzzle_identifiers = np.asarray(self.puzzle_identifiers[puzzle_indices], dtype=np.int32)

            if self.metadata.ignore_label_id is not None:
                labels = labels.copy()
                labels[labels == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

            effective_size = int(current_indices.size)
            if effective_size < batch_size:
                pad = batch_size - effective_size
                inputs = np.pad(inputs, ((0, pad), (0, 0)), constant_values=self.metadata.pad_id)
                labels = np.pad(labels, ((0, pad), (0, 0)), constant_values=IGNORE_LABEL_ID)
                puzzle_identifiers = np.pad(
                    puzzle_identifiers,
                    ((0, pad),),
                    constant_values=self.metadata.blank_identifier_id,
                )

            batch = {
                "inputs": torch.from_numpy(inputs),
                "labels": torch.from_numpy(labels),
                "puzzle_identifiers": torch.from_numpy(puzzle_identifiers),
            }
            if pin_memory:
                batch = {key: value.pin_memory() for key, value in batch.items()}
            batches.append(PreparedBatch(batch=batch, effective_size=effective_size))

        return batches


def evaluate_prepared_batches(
    model: torch.nn.Module,
    prepared_batches: Sequence[PreparedBatch],
    *,
    device: torch.device,
    hidden_diff_threshold: Optional[float],
) -> Dict[str, float]:
    if not prepared_batches:
        return {
            "count": 0.0,
            "accuracy": 0.0,
            "exact_accuracy": 0.0,
            "lm_loss": 0.0,
            "steps": 0.0,
        }

    model.eval()
    metric_totals = {key: 0.0 for key in ("count", "accuracy", "exact_accuracy", "lm_loss", "steps")}

    metric_return_keys = {"preds", "logits"}
    return_keys = set(metric_return_keys)
    act_early_stop_enabled = _act_early_stop_enabled(model)
    if act_early_stop_enabled:
        return_keys.add("q_halt_logits")

    hidden_pruning_enabled = (
        hidden_diff_threshold is not None
        and hidden_diff_threshold > 0
        and _supports_hidden_pruning(model)
    )

    with torch.inference_mode():
        for prepared in prepared_batches:
            batch = {key: value.to(device, non_blocking=True) for key, value in prepared.batch.items()}
            batch_size = batch["inputs"].shape[0]
            keep_mask = _valid_example_mask(batch["labels"])

            with torch.device(str(device)):
                carry = model.initial_carry(batch)  # type: ignore[misc]

            active_batch = batch
            active_indices = torch.arange(batch_size, device=device)
            pruned_chunks: List[Dict[str, Any]] = []
            preds: Optional[Dict[str, torch.Tensor]] = None

            while True:
                carry_before_step = None
                if hidden_pruning_enabled and active_indices.numel() > 0 and hasattr(carry, "current_hidden"):
                    carry_before_step = carry

                carry, _loss, _metrics, preds, all_finish = model(
                    carry=carry,
                    batch=active_batch,
                    return_keys=return_keys,
                )

                model_halted_mask: Optional[torch.Tensor] = None
                if hasattr(carry, "halted") and carry.halted is not None and not all_finish:
                    model_halted_mask = carry.halted

                act_pruned_mask: Optional[torch.Tensor] = model_halted_mask
                if act_early_stop_enabled and preds is not None and not all_finish:
                    q_halt_logits = preds.get("q_halt_logits")
                    if q_halt_logits is None:
                        raise RuntimeError("ACT early stopping requires q_halt_logits during proxy evaluation.")
                    q_halt_pruned_mask = q_halt_logits > 0
                    act_pruned_mask = (
                        q_halt_pruned_mask
                        if act_pruned_mask is None
                        else (act_pruned_mask | q_halt_pruned_mask)
                    )

                hidden_pruned_mask: Optional[torch.Tensor] = None
                if hidden_pruning_enabled and carry_before_step is not None and not all_finish:
                    current_hidden = carry.current_hidden.detach()
                    previous_hidden = carry_before_step.current_hidden.detach()
                    hidden_diff_norm = torch.norm(current_hidden - previous_hidden, dim=(1, 2)) / (
                        1e-7 + torch.norm(current_hidden + previous_hidden, dim=(1, 2)) / 2
                    )
                    hidden_pruned_mask = hidden_diff_norm <= hidden_diff_threshold

                prune_mask: Optional[torch.Tensor] = None
                if act_pruned_mask is not None and hidden_pruned_mask is not None:
                    prune_mask = act_pruned_mask | hidden_pruned_mask
                elif act_pruned_mask is not None:
                    prune_mask = act_pruned_mask
                else:
                    prune_mask = hidden_pruned_mask

                if prune_mask is not None and torch.any(prune_mask):
                    _append_pruned_chunk(
                        pruned_chunks=pruned_chunks,
                        model=model,
                        active_indices=active_indices,
                        carry=carry,
                        preds=preds,
                        prune_mask=prune_mask,
                    )
                    keep_indices = torch.nonzero(~prune_mask, as_tuple=False).squeeze(-1)
                    if keep_indices.numel() == 0:
                        active_indices = active_indices[:0]
                        preds = None
                        break

                    active_indices = active_indices[keep_indices]
                    active_batch = {
                        key: value.index_select(0, keep_indices)
                        for key, value in active_batch.items()
                    }
                    carry = _index_structure(carry, keep_indices)

                if all_finish:
                    break

            if pruned_chunks:
                preds, final_steps = _merge_pruned_predictions(
                    model=model,
                    batch_size=batch_size,
                    return_keys=metric_return_keys,
                    active_indices=active_indices,
                    active_preds=preds,
                    active_steps=carry.steps if preds is not None else None,
                    pruned_chunks=pruned_chunks,
                )
            else:
                if preds is None:
                    raise RuntimeError("Proxy evaluation did not produce predictions.")
                final_steps = carry.steps

            batch_metric_sums = _compute_batch_metric_sums(
                model=model,
                batch=batch,
                preds=preds,
                final_steps=final_steps,
                keep_mask=keep_mask,
            )
            for key, value in batch_metric_sums.items():
                metric_totals[key] += float(value)

    count = max(metric_totals["count"], 1.0)
    return {
        "count": metric_totals["count"],
        "accuracy": metric_totals["accuracy"] / count,
        "exact_accuracy": metric_totals["exact_accuracy"] / count,
        "lm_loss": metric_totals["lm_loss"] / count,
        "steps": metric_totals["steps"] / count,
    }


def rank_groups_by_hardness(group_scores: Sequence[GroupScore], hard_group_pool: int) -> List[int]:
    ranked = sorted(
        group_scores,
        key=lambda item: (-item.failure_rate, -item.lm_loss, item.group_id),
    )
    return [score.group_id for score in ranked[:hard_group_pool]]


def split_hard_groups(
    hard_group_ids: Sequence[int],
    *,
    es_train_groups: int,
    es_val_groups: int,
    seed: int,
) -> tuple[List[int], List[int]]:
    if es_train_groups + es_val_groups > len(hard_group_ids):
        raise ValueError(
            "Requested train/val hard groups exceed the hard-group pool: "
            f"{es_train_groups} + {es_val_groups} > {len(hard_group_ids)}"
        )

    rng = np.random.default_rng(seed)
    permuted = rng.permutation(np.asarray(hard_group_ids, dtype=np.int64))
    es_train = permuted[:es_train_groups].tolist()
    es_val = permuted[es_train_groups : es_train_groups + es_val_groups].tolist()
    return es_train, es_val


def mine_hard_groups(
    model: torch.nn.Module,
    train_data: IndexedSplitData,
    *,
    batch_size: int,
    device: torch.device,
    proxy_loops: int,
    hidden_diff_threshold: Optional[float],
    hard_group_pool: int,
) -> List[GroupScore]:
    original_loops = get_model_loops(model)
    set_model_loops(model, proxy_loops)

    scores: List[GroupScore] = []
    try:
        for group_id in tqdm(range(train_data.metadata.total_groups), desc="Mining hard groups"):
            prepared_batches = train_data.build_batches_for_group(group_id, batch_size, pin_memory=torch.cuda.is_available())
            metrics = evaluate_prepared_batches(
                model,
                prepared_batches,
                device=device,
                hidden_diff_threshold=hidden_diff_threshold,
            )
            scores.append(
                GroupScore(
                    group_id=group_id,
                    failure_rate=1.0 - metrics["exact_accuracy"],
                    exact_accuracy=metrics["exact_accuracy"],
                    accuracy=metrics["accuracy"],
                    lm_loss=metrics["lm_loss"],
                    steps=metrics["steps"],
                    count=int(metrics["count"]),
                )
            )
    finally:
        if original_loops is not None:
            set_model_loops(model, original_loops)

    ranked_group_ids = rank_groups_by_hardness(scores, hard_group_pool)
    rank_lookup = set(ranked_group_ids)
    ranked_scores = [score for score in scores if score.group_id in rank_lookup]
    ranked_scores.sort(key=lambda item: ranked_group_ids.index(item.group_id))
    return ranked_scores


def _make_generation_seed(base_seed: int, generation_index: int, salt: int = 0) -> int:
    return int(base_seed + generation_index * 1009 + salt * 100_003)


def run_one_generation(
    model: torch.nn.Module,
    evolvable_specs,
    *,
    es_train_batches: Sequence[PreparedBatch],
    es_val_batches: Sequence[PreparedBatch],
    population: int,
    perturb_rank: int,
    sigma: float,
    lr: float,
    generation_seed: int,
    hidden_diff_threshold: Optional[float],
    device: torch.device,
) -> Dict[str, Any]:
    perturbations = PopulationPerturbations(
        list(evolvable_specs),
        population=population,
        rank=perturb_rank,
        sigma=sigma,
        seed=generation_seed,
    )

    candidate_fitness: List[float] = []
    for member_index in range(population):
        active_deltas = perturbations.apply_member_(member_index)
        try:
            metrics = evaluate_prepared_batches(
                model,
                es_train_batches,
                device=device,
                hidden_diff_threshold=hidden_diff_threshold,
            )
            candidate_fitness.append(-metrics["lm_loss"])
        finally:
            perturbations.revert_member_(active_deltas)

    normalized = normalize_fitness(candidate_fitness, device=device)
    updates = perturbations.compute_update_tensors(normalized)
    apply_updates_(evolvable_specs, updates, lr=lr)

    parent_val_metrics = evaluate_prepared_batches(
        model,
        es_val_batches,
        device=device,
        hidden_diff_threshold=hidden_diff_threshold,
    )

    return {
        "raw_fitness": [float(value) for value in candidate_fitness],
        "normalized_fitness_mean": float(normalized.mean().item()),
        "normalized_fitness_std": float(normalized.std(unbiased=False).item()),
        "parent_val_metrics": _floatify_metrics(parent_val_metrics),
    }


def choose_best_pilot_trial(trial_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not trial_results:
        raise ValueError("No pilot trial results were recorded.")

    return max(
        trial_results,
        key=lambda item: (
            item["final_val_metrics"]["exact_accuracy"],
            -item["final_val_metrics"]["lm_loss"],
        ),
    )


def _reference_eval_path(
    resolved_checkpoint: Path,
    *,
    loops: int,
    hidden_diff_threshold: float,
    max_problems: int,
) -> Path:
    return resolved_checkpoint.parent / (
        f"{resolved_checkpoint.stem}_evaluation_test_max_problems_{max_problems}"
        f"_loops_{loops}{threshold_suffix(hidden_diff_threshold)}.json"
    )


def load_reference_baseline(
    checkpoint_path: str,
    *,
    loops: int,
    hidden_diff_threshold: float,
    max_problems: int,
) -> Optional[BaselineReference]:
    resolved = _resolve_checkpoint_path(checkpoint_path)
    if resolved is None:
        return None

    resolved_checkpoint = Path(resolved)
    reference_path = _reference_eval_path(
        resolved_checkpoint,
        loops=loops,
        hidden_diff_threshold=hidden_diff_threshold,
        max_problems=max_problems,
    )
    if not reference_path.is_file():
        return None

    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    exact_accuracy = float(payload["metrics"]["overall"]["exact_accuracy"])
    return BaselineReference(
        path=reference_path,
        exact_accuracy=exact_accuracy,
        batch_size=int(payload.get("batch_size", FULL_CONFIRM_DEFAULT_BATCH_SIZE)),
        max_problems=int(payload.get("max_problems", max_problems)),
        loops=int(payload.get("loops", loops)),
        hidden_diff_threshold=float(payload.get("hidden_diff_threshold", hidden_diff_threshold)),
        payload=payload,
    )


def run_full_confirmation(
    model: torch.nn.Module,
    config,
    *,
    checkpoint_path: str,
    output_path: Path,
    batch_size: int,
    loops: int,
    max_problems: int,
    hidden_diff_threshold: float,
    device: torch.device,
    enable_arc_majority: bool = True,
) -> Dict[str, Any]:
    original_loops = get_model_loops(model)
    set_model_loops(model, loops)

    confirm_config = copy.deepcopy(config)
    confirm_config.global_batch_size = batch_size

    try:
        dataloader = create_test_dataloader(confirm_config, "test", batch_size)
        metadata = dataloader.dataset.metadata
        arc_evaluator = None
        if enable_arc_majority:
            arc_evaluator = maybe_create_arc_majority_vote_evaluator(
                data_path=confirm_config.data_path,
                eval_metadata=metadata,
                split="test",
            )

        metrics_by_set, loop_metrics_by_step, _saved_outputs, processed_batches, processed_problems = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=device,
            max_problems=max_problems,
            max_batches=None,
            save_predictions=False,
            hidden_diff_threshold=hidden_diff_threshold,
            loop_checkpoints=_power_of_two_loop_checkpoints(loops),
            arc_majority_evaluator=arc_evaluator,
        )

        arc_metrics = None
        arc_submission_path = None
        if arc_evaluator is not None:
            arc_submission_dir = default_arc_submission_dir(output_path)
            arc_metrics = arc_evaluator.result(str(arc_submission_dir))
            arc_submission_path = arc_submission_dir / "submission.json"

        payload = {
            "checkpoint": checkpoint_path,
            "data_path": confirm_config.data_path,
            "split": "test",
            "batch_size": batch_size,
            "loops": loops,
            "hidden_diff_threshold": hidden_diff_threshold,
            "max_problems": max_problems,
            "processed_batches": processed_batches,
            "processed_problems": processed_problems,
            "metrics": metrics_by_set,
            "power_of_two_loop_metrics": loop_metrics_by_step,
            "arc_majority_vote_metrics": arc_metrics,
            "arc_submission_path": str(arc_submission_path) if arc_submission_path is not None else None,
        }
        write_json(output_path, payload)
        return payload
    finally:
        if original_loops is not None:
            set_model_loops(model, original_loops)


def run_short_evaluation(
    model: torch.nn.Module,
    config,
    *,
    checkpoint_path: str,
    output_path: Path,
    batch_size: int,
    loops: int,
    max_problems: int,
    hidden_diff_threshold: float,
    device: torch.device,
) -> Dict[str, Any]:
    payload = run_full_confirmation(
        model,
        config,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        batch_size=batch_size,
        loops=loops,
        max_problems=max_problems,
        hidden_diff_threshold=hidden_diff_threshold,
        device=device,
        enable_arc_majority=False,
    )
    payload["evaluation_kind"] = "short"
    return payload


def init_wandb_run(
    run_config: EggrollFinetuneConfig,
    *,
    output_dir: Path,
    resume_run_id: Optional[str],
) -> tuple[Optional[Any], Optional[str]]:
    if not run_config.wandb.enabled:
        return None, None

    run_name = run_config.wandb.name or output_dir.name
    run_dir = run_config.wandb.directory or str(output_dir / "wandb")
    run_id = resume_run_id or output_dir.name
    run = wandb.init(
        project=run_config.wandb.project,
        name=run_name,
        config=run_config.model_dump(mode="json"),
        dir=run_dir,
        mode=run_config.wandb.mode,
        tags=run_config.wandb.tags,
        id=run_id,
        resume="allow" if run_config.wandb.resume else None,
        settings=wandb.Settings(_disable_stats=True),
    )
    return run, run_id


def save_search_state(
    output_dir: Path,
    *,
    run_config: EggrollFinetuneConfig,
    phase: str,
    generation: int,
    current_lr: float,
    current_sigma: float,
    best_val_metrics: Dict[str, float],
    best_generation: int,
    baseline_proxy_metrics: Dict[str, Any],
    reference_baseline: Optional[BaselineReference],
    current_state: Dict[str, torch.Tensor],
    best_state: Dict[str, torch.Tensor],
    hard_group_ids: Sequence[int],
    es_train_group_ids: Sequence[int],
    es_val_group_ids: Sequence[int],
    widened: bool,
    patience_counter: int,
    stalled_generations: int,
    last_confirm_exact: float,
    wandb_run_id: Optional[str],
) -> None:
    state_path = output_dir / "search_state.pt"
    torch.save(
        {
            "version": 1,
            "phase": phase,
            "generation": generation,
            "run_config": run_config.model_dump(mode="json"),
            "current_lr": current_lr,
            "current_sigma": current_sigma,
            "best_val_metrics": best_val_metrics,
            "best_generation": best_generation,
            "baseline_proxy_metrics": baseline_proxy_metrics,
            "reference_baseline": None if reference_baseline is None else asdict(reference_baseline),
            "current_state": current_state,
            "best_state": best_state,
            "hard_group_ids": list(hard_group_ids),
            "es_train_group_ids": list(es_train_group_ids),
            "es_val_group_ids": list(es_val_group_ids),
            "widened": widened,
            "patience_counter": patience_counter,
            "stalled_generations": stalled_generations,
            "last_confirm_exact": last_confirm_exact,
            "wandb_run_id": wandb_run_id,
        },
        state_path,
    )


def load_search_state(output_dir: Path) -> Optional[Dict[str, Any]]:
    state_path = output_dir / "search_state.pt"
    if not state_path.is_file():
        return None
    return torch.load(state_path, map_location="cpu")


def _load_external_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if config_path.suffix.lower() == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _set_nested_value(payload: Dict[str, Any], dotted_key: str, value: Any) -> None:
    current = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def parse_config(argv: Optional[Sequence[str]] = None) -> EggrollFinetuneConfig:
    parser = argparse.ArgumentParser(description="Continue ARC-AGI URM training with a PyTorch Eggroll ES loop.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML/JSON config file.")

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--proxy_loops", type=int, default=None)
    parser.add_argument("--confirm_loops", type=int, default=None)
    parser.add_argument("--hidden_diff_threshold", type=float, default=None)
    parser.add_argument("--hard_group_pool", type=int, default=None)
    parser.add_argument("--es_train_groups", type=int, default=None)
    parser.add_argument("--es_val_groups", type=int, default=None)
    parser.add_argument("--max_generations", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--pilot_generations", type=int, default=None)
    parser.add_argument("--pilot_sigmas", type=float, nargs="+", default=None)
    parser.add_argument("--pilot_lrs", type=float, nargs="+", default=None)
    parser.add_argument("--confirm_every_generations", type=int, default=None)
    parser.add_argument("--confirm_improvement_delta", type=float, default=None)
    parser.add_argument("--widen_after_generations", type=int, default=None)
    parser.add_argument("--acceptance_exact_accuracy", type=float, default=None)

    parser.add_argument("--wandb_enabled", "--wandb-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb_project", "--wandb-project", type=str, default=None)
    parser.add_argument("--wandb_name", "--wandb-name", type=str, default=None)
    parser.add_argument("--wandb_mode", "--wandb-mode", type=str, default=None)
    parser.add_argument("--wandb_directory", "--wandb-directory", type=str, default=None)
    parser.add_argument("--wandb_tags", "--wandb-tags", type=str, nargs="+", default=None)
    parser.add_argument("--wandb_resume", "--wandb-resume", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--short_eval_enabled", "--short-eval-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--short_eval_every_generations", "--short-eval-every-generations", type=int, default=None)
    parser.add_argument("--short_eval_max_problems", "--short-eval-max-problems", type=int, default=None)
    parser.add_argument("--short_eval_batch_size", "--short-eval-batch-size", type=int, default=None)
    parser.add_argument("--short_eval_loops", "--short-eval-loops", type=int, default=None)
    parser.add_argument("--short_eval_hidden_diff_threshold", "--short-eval-hidden-diff-threshold", type=float, default=None)

    namespace = parser.parse_args(argv)
    raw_payload: Dict[str, Any] = {}
    if namespace.config is not None:
        raw_payload = _load_external_config(namespace.config) or {}

    field_mapping = {
        "checkpoint": "checkpoint",
        "output_dir": "output_dir",
        "seed": "seed",
        "population": "population",
        "rank": "rank",
        "sigma": "sigma",
        "lr": "lr",
        "proxy_loops": "proxy_loops",
        "confirm_loops": "confirm_loops",
        "hidden_diff_threshold": "hidden_diff_threshold",
        "hard_group_pool": "hard_group_pool",
        "es_train_groups": "es_train_groups",
        "es_val_groups": "es_val_groups",
        "max_generations": "max_generations",
        "patience": "patience",
        "pilot_generations": "pilot_generations",
        "pilot_sigmas": "pilot_sigmas",
        "pilot_lrs": "pilot_lrs",
        "confirm_every_generations": "confirm_every_generations",
        "confirm_improvement_delta": "confirm_improvement_delta",
        "widen_after_generations": "widen_after_generations",
        "acceptance_exact_accuracy": "acceptance_exact_accuracy",
        "wandb_enabled": "wandb.enabled",
        "wandb_project": "wandb.project",
        "wandb_name": "wandb.name",
        "wandb_mode": "wandb.mode",
        "wandb_directory": "wandb.directory",
        "wandb_tags": "wandb.tags",
        "wandb_resume": "wandb.resume",
        "short_eval_enabled": "short_eval.enabled",
        "short_eval_every_generations": "short_eval.every_generations",
        "short_eval_max_problems": "short_eval.max_problems",
        "short_eval_batch_size": "short_eval.batch_size",
        "short_eval_loops": "short_eval.loops",
        "short_eval_hidden_diff_threshold": "short_eval.hidden_diff_threshold",
    }
    for arg_name, dotted_key in field_mapping.items():
        value = getattr(namespace, arg_name)
        if value is not None:
            _set_nested_value(raw_payload, dotted_key, value)

    return EggrollFinetuneConfig(**raw_payload)


def main() -> None:
    run_config = parse_config()
    output_dir = Path(run_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    confirm_history_path = output_dir / "confirmations.jsonl"
    short_eval_history_path = output_dir / "short_evaluations.jsonl"
    pilot_results_path = output_dir / "pilot_results.json"
    hard_groups_path = output_dir / "hard_groups.json"
    config_json_path = output_dir / "eggroll_config.json"
    config_yaml_path = output_dir / "eggroll_config.yaml"

    resolved_checkpoint = _resolve_checkpoint_path(run_config.checkpoint)
    if resolved_checkpoint is None:
        raise FileNotFoundError(f"Could not resolve checkpoint path from {run_config.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(run_config.seed)
    np.random.seed(run_config.seed)

    print(f"Loading checkpoint config from: {run_config.checkpoint}")
    config = load_config_from_checkpoint_path(run_config.checkpoint)
    print(f"Using device: {device}")
    print(f"Using dataset: {config.data_path}")

    train_data = IndexedSplitData(config.data_path, "train")

    print("Reconstructing evaluation model...")
    model = create_model_for_evaluation(config, train_data.metadata, device=device)
    checkpoint_step = load_model_weights(model, resolved_checkpoint, device=device, strict=True)
    print(f"Loaded model weights from: {resolved_checkpoint}")
    if checkpoint_step is not None:
        print(f"Checkpoint step: {checkpoint_step}")

    reference_baseline = load_reference_baseline(
        resolved_checkpoint,
        loops=run_config.confirm_loops,
        hidden_diff_threshold=run_config.hidden_diff_threshold,
        max_problems=FULL_CONFIRM_DEFAULT_MAX_PROBLEMS,
    )
    if reference_baseline is not None:
        print(f"Found reference baseline JSON: {reference_baseline.path}")
    else:
        print("Reference baseline JSON was not found next to the checkpoint; confirmations will still run.")

    acceptance_exact = (
        float(run_config.acceptance_exact_accuracy)
        if run_config.acceptance_exact_accuracy is not None
        else (
            reference_baseline.exact_accuracy
            if reference_baseline is not None
            else 0.488525390625
        )
    )
    confirm_batch_size = (
        reference_baseline.batch_size
        if reference_baseline is not None
        else FULL_CONFIRM_DEFAULT_BATCH_SIZE
    )
    confirm_max_problems = (
        reference_baseline.max_problems
        if reference_baseline is not None
        else FULL_CONFIRM_DEFAULT_MAX_PROBLEMS
    )

    state = load_search_state(output_dir)
    if state is not None and state.get("phase") == "accepted":
        print(f"An accepted run already exists in {output_dir}. Nothing else to do.")
        return

    write_json(config_json_path, run_config.model_dump(mode="json"))
    write_yaml(config_yaml_path, run_config.model_dump(mode="json"))

    hard_group_ids: List[int]
    es_train_group_ids: List[int]
    es_val_group_ids: List[int]
    baseline_proxy_metrics: Dict[str, Any]
    best_state: Dict[str, torch.Tensor]
    current_state: Dict[str, torch.Tensor]
    best_generation: int
    best_val_metrics: Dict[str, float]
    current_lr: float
    current_sigma: float
    generation_start: int
    widened: bool
    patience_counter: int
    stalled_generations: int
    last_confirm_exact: float
    wandb_run: Optional[Any] = None
    wandb_run_id: Optional[str] = None

    try:
        wandb_run, wandb_run_id = init_wandb_run(
            run_config,
            output_dir=output_dir,
            resume_run_id=None if state is None else state.get("wandb_run_id"),
        )

        if state is not None:
            print(f"Resuming search state from: {output_dir / 'search_state.pt'}")
            widened = bool(state["widened"])
            evolvable_specs = collect_evolvable_parameters(model, include_small_tensors=widened)
            load_parameter_state_(evolvable_specs, state["current_state"])

            hard_group_ids = [int(value) for value in state["hard_group_ids"]]
            es_train_group_ids = [int(value) for value in state["es_train_group_ids"]]
            es_val_group_ids = [int(value) for value in state["es_val_group_ids"]]
            baseline_proxy_metrics = state["baseline_proxy_metrics"]
            best_state = state["best_state"]
            current_state = state["current_state"]
            best_generation = int(state["best_generation"])
            best_val_metrics = _floatify_metrics(state["best_val_metrics"])
            current_lr = float(state["current_lr"])
            current_sigma = float(state["current_sigma"])
            generation_start = int(state["generation"])
            patience_counter = int(state["patience_counter"])
            stalled_generations = int(state["stalled_generations"])
            last_confirm_exact = float(state["last_confirm_exact"])

            maybe_log_wandb(
                wandb_run,
                {
                    "resume": {
                        "generation_start": generation_start,
                        "current_lr": current_lr,
                        "current_sigma": current_sigma,
                        "best_generation": best_generation,
                        "best_val_metrics": best_val_metrics,
                    }
                },
                step=generation_start,
            )
        else:
            print("Collecting v1 evolvable parameters (2D non-embedding tensors only)...")
            evolvable_specs = collect_evolvable_parameters(model, include_small_tensors=False)
            print(
                f"Selected {len(evolvable_specs)} evolvable tensors covering "
                f"{sum(spec.parameter.numel() for spec in evolvable_specs):,} parameters."
            )

            hard_scores = mine_hard_groups(
                model,
                train_data,
                batch_size=config.global_batch_size,
                device=device,
                proxy_loops=run_config.proxy_loops,
                hidden_diff_threshold=run_config.hidden_diff_threshold,
                hard_group_pool=run_config.hard_group_pool,
            )
            hard_group_ids = [score.group_id for score in hard_scores]
            es_train_group_ids, es_val_group_ids = split_hard_groups(
                hard_group_ids,
                es_train_groups=run_config.es_train_groups,
                es_val_groups=run_config.es_val_groups,
                seed=0,
            )
            write_json(
                hard_groups_path,
                {
                    "hard_group_ids": hard_group_ids,
                    "es_train_group_ids": es_train_group_ids,
                    "es_val_group_ids": es_val_group_ids,
                    "scores": [asdict(score) for score in hard_scores],
                },
            )

            es_train_batches = train_data.build_batches_from_group_ids(
                es_train_group_ids,
                config.global_batch_size,
                pin_memory=torch.cuda.is_available(),
            )
            es_val_batches = train_data.build_batches_from_group_ids(
                es_val_group_ids,
                config.global_batch_size,
                pin_memory=torch.cuda.is_available(),
            )

            original_loops = get_model_loops(model)
            set_model_loops(model, run_config.proxy_loops)
            try:
                baseline_proxy_metrics = {
                    "es_train": evaluate_prepared_batches(
                        model,
                        es_train_batches,
                        device=device,
                        hidden_diff_threshold=run_config.hidden_diff_threshold,
                    ),
                    "es_val": evaluate_prepared_batches(
                        model,
                        es_val_batches,
                        device=device,
                        hidden_diff_threshold=run_config.hidden_diff_threshold,
                    ),
                }
            finally:
                if original_loops is not None:
                    set_model_loops(model, original_loops)

            print("Baseline proxy metrics:")
            print(json.dumps(baseline_proxy_metrics, indent=2))
            write_json(output_dir / "baseline_proxy_metrics.json", baseline_proxy_metrics)
            maybe_log_wandb(wandb_run, {"baseline_proxy": baseline_proxy_metrics}, step=0)

            baseline_confirm = run_full_confirmation(
                model,
                config,
                checkpoint_path=resolved_checkpoint,
                output_path=output_dir / "baseline_confirmation.json",
                batch_size=confirm_batch_size,
                loops=run_config.confirm_loops,
                max_problems=confirm_max_problems,
                hidden_diff_threshold=run_config.hidden_diff_threshold,
                device=device,
            )
            append_jsonl(confirm_history_path, {"generation": 0, "phase": "baseline", **baseline_confirm})
            maybe_log_wandb(
                wandb_run,
                {"baseline_confirmation": baseline_confirm},
                step=0,
            )

            if reference_baseline is not None:
                observed_exact = float(baseline_confirm["metrics"]["overall"]["exact_accuracy"])
                tolerance = 1.0 / max(1, reference_baseline.max_problems)
                delta = abs(observed_exact - reference_baseline.exact_accuracy)
                if delta > tolerance:
                    print(
                        "Warning: baseline full confirmation does not match the stored reference within tolerance. "
                        f"observed={observed_exact:.9f}, reference={reference_baseline.exact_accuracy:.9f}, "
                        f"tolerance={tolerance:.9f}"
                    )

            baseline_state = extract_parameter_state(evolvable_specs)
            trial_results: List[Dict[str, Any]] = []
            print("Starting pilot sweep...")
            for sigma in run_config.pilot_sigmas:
                for lr in run_config.pilot_lrs:
                    load_parameter_state_(evolvable_specs, baseline_state)
                    final_val_metrics: Dict[str, float] = baseline_proxy_metrics["es_val"]
                    for pilot_generation in range(1, run_config.pilot_generations + 1):
                        set_model_loops(model, run_config.proxy_loops)
                        generation_summary = run_one_generation(
                            model,
                            evolvable_specs,
                            es_train_batches=es_train_batches,
                            es_val_batches=es_val_batches,
                            population=run_config.population,
                            perturb_rank=run_config.rank,
                            sigma=sigma,
                            lr=lr,
                            generation_seed=_make_generation_seed(
                                run_config.seed,
                                pilot_generation,
                                salt=int(sigma * 1e8) + int(lr * 1e8),
                            ),
                            hidden_diff_threshold=run_config.hidden_diff_threshold,
                            device=device,
                        )
                        final_val_metrics = generation_summary["parent_val_metrics"]
                        pilot_row = {
                            "phase": "pilot",
                            "sigma": sigma,
                            "lr": lr,
                            "pilot_generation": pilot_generation,
                            **generation_summary,
                        }
                        append_jsonl(history_path, pilot_row)
                        maybe_log_wandb(wandb_run, {"pilot": pilot_row})

                    trial_results.append(
                        {
                            "sigma": sigma,
                            "lr": lr,
                            "final_val_metrics": final_val_metrics,
                            "state": extract_parameter_state(evolvable_specs),
                        }
                    )

            serializable_trials = [
                {
                    "sigma": result["sigma"],
                    "lr": result["lr"],
                    "final_val_metrics": result["final_val_metrics"],
                }
                for result in trial_results
            ]
            write_json(pilot_results_path, {"pilot_trials": serializable_trials})

            winner = choose_best_pilot_trial(trial_results)
            print("Pilot winner:")
            print(json.dumps({key: value for key, value in winner.items() if key != "state"}, indent=2))
            maybe_log_wandb(
                wandb_run,
                {
                    "pilot_winner": {
                        "sigma": winner["sigma"],
                        "lr": winner["lr"],
                        "final_val_metrics": winner["final_val_metrics"],
                    }
                },
                step=0,
            )
            load_parameter_state_(evolvable_specs, winner["state"])

            current_state = extract_parameter_state(evolvable_specs)
            best_state = extract_parameter_state(evolvable_specs)
            best_val_metrics = _floatify_metrics(winner["final_val_metrics"])
            best_generation = 0
            current_lr = float(winner["lr"])
            current_sigma = float(winner["sigma"])
            generation_start = 0
            widened = False
            patience_counter = 0
            stalled_generations = 0
            last_confirm_exact = float(baseline_proxy_metrics["es_val"]["exact_accuracy"])

            if best_val_metrics["exact_accuracy"] - last_confirm_exact >= run_config.confirm_improvement_delta:
                confirmation_payload = run_full_confirmation(
                    model,
                    config,
                    checkpoint_path=resolved_checkpoint,
                    output_path=output_dir / "confirmation_generation_0.json",
                    batch_size=confirm_batch_size,
                    loops=run_config.confirm_loops,
                    max_problems=confirm_max_problems,
                    hidden_diff_threshold=run_config.hidden_diff_threshold,
                    device=device,
                )
                append_jsonl(
                    confirm_history_path,
                    {
                        "generation": 0,
                        "reason": "post_pilot_gain",
                        **confirmation_payload,
                    },
                )
                maybe_log_wandb(
                    wandb_run,
                    {
                        "confirmation": {
                            "generation": 0,
                            **confirmation_payload,
                        },
                        "confirmation_trigger_post_pilot_gain": 1.0,
                    },
                    step=0,
                )
                confirmed_exact = float(confirmation_payload["metrics"]["overall"]["exact_accuracy"])
                last_confirm_exact = best_val_metrics["exact_accuracy"]
                if confirmed_exact > acceptance_exact:
                    accepted_dir = output_dir / "accepted"
                    accepted_dir.mkdir(parents=True, exist_ok=True)
                    accepted_checkpoint_path = accepted_dir / f"{Path(resolved_checkpoint).stem}_eggroll_accepted.pt"
                    torch.save(
                        {
                            "step": checkpoint_step,
                            "model_state_dict": model.state_dict(),
                            "source_checkpoint": resolved_checkpoint,
                            "es_generation": 0,
                            "es_val_metrics": best_val_metrics,
                            "confirmed_exact_accuracy": confirmed_exact,
                        },
                        accepted_checkpoint_path,
                    )
                    write_json(
                        accepted_dir / "accepted_summary.json",
                        {
                            "accepted_checkpoint": accepted_checkpoint_path,
                            "generation": 0,
                            "es_val_metrics": best_val_metrics,
                            "confirmation": confirmation_payload,
                            "reference_exact_accuracy": acceptance_exact,
                        },
                    )
                    maybe_log_wandb(
                        wandb_run,
                        {
                            "accepted": {
                                "generation": 0,
                                "confirmed_exact_accuracy": confirmed_exact,
                                "reference_exact_accuracy": acceptance_exact,
                            }
                        },
                        step=0,
                    )
                    save_search_state(
                        output_dir,
                        run_config=run_config,
                        phase="accepted",
                        generation=0,
                        current_lr=current_lr,
                        current_sigma=current_sigma,
                        best_val_metrics=best_val_metrics,
                        best_generation=best_generation,
                        baseline_proxy_metrics=baseline_proxy_metrics,
                        reference_baseline=reference_baseline,
                        current_state=current_state,
                        best_state=best_state,
                        hard_group_ids=hard_group_ids,
                        es_train_group_ids=es_train_group_ids,
                        es_val_group_ids=es_val_group_ids,
                        widened=widened,
                        patience_counter=patience_counter,
                        stalled_generations=stalled_generations,
                        last_confirm_exact=last_confirm_exact,
                        wandb_run_id=wandb_run_id,
                    )
                    print(
                        f"Accepted checkpoint saved to {accepted_checkpoint_path} directly after the pilot sweep with "
                        f"confirmed exact_accuracy={confirmed_exact:.6f} > {acceptance_exact:.6f}"
                    )
                    return

            save_search_state(
                output_dir,
                run_config=run_config,
                phase="main",
                generation=generation_start,
                current_lr=current_lr,
                current_sigma=current_sigma,
                best_val_metrics=best_val_metrics,
                best_generation=best_generation,
                baseline_proxy_metrics=baseline_proxy_metrics,
                reference_baseline=reference_baseline,
                current_state=current_state,
                best_state=best_state,
                hard_group_ids=hard_group_ids,
                es_train_group_ids=es_train_group_ids,
                es_val_group_ids=es_val_group_ids,
                widened=widened,
                patience_counter=patience_counter,
                stalled_generations=stalled_generations,
                last_confirm_exact=last_confirm_exact,
                wandb_run_id=wandb_run_id,
            )

        es_train_batches = train_data.build_batches_from_group_ids(
            es_train_group_ids,
            config.global_batch_size,
            pin_memory=torch.cuda.is_available(),
        )
        es_val_batches = train_data.build_batches_from_group_ids(
            es_val_group_ids,
            config.global_batch_size,
            pin_memory=torch.cuda.is_available(),
        )

        if state is not None:
            evolvable_specs = collect_evolvable_parameters(model, include_small_tensors=widened)
            print(
                f"Resumed with {len(evolvable_specs)} evolvable tensors covering "
                f"{sum(spec.parameter.numel() for spec in evolvable_specs):,} parameters."
            )

        print(
            f"Starting main search from generation {generation_start}, "
            f"lr={current_lr:.6g}, sigma={current_sigma:.6g}, "
            f"best es_val exact={best_val_metrics['exact_accuracy']:.6f}"
        )

        for generation in range(generation_start + 1, run_config.max_generations + 1):
            set_model_loops(model, run_config.proxy_loops)
            generation_summary = run_one_generation(
                model,
                evolvable_specs,
                es_train_batches=es_train_batches,
                es_val_batches=es_val_batches,
                population=run_config.population,
                perturb_rank=run_config.rank,
                sigma=current_sigma,
                lr=current_lr,
                generation_seed=_make_generation_seed(run_config.seed, generation),
                hidden_diff_threshold=run_config.hidden_diff_threshold,
                device=device,
            )
            parent_val_metrics = generation_summary["parent_val_metrics"]
            improved = parent_val_metrics["exact_accuracy"] > best_val_metrics["exact_accuracy"]

            if improved:
                best_val_metrics = _floatify_metrics(parent_val_metrics)
                best_state = extract_parameter_state(evolvable_specs)
                best_generation = generation
                patience_counter = 0
                stalled_generations = 0
                write_json(
                    output_dir / "best_parent_summary.json",
                    {
                        "generation": generation,
                        "best_val_metrics": best_val_metrics,
                        "lr": current_lr,
                        "sigma": current_sigma,
                    },
                )
                torch.save(
                    {
                        "generation": generation,
                        "metrics": best_val_metrics,
                        "evolved_state": best_state,
                    },
                    output_dir / "best_parent_state.pt",
                )
            else:
                patience_counter += 1
                stalled_generations += 1

            current_state = extract_parameter_state(evolvable_specs)
            history_row = {
                "phase": "main",
                "generation": generation,
                "lr": current_lr,
                "sigma": current_sigma,
                "improved": improved,
                "best_generation": best_generation,
                "best_val_metrics": best_val_metrics,
                **generation_summary,
            }
            append_jsonl(history_path, history_row)
            maybe_log_wandb(wandb_run, {"search": history_row}, step=generation)

            confirmation_reason: Optional[str] = None
            if improved and (best_val_metrics["exact_accuracy"] - last_confirm_exact) >= run_config.confirm_improvement_delta:
                confirmation_reason = "es_val_gain"
            elif generation % run_config.confirm_every_generations == 0:
                confirmation_reason = "periodic"

            if confirmation_reason is not None:
                target_state = best_state
                restore_after_confirm = current_state
                load_parameter_state_(evolvable_specs, target_state)
                confirmation_payload = run_full_confirmation(
                    model,
                    config,
                    checkpoint_path=resolved_checkpoint,
                    output_path=output_dir / f"confirmation_generation_{generation}.json",
                    batch_size=confirm_batch_size,
                    loops=run_config.confirm_loops,
                    max_problems=confirm_max_problems,
                    hidden_diff_threshold=run_config.hidden_diff_threshold,
                    device=device,
                )
                append_jsonl(
                    confirm_history_path,
                    {
                        "generation": generation,
                        "reason": confirmation_reason,
                        **confirmation_payload,
                    },
                )
                maybe_log_wandb(
                    wandb_run,
                    {
                        "confirmation": {
                            "generation": generation,
                            "metrics": confirmation_payload["metrics"],
                            "loops": run_config.confirm_loops,
                            "max_problems": confirm_max_problems,
                        },
                        f"confirmation_trigger_{confirmation_reason}": 1.0,
                    },
                    step=generation,
                )
                confirmed_exact = float(confirmation_payload["metrics"]["overall"]["exact_accuracy"])
                last_confirm_exact = best_val_metrics["exact_accuracy"]
                if confirmed_exact > acceptance_exact:
                    accepted_dir = output_dir / "accepted"
                    accepted_dir.mkdir(parents=True, exist_ok=True)
                    accepted_checkpoint_path = accepted_dir / f"{Path(resolved_checkpoint).stem}_eggroll_accepted.pt"
                    torch.save(
                        {
                            "step": checkpoint_step,
                            "model_state_dict": model.state_dict(),
                            "source_checkpoint": resolved_checkpoint,
                            "es_generation": generation,
                            "es_val_metrics": best_val_metrics,
                            "confirmed_exact_accuracy": confirmed_exact,
                        },
                        accepted_checkpoint_path,
                    )
                    write_json(
                        accepted_dir / "accepted_summary.json",
                        {
                            "accepted_checkpoint": accepted_checkpoint_path,
                            "generation": generation,
                            "es_val_metrics": best_val_metrics,
                            "confirmation": confirmation_payload,
                            "reference_exact_accuracy": acceptance_exact,
                        },
                    )
                    maybe_log_wandb(
                        wandb_run,
                        {
                            "accepted": {
                                "generation": generation,
                                "confirmed_exact_accuracy": confirmed_exact,
                                "reference_exact_accuracy": acceptance_exact,
                            }
                        },
                        step=generation,
                    )
                    save_search_state(
                        output_dir,
                        run_config=run_config,
                        phase="accepted",
                        generation=generation,
                        current_lr=current_lr,
                        current_sigma=current_sigma,
                        best_val_metrics=best_val_metrics,
                        best_generation=best_generation,
                        baseline_proxy_metrics=baseline_proxy_metrics,
                        reference_baseline=reference_baseline,
                        current_state=target_state,
                        best_state=best_state,
                        hard_group_ids=hard_group_ids,
                        es_train_group_ids=es_train_group_ids,
                        es_val_group_ids=es_val_group_ids,
                        widened=widened,
                        patience_counter=patience_counter,
                        stalled_generations=stalled_generations,
                        last_confirm_exact=last_confirm_exact,
                        wandb_run_id=wandb_run_id,
                    )
                    print(
                        f"Accepted checkpoint saved to {accepted_checkpoint_path} with "
                        f"confirmed exact_accuracy={confirmed_exact:.6f} > {acceptance_exact:.6f}"
                    )
                    return

                load_parameter_state_(evolvable_specs, restore_after_confirm)

            if patience_counter >= run_config.patience:
                print(
                    f"No es_val exact_accuracy improvement for {run_config.patience} generations. "
                    "Reverting to best parent and halving lr/sigma."
                )
                load_parameter_state_(evolvable_specs, best_state)
                current_state = extract_parameter_state(evolvable_specs)
                current_lr *= 0.5
                current_sigma *= 0.5
                patience_counter = 0
                maybe_log_wandb(
                    wandb_run,
                    {
                        "search_adjustment": {
                            "generation": generation,
                            "current_lr": current_lr,
                            "current_sigma": current_sigma,
                            "patience_reset": 1,
                        }
                    },
                    step=generation,
                )

            if (not widened) and stalled_generations >= run_config.widen_after_generations:
                print(
                    f"Search has stalled for {run_config.widen_after_generations} generations. "
                    "Widening the evolved set to include tiny tensors and dwconv."
                )
                load_parameter_state_(evolvable_specs, best_state)
                widened = True
                evolvable_specs = collect_evolvable_parameters(model, include_small_tensors=True)
                best_state = extract_parameter_state(evolvable_specs)
                current_state = extract_parameter_state(evolvable_specs)
                stalled_generations = 0
                patience_counter = 0
                widening_payload = {
                    "generation": generation,
                    "widened": True,
                    "num_evolved_tensors": len(evolvable_specs),
                    "num_evolved_parameters": sum(spec.parameter.numel() for spec in evolvable_specs),
                }
                torch.save(widening_payload, output_dir / "widened_search_state.pt")
                maybe_log_wandb(wandb_run, {"search_adjustment": widening_payload}, step=generation)

            if run_config.short_eval.enabled and generation % run_config.short_eval.every_generations == 0:
                short_eval_loops = (
                    run_config.short_eval.loops
                    if run_config.short_eval.loops is not None
                    else run_config.confirm_loops
                )
                short_eval_hidden_diff_threshold = (
                    run_config.short_eval.hidden_diff_threshold
                    if run_config.short_eval.hidden_diff_threshold is not None
                    else run_config.hidden_diff_threshold
                )
                short_eval_payload = run_short_evaluation(
                    model,
                    config,
                    checkpoint_path=resolved_checkpoint,
                    output_path=output_dir / f"short_eval_generation_{generation}.json",
                    batch_size=run_config.short_eval.batch_size,
                    loops=short_eval_loops,
                    max_problems=run_config.short_eval.max_problems,
                    hidden_diff_threshold=short_eval_hidden_diff_threshold,
                    device=device,
                )
                append_jsonl(
                    short_eval_history_path,
                    {
                        "generation": generation,
                        **short_eval_payload,
                    },
                )
                maybe_log_wandb(
                    wandb_run,
                    {"short_eval": short_eval_payload},
                    step=generation,
                )

            save_search_state(
                output_dir,
                run_config=run_config,
                phase="main",
                generation=generation,
                current_lr=current_lr,
                current_sigma=current_sigma,
                best_val_metrics=best_val_metrics,
                best_generation=best_generation,
                baseline_proxy_metrics=baseline_proxy_metrics,
                reference_baseline=reference_baseline,
                current_state=current_state,
                best_state=best_state,
                hard_group_ids=hard_group_ids,
                es_train_group_ids=es_train_group_ids,
                es_val_group_ids=es_val_group_ids,
                widened=widened,
                patience_counter=patience_counter,
                stalled_generations=stalled_generations,
                last_confirm_exact=last_confirm_exact,
                wandb_run_id=wandb_run_id,
            )

        print(
            f"Reached max_generations={run_config.max_generations} without surpassing the acceptance target "
            f"exact_accuracy={acceptance_exact:.6f}. Best es_val exact_accuracy was {best_val_metrics['exact_accuracy']:.6f}."
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
