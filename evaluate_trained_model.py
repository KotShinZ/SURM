# python evaluate_trained_model.py --checkpoint checkpoints/URM-sudoku-base --max_problems 4096 --loops 32 --batch_size 4096 --hidden_diff_threshold 0.1

import argparse
import json
import math
import os
import re
from collections import defaultdict, deque
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set
import time

import pydantic
import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.losses import IGNORE_LABEL_ID
from puzzle_dataset import MaskedInputConfig, PuzzleDataset, PuzzleDatasetConfig
from utils import load_model_class


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    name: str


class PretrainConfig(pydantic.BaseModel):
    arch: ArchConfig
    data_path: str
    evaluators: List[EvaluatorConfig] = []
    global_batch_size: int
    epochs: int
    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    target_q_update_every: int
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float
    grad_accum_steps: int = 1
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    load_checkpoint: Optional[str] = None
    load_strict: bool = True
    load_optimizer_state: bool = True
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []
    loop_deltas: List[str] = []
    ema: bool = False
    ema_rate: float = 0.999
    use_muon: bool = False
    data_fraction: float = 1.0
    masked_input: Optional[MaskedInputConfig] = None


def _ensure_arch_extra(config: PretrainConfig) -> Dict[str, Any]:
    if config.arch.__pydantic_extra__ is None:
        config.arch.__pydantic_extra__ = {}
    return config.arch.__pydantic_extra__


def _loop_count_config_key(config: PretrainConfig) -> str:
    arch_extra = config.arch.__pydantic_extra__ or {}
    if "halt_max_steps" in arch_extra:
        return "halt_max_steps"
    return "loops"


def _set_effective_loop_count(config: PretrainConfig, loops: int) -> str:
    arch_extra = _ensure_arch_extra(config)
    loop_key = _loop_count_config_key(config)
    arch_extra[loop_key] = loops
    return loop_key


def _get_effective_loop_count(config: PretrainConfig) -> Optional[int]:
    arch_extra = config.arch.__pydantic_extra__ or {}
    loop_key = _loop_count_config_key(config)
    loop_value = arch_extra.get(loop_key)
    if loop_value is None and loop_key != "loops":
        loop_value = arch_extra.get("loops")
    if loop_value is None and loop_key != "halt_max_steps":
        loop_value = arch_extra.get("halt_max_steps")
    return None if loop_value is None else int(loop_value)


def _resolve_checkpoint_path(path: str) -> Optional[str]:
    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        pattern = re.compile(r"step_(\d+)(?:\.pt)?$")
        candidates = []
        for file_name in os.listdir(path):
            match = pattern.match(file_name)
            if match:
                candidates.append((int(match.group(1)), os.path.join(path, file_name)))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[-1][1]

    return None


def load_config_from_checkpoint_path(path: str) -> PretrainConfig:
    resolved_path = _resolve_checkpoint_path(path)
    checkpoint_dir = Path(resolved_path if resolved_path is not None else path)
    if checkpoint_dir.is_file():
        checkpoint_dir = checkpoint_dir.parent

    candidates = [
        checkpoint_dir / "config.yaml",
        checkpoint_dir / "config.json",
        checkpoint_dir / "all_config.yaml",
        checkpoint_dir / ".hydra" / "config.yaml",
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue

        try:
            conf = OmegaConf.load(candidate)
            as_dict = OmegaConf.to_container(conf, resolve=True)
            if isinstance(as_dict, dict):
                return PretrainConfig(**as_dict)
        except Exception:
            pass

        try:
            with open(candidate, "r", encoding="utf-8") as f:
                config_dict = json.load(f) if candidate.suffix == ".json" else yaml.safe_load(f)
            if isinstance(config_dict, dict):
                return PretrainConfig(**config_dict)
        except Exception:
            pass

    raise FileNotFoundError(f"Could not find a valid config next to checkpoint path: {path}")


def create_test_dataloader(config: PretrainConfig, split: str, global_batch_size: int) -> DataLoader:
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_path=config.data_path,
            global_batch_size=global_batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
            masked_input=config.masked_input,
        ),
        split=split,
    )
    print(f"Dataset {split} has {dataset.metadata.total_groups} groups.")
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )


def create_model_for_evaluation(config: PretrainConfig, metadata, device: torch.device) -> nn.Module:
    arch_extra = config.arch.__pydantic_extra__ or {}
    model_cfg = dict(
        **arch_extra,
        batch_size=config.global_batch_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    model = model_cls(model_cfg)
    model = loss_head_cls(model, **(config.arch.loss.__pydantic_extra__ or {}))
    model = model.to(device)
    model_config = getattr(getattr(model, "model", None), "config", None)
    should_compile = (
        device.type == "cuda"
        and "DISABLE_COMPILE" not in os.environ
        and (model_config is None or not getattr(model_config, "profile", False))
    )
    if should_compile:
        model = torch.compile(model, dynamic=False)  # type: ignore[assignment]
    return model


def _remap_state_dict_prefix_if_needed(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    model_keys = list(model.state_dict().keys())
    if not model_keys or not state_dict:
        return state_dict

    model_has_orig_mod = model_keys[0].startswith("_orig_mod.")
    ckpt_has_orig_mod = next(iter(state_dict)).startswith("_orig_mod.")

    if model_has_orig_mod == ckpt_has_orig_mod:
        return state_dict

    if ckpt_has_orig_mod:
        return {
            key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key: value
            for key, value in state_dict.items()
        }

    return {f"_orig_mod.{key}": value for key, value in state_dict.items()}


def _resize_puzzle_embedding_if_needed(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    model_key = next((key for key in model.state_dict().keys() if key.endswith("puzzle_emb.weights")), None)
    state_key = next((key for key in state_dict.keys() if key.endswith("puzzle_emb.weights")), None)

    if model_key is None or state_key is None:
        return

    expected_shape = model.state_dict()[model_key].shape
    current_shape = state_dict[state_key].shape
    if current_shape == expected_shape:
        return

    print(
        "Resetting puzzle embedding because the stored shape does not match "
        f"(found {tuple(current_shape)}, expected {tuple(expected_shape)})."
    )
    puzzle_emb = state_dict[state_key]
    state_dict[state_key] = torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()


def load_model_weights(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool,
) -> Optional[int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = _remap_state_dict_prefix_if_needed(model, state_dict)
    _resize_puzzle_embedding_if_needed(model, state_dict)

    load_result = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        missing, unexpected = load_result
        if missing:
            print(f"Missing keys during checkpoint load: {missing}")
        if unexpected:
            print(f"Unexpected keys during checkpoint load: {unexpected}")

    if isinstance(checkpoint, dict):
        step = checkpoint.get("step")
        return None if step is None else int(step)
    return None


def _valid_example_mask(labels: torch.Tensor) -> torch.Tensor:
    return (labels != IGNORE_LABEL_ID).any(dim=1)


def _finalize_metric_totals(metric_totals: Dict[str, float]) -> Dict[str, float]:
    if not metric_totals:
        return {}

    count = max(metric_totals.get("count", 0.0), 1.0)
    finalized: Dict[str, float] = {}
    for key, value in metric_totals.items():
        if key == "count":
            finalized[key] = value
        elif key.startswith("profile/"):
            finalized[key] = value
        else:
            finalized[key] = value / count
    return finalized


def _concat_saved_outputs(saved_outputs: Dict[str, Dict[str, List[torch.Tensor]]]) -> Dict[str, Dict[str, torch.Tensor]]:
    concatenated: Dict[str, Dict[str, torch.Tensor]] = {}
    for set_name, per_set_outputs in saved_outputs.items():
        concatenated[set_name] = {}
        order_tensors = per_set_outputs.get("__order__")
        sort_indices: Optional[torch.Tensor] = None
        if order_tensors:
            concatenated_orders = torch.cat(order_tensors, dim=0)
            sort_indices = torch.argsort(concatenated_orders)

        for key, tensors in per_set_outputs.items():
            if key == "__order__":
                continue
            concatenated_tensor = torch.cat(tensors, dim=0) if tensors else torch.empty(0)
            if sort_indices is not None and concatenated_tensor.shape[0] == sort_indices.shape[0]:
                concatenated_tensor = concatenated_tensor.index_select(0, sort_indices)
            concatenated[set_name][key] = concatenated_tensor
    return concatenated


def _estimate_total_batches(dataloader: DataLoader) -> Optional[int]:
    dataset = getattr(dataloader, "dataset", None)
    if not isinstance(dataset, PuzzleDataset):
        return None

    try:
        dataset._lazy_load_dataset()
    except Exception:
        return None

    if dataset._data is None:
        return None

    return sum(
        math.ceil(len(per_set_data["inputs"]) / dataset.config.global_batch_size)
        for per_set_data in dataset._data.values()
    )


def _unwrap_eval_model(model: nn.Module) -> nn.Module:
    return getattr(model, "model", model)


def _supports_hidden_pruning(model: nn.Module) -> bool:
    base_model = _unwrap_eval_model(model)
    inner = getattr(base_model, "inner", None)
    return (
        inner is not None
        and hasattr(inner, "reset_carry")
        and hasattr(inner, "lm_head")
        and hasattr(inner, "q_head")
        and hasattr(inner, "puzzle_emb_len")
    )


def _act_early_stop_enabled(model: nn.Module) -> bool:
    base_model = _unwrap_eval_model(model)
    config = getattr(base_model, "config", None)
    return bool(getattr(config, "act_inference", False) or getattr(config, "eval_act_early_stop", False))


def _hidden_diff_norm(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.norm(x1 - x2, dim=(1, 2)) / (1e-7 + torch.norm(x1 + x2, dim=(1, 2)) / 2)


def _index_structure(obj: Any, indices: torch.Tensor) -> Any:
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.index_select(0, indices)
    if isinstance(obj, dict):
        return {key: _index_structure(value, indices) for key, value in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_index_structure(value, indices) for value in obj)
    if isinstance(obj, list):
        return [_index_structure(value, indices) for value in obj]
    if is_dataclass(obj):
        return type(obj)(**{field.name: _index_structure(getattr(obj, field.name), indices) for field in fields(obj)})
    return obj


def _concat_structure(*objs: Any) -> Any:
    non_null_objs = [obj for obj in objs if obj is not None]
    if not non_null_objs:
        return None

    first = non_null_objs[0]
    if torch.is_tensor(first):
        return torch.cat(non_null_objs, dim=0)
    if isinstance(first, dict):
        return {
            key: _concat_structure(*(obj[key] for obj in non_null_objs))
            for key in first
        }
    if isinstance(first, tuple):
        concatenated_items = tuple(
            _concat_structure(*(obj[index] for obj in non_null_objs))
            for index in range(len(first))
        )
        if hasattr(first, "_fields"):
            return type(first)(*concatenated_items)
        return concatenated_items
    if isinstance(first, list):
        return [
            _concat_structure(*(obj[index] for obj in non_null_objs))
            for index in range(len(first))
        ]
    if is_dataclass(first):
        return type(first)(
            **{
                field.name: _concat_structure(*(getattr(obj, field.name) for obj in non_null_objs))
                for field in fields(first)
            }
        )
    return first


def _slice_tensor_batch(
    batch: Dict[str, torch.Tensor],
    start: int,
    end: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    return {key: value[start:end] for key, value in batch.items()}


def _run_heads_from_hidden(
    model: nn.Module,
    hidden_states: torch.Tensor,
    return_keys: Set[str],
) -> Dict[str, torch.Tensor]:
    base_model = _unwrap_eval_model(model)
    inner = getattr(base_model, "inner", None)
    if inner is None:
        raise ValueError("This model does not expose an inner module for hidden-state pruning.")

    logits = inner.lm_head(hidden_states)[:, inner.puzzle_emb_len :]
    outputs: Dict[str, torch.Tensor] = {
        "logits": logits,
        "preds": torch.argmax(logits, dim=-1),
    }

    if "q_halt_logits" in return_keys or "q_continue_logits" in return_keys:
        q_logits = inner.q_head(hidden_states[:, 0]).to(torch.float32)
        if "q_halt_logits" in return_keys:
            outputs["q_halt_logits"] = q_logits[..., 0]
        if "q_continue_logits" in return_keys:
            outputs["q_continue_logits"] = q_logits[..., 1]

    return outputs


def _allocate_prediction_buffers(
    template_outputs: Dict[str, torch.Tensor],
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    return {
        key: torch.empty((batch_size, *value.shape[1:]), dtype=value.dtype, device=value.device)
        for key, value in template_outputs.items()
    }


def _merge_pruned_predictions(
    model: nn.Module,
    batch_size: int,
    return_keys: Set[str],
    active_indices: torch.Tensor,
    active_preds: Optional[Dict[str, torch.Tensor]],
    active_steps: Optional[torch.Tensor],
    pruned_chunks: List[Dict[str, Any]],
) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
    merged_preds: Optional[Dict[str, torch.Tensor]] = None
    merged_steps: Optional[torch.Tensor] = None

    if active_preds is not None and active_steps is not None and active_indices.numel() > 0:
        merged_preds = _allocate_prediction_buffers(active_preds, batch_size)
        merged_steps = torch.empty((batch_size,), dtype=active_steps.dtype, device=active_steps.device)
        for key, value in active_preds.items():
            merged_preds[key][active_indices] = value
        merged_steps[active_indices] = active_steps

    for chunk in pruned_chunks:
        chunk_outputs = chunk.get("outputs")
        if chunk_outputs is not None:
            chunk_preds = {key: value for key, value in chunk_outputs.items() if key in return_keys}
        else:
            chunk_preds = _run_heads_from_hidden(model, chunk["hidden"], return_keys=return_keys)
        chunk_indices = chunk["indices"]
        chunk_steps = chunk["steps"]

        if merged_preds is None:
            merged_preds = _allocate_prediction_buffers(chunk_preds, batch_size)
            merged_steps = torch.empty((batch_size,), dtype=chunk_steps.dtype, device=chunk_steps.device)

        for key, value in chunk_preds.items():
            merged_preds[key][chunk_indices] = value
        merged_steps[chunk_indices] = chunk_steps

    if merged_preds is None or merged_steps is None:
        raise RuntimeError("Failed to collect predictions for the evaluated batch.")

    return merged_preds, merged_steps


def _append_pruned_chunk(
    pruned_chunks: List[Dict[str, Any]],
    model: nn.Module,
    active_indices: torch.Tensor,
    carry: Any,
    preds: Optional[Dict[str, torch.Tensor]],
    prune_mask: torch.Tensor,
) -> None:
    pruned_indices = torch.nonzero(prune_mask, as_tuple=False).squeeze(-1)
    if pruned_indices.numel() == 0:
        return

    chunk: Dict[str, Any] = {
        "indices": active_indices[pruned_indices].detach().clone(),
        "steps": carry.steps[prune_mask].detach(),
    }

    if hasattr(carry, "current_hidden") and _supports_hidden_pruning(model):
        chunk["hidden"] = carry.current_hidden[prune_mask].detach()
    else:
        if preds is None:
            raise RuntimeError("Model evaluation did not produce predictions for the pruned examples.")
        chunk["outputs"] = {key: value[prune_mask].detach() for key, value in preds.items()}

    pruned_chunks.append(chunk)


def _compute_batch_metric_sums(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    preds: Dict[str, torch.Tensor],
    final_steps: torch.Tensor,
    keep_mask: torch.Tensor,
) -> Dict[str, float]:
    labels = batch["labels"]
    selected_labels = labels[keep_mask]
    selected_preds = preds["preds"][keep_mask]
    selected_steps = final_steps[keep_mask]

    token_mask = selected_labels != IGNORE_LABEL_ID
    token_counts = token_mask.sum(dim=1)
    valid_examples = token_counts > 0
    if not torch.any(valid_examples):
        return {"count": 0.0, "accuracy": 0.0, "exact_accuracy": 0.0, "lm_loss": 0.0, "steps": 0.0}

    selected_labels = selected_labels[valid_examples]
    selected_preds = selected_preds[valid_examples]
    selected_steps = selected_steps[valid_examples]
    token_mask = token_mask[valid_examples]
    token_counts = token_counts[valid_examples]

    is_correct = token_mask & (selected_preds == selected_labels)
    seq_token_accuracy = is_correct.to(torch.float32).sum(dim=1) / token_counts.clamp_min(1)
    seq_exact_accuracy = (is_correct.sum(dim=1) == token_counts).to(torch.float32)

    logits = preds["logits"][keep_mask][valid_examples]
    if hasattr(model, "loss_fn"):
        token_loss = model.loss_fn(logits, selected_labels, ignore_index=IGNORE_LABEL_ID)  # type: ignore[misc]
    else:
        token_loss = F.cross_entropy(
            logits.to(torch.float32).view(-1, logits.shape[-1]),
            selected_labels.to(torch.long).view(-1),
            ignore_index=IGNORE_LABEL_ID,
            reduction="none",
        ).view(selected_labels.shape)
    seq_lm_loss = token_loss.sum(dim=1) / token_counts.clamp_min(1)

    return {
        "count": float(valid_examples.sum().item()),
        "accuracy": float(seq_token_accuracy.sum().item()),
        "exact_accuracy": float(seq_exact_accuracy.sum().item()),
        "lm_loss": float(seq_lm_loss.sum().item()),
        "steps": float(selected_steps.to(torch.float32).sum().item()),
    }


def _accumulate_metric_sums_by_set_ids(
    metric_totals_by_set: Dict[str, Dict[str, float]],
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    preds: Dict[str, torch.Tensor],
    final_steps: torch.Tensor,
    set_ids: torch.Tensor,
    set_names: List[str],
) -> None:
    if set_ids.numel() == 0:
        return

    unique_set_ids = torch.unique(set_ids.to(torch.int64), sorted=True)
    for set_id in unique_set_ids.tolist():
        set_mask = set_ids == set_id
        metric_sums = _compute_batch_metric_sums(
            model=model,
            batch=batch,
            preds=preds,
            final_steps=final_steps,
            keep_mask=set_mask,
        )
        for metric_name, metric_value in metric_sums.items():
            metric_totals_by_set[set_names[set_id]][metric_name] += metric_value


def _accumulate_completed_steps_by_set_ids(
    completed_step_counts_by_set: Dict[str, Dict[int, float]],
    completed_total_counts_by_set: Dict[str, float],
    final_steps: torch.Tensor,
    set_ids: torch.Tensor,
    set_names: List[str],
) -> None:
    if final_steps.numel() == 0:
        return

    unique_set_ids = torch.unique(set_ids.to(torch.int64), sorted=True)
    for set_id in unique_set_ids.tolist():
        set_name = set_names[set_id]
        set_steps = final_steps[set_ids == set_id]
        if set_steps.numel() == 0:
            continue

        unique_steps, counts = torch.unique(set_steps.to(torch.int64), return_counts=True)
        completed_total_counts_by_set[set_name] += float(set_steps.numel())
        for step_value, count_value in zip(unique_steps.tolist(), counts.tolist()):
            completed_step_counts_by_set[set_name][int(step_value)] += float(count_value)


def _append_saved_output_records(
    saved_outputs: Dict[str, Dict[str, List[torch.Tensor]]],
    batch: Dict[str, torch.Tensor],
    preds: Dict[str, torch.Tensor],
    set_ids: torch.Tensor,
    set_names: List[str],
    orders: torch.Tensor,
) -> None:
    if set_ids.numel() == 0:
        return

    unique_set_ids = torch.unique(set_ids.to(torch.int64), sorted=True)
    for set_id in unique_set_ids.tolist():
        set_mask = set_ids == set_id
        set_name = set_names[set_id]
        saved_outputs[set_name]["__order__"].append(orders[set_mask].detach().cpu())
        for key, value in batch.items():
            if key in {"inputs", "labels", "puzzle_identifiers", "source_inputs"}:
                saved_outputs[set_name][key].append(value[set_mask].detach().cpu())
        for key, value in preds.items():
            saved_outputs[set_name][key].append(value[set_mask].detach().cpu())


def _finalize_evaluation_outputs(
    metric_totals_by_set: Dict[str, Dict[str, float]],
    loop_metric_totals_by_step: Dict[int, Dict[str, Dict[str, float]]],
    completed_step_counts_by_set: Dict[str, Dict[int, float]],
    completed_total_counts_by_set: Dict[str, float],
    saved_outputs: Dict[str, Dict[str, List[torch.Tensor]]],
) -> tuple[Dict[str, Dict[str, float]], Dict[int, Dict[str, Dict[str, float]]], Dict[str, Dict[str, torch.Tensor]]]:
    finalized_by_set = {
        set_name: _finalize_metric_totals(metric_totals)
        for set_name, metric_totals in metric_totals_by_set.items()
    }

    overall_totals: Dict[str, float] = defaultdict(float)
    for metric_totals in metric_totals_by_set.values():
        for key, value in metric_totals.items():
            overall_totals[key] += value
    finalized_by_set["overall"] = _finalize_metric_totals(overall_totals)

    finalized_loop_metrics_by_step: Dict[int, Dict[str, Dict[str, float]]] = {}
    loop_steps = sorted(loop_metric_totals_by_step)
    finished_fraction_by_loop_by_set = {
        set_name: _finished_fraction_by_loop(completed_step_counts, completed_total_counts_by_set[set_name], loop_steps)
        for set_name, completed_step_counts in completed_step_counts_by_set.items()
    }
    overall_completed_step_counts: Dict[int, float] = defaultdict(float)
    overall_completed_total_count = 0.0
    for set_name, completed_step_counts in completed_step_counts_by_set.items():
        overall_completed_total_count += completed_total_counts_by_set[set_name]
        for step_value, count_value in completed_step_counts.items():
            overall_completed_step_counts[step_value] += count_value
    overall_finished_fraction_by_loop = _finished_fraction_by_loop(
        dict(overall_completed_step_counts),
        overall_completed_total_count,
        loop_steps,
    )

    for loop_step, metric_totals_by_set_at_step in sorted(loop_metric_totals_by_step.items()):
        finalized_loop_metrics_by_step[loop_step] = {
            set_name: _finalize_metric_totals(metric_totals)
            for set_name, metric_totals in metric_totals_by_set_at_step.items()
        }
        for set_name, finished_fraction_by_loop in finished_fraction_by_loop_by_set.items():
            finalized_loop_metrics_by_step[loop_step].setdefault(set_name, {})
            finalized_loop_metrics_by_step[loop_step][set_name]["finished_fraction"] = finished_fraction_by_loop[
                loop_step
            ]

        overall_loop_totals: Dict[str, float] = defaultdict(float)
        for metric_totals in metric_totals_by_set_at_step.values():
            for key, value in metric_totals.items():
                overall_loop_totals[key] += value
        finalized_loop_metrics_by_step[loop_step]["overall"] = _finalize_metric_totals(overall_loop_totals)
        finalized_loop_metrics_by_step[loop_step]["overall"]["finished_fraction"] = overall_finished_fraction_by_loop[
            loop_step
        ]

    return finalized_by_set, finalized_loop_metrics_by_step, _concat_saved_outputs(saved_outputs)


def _power_of_two_loop_checkpoints(loops: Optional[int]) -> List[int]:
    if loops is None or loops < 1:
        return []

    checkpoints: List[int] = []
    current = 1
    while current <= loops:
        checkpoints.append(current)
        current *= 2
    return checkpoints


def _finished_fraction_by_loop(
    completed_step_counts: Dict[int, float],
    total_count: float,
    loop_steps: List[int],
) -> Dict[int, float]:
    if total_count <= 0:
        return {loop_step: 0.0 for loop_step in loop_steps}

    sorted_step_counts = sorted((int(step), float(count)) for step, count in completed_step_counts.items())
    finished_fraction_by_loop: Dict[int, float] = {}
    cumulative_finished = 0.0
    count_index = 0

    for loop_step in sorted(loop_steps):
        while count_index < len(sorted_step_counts) and sorted_step_counts[count_index][0] <= loop_step:
            cumulative_finished += sorted_step_counts[count_index][1]
            count_index += 1
        finished_fraction_by_loop[loop_step] = cumulative_finished / total_count

    return finished_fraction_by_loop


def _uniform_step_value(steps: torch.Tensor) -> Optional[int]:
    if steps.numel() == 0:
        return None

    first_step = steps[0]
    if bool(torch.all(steps == first_step).item()):
        return int(first_step.item())
    return None


def _evaluate_model_shrink(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_problems: Optional[int],
    max_batches: Optional[int],
    save_predictions: bool,
    hidden_diff_threshold: Optional[float],
    loop_checkpoints: Optional[List[int]] = None,
) -> tuple[Dict[str, Dict[str, float]], Dict[int, Dict[str, Dict[str, float]]], Dict[str, Dict[str, torch.Tensor]], int, int]:
    model.eval()

    return_keys: Set[str] = {"preds", "logits"}
    act_early_stop_enabled = _act_early_stop_enabled(model)
    if act_early_stop_enabled:
        return_keys.add("q_halt_logits")
    if save_predictions:
        return_keys.update({"q_halt_logits", "q_continue_logits"})
    metric_return_keys: Set[str] = {"preds", "logits"}
    loop_checkpoints = sorted(set(loop_checkpoints or []))

    metric_totals_by_set: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    loop_metric_totals_by_step: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    completed_step_counts_by_set: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    completed_total_counts_by_set: Dict[str, float] = defaultdict(float)
    saved_outputs: Dict[str, Dict[str, List[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))

    processed_batches = 0
    processed_problems = 0
    progress_total = _estimate_total_batches(dataloader)
    if progress_total is not None and max_batches is not None:
        progress_total = min(progress_total, max_batches)
    if progress_total is not None and max_problems is not None:
        progress_total = min(progress_total, math.ceil(max_problems / dataloader.dataset.config.global_batch_size))  # type: ignore[attr-defined]

    hidden_pruning_enabled = (
        hidden_diff_threshold is not None
        and hidden_diff_threshold > 0
        and _supports_hidden_pruning(model)
    )
    print("hidden_pruning_enabled:", hidden_pruning_enabled, ", hidden_diff_threshold:", hidden_diff_threshold)
    print("ACT early stopping enabled:", act_early_stop_enabled)
    if hidden_diff_threshold is not None and hidden_diff_threshold > 0 and not hidden_pruning_enabled:
        print("Hidden-state pruning is not supported by this model; running standard evaluation.")

    with torch.inference_mode():
        for batch_index, (set_name, batch, _global_batch_size) in enumerate(
            tqdm(dataloader, desc="Evaluating", total=progress_total)
        ):
            if max_batches is not None and batch_index >= max_batches:
                break
            if max_problems is not None and processed_problems >= max_problems:
                break

            processed_batches += 1
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            batch_size = batch["inputs"].shape[0]

            valid_example_mask = _valid_example_mask(batch["labels"])
            keep_mask = valid_example_mask.clone()
            if max_problems is not None:
                remaining = max_problems - processed_problems
                valid_indices = torch.nonzero(valid_example_mask, as_tuple=False).squeeze(-1)
                if remaining <= 0:
                    break
                if valid_indices.numel() > remaining:
                    keep_mask = torch.zeros_like(valid_example_mask)
                    keep_mask[valid_indices[:remaining]] = True

            with torch.device(str(device)):
                carry = model.initial_carry(batch)  # type: ignore[misc]

            active_batch = batch
            active_indices = torch.arange(batch_size, device=device)
            pruned_chunks: List[Dict[str, Any]] = []
            preds: Optional[Dict[str, torch.Tensor]] = None
            captured_loop_checkpoints: Set[int] = set()

            while True:
                carry_before_step = None
                if hidden_pruning_enabled and active_indices.numel() > 0 and hasattr(carry, "current_hidden"):
                    carry_before_step = carry #_unwrap_eval_model(model).inner.reset_carry(carry.halted, carry)

                carry, _loss, _metrics, preds, all_finish = model(
                    carry=carry,
                    batch=active_batch,
                    return_keys=return_keys,
                )

                current_step = _uniform_step_value(carry.steps)
                if (
                    current_step is not None
                    and current_step in loop_checkpoints
                    and current_step not in captured_loop_checkpoints
                ):
                    if pruned_chunks:
                        checkpoint_preds, checkpoint_steps = _merge_pruned_predictions(
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
                            raise RuntimeError("Model evaluation did not produce predictions for the batch.")
                        checkpoint_preds = preds
                        checkpoint_steps = carry.steps

                    checkpoint_metric_sums = _compute_batch_metric_sums(
                        model=model,
                        batch=batch,
                        preds=checkpoint_preds,
                        final_steps=checkpoint_steps,
                        keep_mask=keep_mask,
                    )
                    for metric_name, metric_value in checkpoint_metric_sums.items():
                        loop_metric_totals_by_step[current_step][set_name][metric_name] += metric_value
                    captured_loop_checkpoints.add(current_step)

                model_halted_mask: Optional[torch.Tensor] = None
                if hasattr(carry, "halted") and carry.halted is not None and not all_finish:
                    model_halted_mask = carry.halted

                act_pruned_mask: Optional[torch.Tensor] = model_halted_mask
                if act_early_stop_enabled and preds is not None and not all_finish:
                    q_halt_logits = preds.get("q_halt_logits")
                    if q_halt_logits is None:
                        raise RuntimeError("ACT early stopping requires q_halt_logits during evaluation.")
                    q_halt_pruned_mask = q_halt_logits > 0
                    act_pruned_mask = (
                        q_halt_pruned_mask
                        if act_pruned_mask is None
                        else (act_pruned_mask | q_halt_pruned_mask)
                    )

                hidden_pruned_mask: Optional[torch.Tensor] = None
                if hidden_pruning_enabled and carry_before_step is not None and not all_finish:
                    hidden_diff_norm = _hidden_diff_norm(
                        carry.current_hidden.detach(),
                        carry_before_step.current_hidden.detach(),
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
                    active_batch = {key: value.index_select(0, keep_indices) for key, value in active_batch.items()}
                    carry = _index_structure(carry, keep_indices)

                if all_finish:
                    break

            if pruned_chunks:
                preds, final_steps = _merge_pruned_predictions(
                    model=model,
                    batch_size=batch_size,
                    return_keys=return_keys,
                    active_indices=active_indices,
                    active_preds=preds,
                    active_steps=carry.steps if preds is not None else None,
                    pruned_chunks=pruned_chunks,
                )
            else:
                if preds is None:
                    raise RuntimeError("Model evaluation did not produce predictions for the batch.")
                final_steps = carry.steps

            batch_metric_sums = _compute_batch_metric_sums(
                model=model,
                batch=batch,
                preds=preds,
                final_steps=final_steps,
                keep_mask=keep_mask,
            )
            selected_final_steps = final_steps[keep_mask]
            if selected_final_steps.numel() > 0:
                unique_steps, counts = torch.unique(selected_final_steps.to(torch.int64), return_counts=True)
                completed_total_counts_by_set[set_name] += float(selected_final_steps.numel())
                for step_value, count_value in zip(unique_steps.tolist(), counts.tolist()):
                    completed_step_counts_by_set[set_name][int(step_value)] += float(count_value)

            processed_problems += int(batch_metric_sums["count"])
            for metric_name, metric_value in batch_metric_sums.items():
                metric_totals_by_set[set_name][metric_name] += metric_value

            if save_predictions:
                for key, value in batch.items():
                    if key in {"inputs", "labels", "puzzle_identifiers", "source_inputs"}:
                        saved_outputs[set_name][key].append(value[keep_mask].detach().cpu())
                for key, value in preds.items():
                    saved_outputs[set_name][key].append(value[keep_mask].detach().cpu())

            if max_problems is not None and processed_problems >= max_problems:
                break

    finalized_by_set, finalized_loop_metrics_by_step, finalized_saved_outputs = _finalize_evaluation_outputs(
        metric_totals_by_set=metric_totals_by_set,
        loop_metric_totals_by_step=loop_metric_totals_by_step,
        completed_step_counts_by_set=completed_step_counts_by_set,
        completed_total_counts_by_set=completed_total_counts_by_set,
        saved_outputs=saved_outputs,
    )

    return (
        finalized_by_set,
        finalized_loop_metrics_by_step,
        finalized_saved_outputs,
        processed_batches,
        processed_problems,
    )


def _evaluate_model_refill(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_problems: Optional[int],
    max_batches: Optional[int],
    save_predictions: bool,
    hidden_diff_threshold: Optional[float],
    loop_checkpoints: Optional[List[int]] = None,
) -> tuple[Dict[str, Dict[str, float]], Dict[int, Dict[str, Dict[str, float]]], Dict[str, Dict[str, torch.Tensor]], int, int]:
    model.eval()

    return_keys: Set[str] = {"preds", "logits"}
    act_early_stop_enabled = _act_early_stop_enabled(model)
    if act_early_stop_enabled:
        return_keys.add("q_halt_logits")
    if save_predictions:
        return_keys.update({"q_halt_logits", "q_continue_logits"})
    loop_checkpoints = sorted(set(loop_checkpoints or []))

    metric_totals_by_set: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    loop_metric_totals_by_step: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    completed_step_counts_by_set: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    completed_total_counts_by_set: Dict[str, float] = defaultdict(float)
    saved_outputs: Dict[str, Dict[str, List[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))

    processed_batches = 0
    processed_problems = 0
    scheduled_problems = 0
    progress_total = _estimate_total_batches(dataloader)
    if progress_total is not None and max_batches is not None:
        progress_total = min(progress_total, max_batches)
    dataset_config = getattr(getattr(dataloader, "dataset", None), "config", None)
    if progress_total is not None and max_problems is not None and dataset_config is not None:
        progress_total = min(progress_total, math.ceil(max_problems / dataset_config.global_batch_size))

    hidden_pruning_enabled = (
        hidden_diff_threshold is not None
        and hidden_diff_threshold > 0
        and _supports_hidden_pruning(model)
    )
    print("hidden_pruning_enabled:", hidden_pruning_enabled, ", hidden_diff_threshold:", hidden_diff_threshold)
    print("ACT early stopping enabled:", act_early_stop_enabled)
    if hidden_diff_threshold is not None and hidden_diff_threshold > 0 and not hidden_pruning_enabled:
        print("Hidden-state pruning is not supported by this model; running standard evaluation.")

    set_name_to_id: Dict[str, int] = {}
    set_names: List[str] = []
    next_saved_order_by_set_id: Dict[int, int] = defaultdict(int)
    pending_chunks: Deque[Dict[str, Any]] = deque()
    dataloader_iter = iter(dataloader)
    source_exhausted = False

    target_batch_size = None
    if dataset_config is not None:
        target_batch_size = int(dataset_config.global_batch_size)

    active_batch: Optional[Dict[str, torch.Tensor]] = None
    active_set_ids = torch.empty((0,), dtype=torch.int64, device=device)
    active_orders = torch.empty((0,), dtype=torch.int64, device=device)
    active_next_checkpoint_indices = torch.empty((0,), dtype=torch.int64, device=device)
    carry: Any = None
    checkpoint_steps_tensor = (
        torch.tensor(loop_checkpoints, dtype=torch.int64, device=device)
        if loop_checkpoints
        else None
    )

    def _get_set_id(set_name: str) -> int:
        set_id = set_name_to_id.get(set_name)
        if set_id is not None:
            return set_id

        set_id = len(set_names)
        set_name_to_id[set_name] = set_id
        set_names.append(set_name)
        return set_id

    def _enqueue_next_source_batch() -> bool:
        nonlocal processed_batches, scheduled_problems, source_exhausted, target_batch_size

        while True:
            if source_exhausted:
                return False
            if max_batches is not None and processed_batches >= max_batches:
                source_exhausted = True
                return False
            if max_problems is not None and scheduled_problems >= max_problems:
                source_exhausted = True
                return False

            try:
                set_name, batch, _global_batch_size = next(dataloader_iter)
            except StopIteration:
                source_exhausted = True
                return False

            processed_batches += 1
            progress_bar.update(1)

            current_batch_size = int(batch["inputs"].shape[0])
            target_batch_size = current_batch_size if target_batch_size is None else max(target_batch_size, current_batch_size)

            valid_example_mask = _valid_example_mask(batch["labels"])
            keep_mask = valid_example_mask.clone()
            if max_problems is not None:
                remaining = max_problems - scheduled_problems
                if remaining <= 0:
                    source_exhausted = True
                    return False
                valid_indices = torch.nonzero(valid_example_mask, as_tuple=False).squeeze(-1)
                if valid_indices.numel() > remaining:
                    keep_mask = torch.zeros_like(valid_example_mask)
                    keep_mask[valid_indices[:remaining]] = True

            keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
            if keep_indices.numel() == 0:
                continue

            kept_batch = {
                key: value.index_select(0, keep_indices).to(device, non_blocking=True)
                for key, value in batch.items()
            }
            set_id = _get_set_id(set_name)
            start_order = next_saved_order_by_set_id[set_id]
            kept_count = int(keep_indices.numel())
            orders = torch.arange(start_order, start_order + kept_count, dtype=torch.int64, device=device)
            next_saved_order_by_set_id[set_id] += kept_count

            pending_chunks.append(
                {
                    "set_id": set_id,
                    "batch": kept_batch,
                    "orders": orders,
                }
            )
            scheduled_problems += kept_count
            if max_problems is not None and scheduled_problems >= max_problems:
                source_exhausted = True
            return True

    def _top_off_active_batch() -> None:
        nonlocal active_batch, active_set_ids, active_orders, active_next_checkpoint_indices, carry

        while True:
            current_active_size = 0 if active_batch is None else int(active_batch["inputs"].shape[0])
            effective_target_size = target_batch_size or current_active_size
            if effective_target_size > 0 and current_active_size >= effective_target_size:
                return

            if not pending_chunks and not _enqueue_next_source_batch():
                return
            if not pending_chunks:
                return

            current_active_size = 0 if active_batch is None else int(active_batch["inputs"].shape[0])
            effective_target_size = target_batch_size or current_active_size or int(pending_chunks[0]["batch"]["inputs"].shape[0])
            slots_to_fill = effective_target_size - current_active_size
            if slots_to_fill <= 0:
                return

            chunk = pending_chunks[0]
            chunk_size = int(chunk["batch"]["inputs"].shape[0])
            take = min(slots_to_fill, chunk_size)
            taken_batch = _slice_tensor_batch(chunk["batch"], 0, take)
            taken_orders = chunk["orders"][:take]
            taken_set_ids = torch.full((take,), chunk["set_id"], dtype=torch.int64, device=device)

            if take == chunk_size:
                pending_chunks.popleft()
            else:
                chunk["batch"] = _slice_tensor_batch(chunk["batch"], take, None)
                chunk["orders"] = chunk["orders"][take:]

            with torch.device(str(device)):
                new_carry = model.initial_carry(taken_batch)  # type: ignore[misc]

            new_next_checkpoint_indices = torch.zeros((take,), dtype=torch.int64, device=device)
            if active_batch is None:
                active_batch = taken_batch
                carry = new_carry
                active_set_ids = taken_set_ids
                active_orders = taken_orders
                active_next_checkpoint_indices = new_next_checkpoint_indices
            else:
                active_batch = _concat_structure(active_batch, taken_batch)
                carry = _concat_structure(carry, new_carry)
                active_set_ids = torch.cat((active_set_ids, taken_set_ids), dim=0)
                active_orders = torch.cat((active_orders, taken_orders), dim=0)
                active_next_checkpoint_indices = torch.cat(
                    (active_next_checkpoint_indices, new_next_checkpoint_indices),
                    dim=0,
                )

    def _record_loop_checkpoint_hits(preds: Dict[str, torch.Tensor], current_steps: torch.Tensor) -> None:
        nonlocal active_next_checkpoint_indices

        if active_batch is None or checkpoint_steps_tensor is None or active_next_checkpoint_indices.numel() == 0:
            return

        eligible_mask = active_next_checkpoint_indices < checkpoint_steps_tensor.numel()
        if not torch.any(eligible_mask):
            return

        expected_steps = checkpoint_steps_tensor.index_select(
            0,
            active_next_checkpoint_indices.clamp(max=checkpoint_steps_tensor.numel() - 1),
        )
        hit_mask = eligible_mask & (current_steps.to(torch.int64) == expected_steps)
        if not torch.any(hit_mask):
            return

        hit_checkpoint_indices = torch.unique(active_next_checkpoint_indices[hit_mask], sorted=True).tolist()
        for checkpoint_index in hit_checkpoint_indices:
            checkpoint_mask = hit_mask & (active_next_checkpoint_indices == checkpoint_index)
            checkpoint_step = loop_checkpoints[checkpoint_index]
            checkpoint_batch = {
                key: value[checkpoint_mask]
                for key, value in active_batch.items()
            }
            checkpoint_preds = {
                key: value[checkpoint_mask]
                for key, value in preds.items()
            }
            checkpoint_steps = current_steps[checkpoint_mask]
            checkpoint_set_ids = active_set_ids[checkpoint_mask]
            _accumulate_metric_sums_by_set_ids(
                metric_totals_by_set=loop_metric_totals_by_step[checkpoint_step],
                model=model,
                batch=checkpoint_batch,
                preds=checkpoint_preds,
                final_steps=checkpoint_steps,
                set_ids=checkpoint_set_ids,
                set_names=set_names,
            )

        active_next_checkpoint_indices = active_next_checkpoint_indices + hit_mask.to(torch.int64)

    def _finalize_examples(
        finalize_mask: torch.Tensor,
        preds: Dict[str, torch.Tensor],
        final_steps: torch.Tensor,
    ) -> None:
        nonlocal processed_problems

        if active_batch is None or not torch.any(finalize_mask):
            return

        finalized_batch = {
            key: value[finalize_mask]
            for key, value in active_batch.items()
        }
        finalized_preds = {
            key: value[finalize_mask]
            for key, value in preds.items()
        }
        finalized_steps = final_steps[finalize_mask]
        finalized_set_ids = active_set_ids[finalize_mask]
        finalized_orders = active_orders[finalize_mask]
        finalized_next_checkpoint_indices = active_next_checkpoint_indices[finalize_mask]

        _accumulate_metric_sums_by_set_ids(
            metric_totals_by_set=metric_totals_by_set,
            model=model,
            batch=finalized_batch,
            preds=finalized_preds,
            final_steps=finalized_steps,
            set_ids=finalized_set_ids,
            set_names=set_names,
        )
        _accumulate_completed_steps_by_set_ids(
            completed_step_counts_by_set=completed_step_counts_by_set,
            completed_total_counts_by_set=completed_total_counts_by_set,
            final_steps=finalized_steps,
            set_ids=finalized_set_ids,
            set_names=set_names,
        )
        if save_predictions:
            _append_saved_output_records(
                saved_outputs=saved_outputs,
                batch=finalized_batch,
                preds=finalized_preds,
                set_ids=finalized_set_ids,
                set_names=set_names,
                orders=finalized_orders,
            )

        for checkpoint_index, checkpoint_step in enumerate(loop_checkpoints):
            remaining_checkpoint_mask = finalized_next_checkpoint_indices <= checkpoint_index
            if not torch.any(remaining_checkpoint_mask):
                continue

            checkpoint_batch = {
                key: value[remaining_checkpoint_mask]
                for key, value in finalized_batch.items()
            }
            checkpoint_preds = {
                key: value[remaining_checkpoint_mask]
                for key, value in finalized_preds.items()
            }
            checkpoint_steps = finalized_steps[remaining_checkpoint_mask]
            checkpoint_set_ids = finalized_set_ids[remaining_checkpoint_mask]
            _accumulate_metric_sums_by_set_ids(
                metric_totals_by_set=loop_metric_totals_by_step[checkpoint_step],
                model=model,
                batch=checkpoint_batch,
                preds=checkpoint_preds,
                final_steps=checkpoint_steps,
                set_ids=checkpoint_set_ids,
                set_names=set_names,
            )

        processed_problems += int(finalized_steps.numel())

    progress_bar = tqdm(total=progress_total, desc="Evaluating")
    try:
        with torch.inference_mode():
            while True:
                _top_off_active_batch()
                if active_batch is None:
                    break

                carry_before_step = None
                if hidden_pruning_enabled and hasattr(carry, "current_hidden"):
                    carry_before_step = carry

                carry, _loss, _metrics, preds, all_finish = model(
                    carry=carry,
                    batch=active_batch,
                    return_keys=return_keys,
                )
                if preds is None:
                    raise RuntimeError("Model evaluation did not produce predictions for the active batch.")

                _record_loop_checkpoint_hits(preds=preds, current_steps=carry.steps)

                model_halted_mask: Optional[torch.Tensor] = None
                if hasattr(carry, "halted") and carry.halted is not None and not all_finish:
                    model_halted_mask = carry.halted

                act_pruned_mask: Optional[torch.Tensor] = model_halted_mask
                if act_early_stop_enabled and not all_finish:
                    q_halt_logits = preds.get("q_halt_logits")
                    if q_halt_logits is None:
                        raise RuntimeError("ACT early stopping requires q_halt_logits during evaluation.")
                    q_halt_pruned_mask = q_halt_logits > 0
                    act_pruned_mask = (
                        q_halt_pruned_mask
                        if act_pruned_mask is None
                        else (act_pruned_mask | q_halt_pruned_mask)
                    )

                hidden_pruned_mask: Optional[torch.Tensor] = None
                if hidden_pruning_enabled and carry_before_step is not None and not all_finish:
                    hidden_diff_norm = _hidden_diff_norm(
                        carry.current_hidden.detach(),
                        carry_before_step.current_hidden.detach(),
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
                    _finalize_examples(
                        finalize_mask=prune_mask,
                        preds=preds,
                        final_steps=carry.steps,
                    )

                    keep_indices = torch.nonzero(~prune_mask, as_tuple=False).squeeze(-1)
                    if keep_indices.numel() == 0:
                        active_batch = None
                        active_set_ids = active_set_ids[:0]
                        active_orders = active_orders[:0]
                        active_next_checkpoint_indices = active_next_checkpoint_indices[:0]
                        carry = None
                    else:
                        active_batch = {
                            key: value.index_select(0, keep_indices)
                            for key, value in active_batch.items()
                        }
                        carry = _index_structure(carry, keep_indices)
                        active_set_ids = active_set_ids.index_select(0, keep_indices)
                        active_orders = active_orders.index_select(0, keep_indices)
                        active_next_checkpoint_indices = active_next_checkpoint_indices.index_select(0, keep_indices)
                    continue

                if all_finish:
                    _finalize_examples(
                        finalize_mask=torch.ones_like(carry.steps, dtype=torch.bool),
                        preds=preds,
                        final_steps=carry.steps,
                    )
                    active_batch = None
                    active_set_ids = active_set_ids[:0]
                    active_orders = active_orders[:0]
                    active_next_checkpoint_indices = active_next_checkpoint_indices[:0]
                    carry = None
    finally:
        progress_bar.close()

    finalized_by_set, finalized_loop_metrics_by_step, finalized_saved_outputs = _finalize_evaluation_outputs(
        metric_totals_by_set=metric_totals_by_set,
        loop_metric_totals_by_step=loop_metric_totals_by_step,
        completed_step_counts_by_set=completed_step_counts_by_set,
        completed_total_counts_by_set=completed_total_counts_by_set,
        saved_outputs=saved_outputs,
    )

    return (
        finalized_by_set,
        finalized_loop_metrics_by_step,
        finalized_saved_outputs,
        processed_batches,
        processed_problems,
    )


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_problems: Optional[int],
    max_batches: Optional[int],
    save_predictions: bool,
    hidden_diff_threshold: Optional[float],
    loop_checkpoints: Optional[List[int]] = None,
    active_batch_strategy: str = "refill",
) -> tuple[Dict[str, Dict[str, float]], Dict[int, Dict[str, Dict[str, float]]], Dict[str, Dict[str, torch.Tensor]], int, int]:
    if active_batch_strategy == "refill":
        return _evaluate_model_refill(
            model=model,
            dataloader=dataloader,
            device=device,
            max_problems=max_problems,
            max_batches=max_batches,
            save_predictions=save_predictions,
            hidden_diff_threshold=hidden_diff_threshold,
            loop_checkpoints=loop_checkpoints,
        )
    if active_batch_strategy == "shrink":
        return _evaluate_model_shrink(
            model=model,
            dataloader=dataloader,
            device=device,
            max_problems=max_problems,
            max_batches=max_batches,
            save_predictions=save_predictions,
            hidden_diff_threshold=hidden_diff_threshold,
            loop_checkpoints=loop_checkpoints,
        )
    raise ValueError(f"Unknown active batch strategy: {active_batch_strategy}")


def _default_output_path(
    checkpoint_path: str,
    split: str,
    max_batches: Optional[int],
    max_problems: Optional[int],
    loops: Optional[int],
    h_cycles: Optional[int],
    l_cycles: Optional[int],
    hidden_diff_threshold: Optional[float],
    active_batch_strategy: str,
) -> Path:
    threshold_suffix = ""
    if hidden_diff_threshold is not None and hidden_diff_threshold > 0:
        threshold_value = f"{hidden_diff_threshold:g}".replace("-", "m").replace(".", "_")
        threshold_suffix = f"_hidden_diff_threshold_{threshold_value}"
    strategy_suffix = f"_active_batch_strategy_{active_batch_strategy}"

    resolved_path = _resolve_checkpoint_path(checkpoint_path)
    if resolved_path is None:
        stem = f"evaluation_results_{split}"
        if max_batches is not None:
            stem += f"_max_batches_{max_batches}"
        if max_problems is not None:
            stem += f"_max_problems_{max_problems}"
        if loops is not None:
            stem += f"_loops_{loops}"
        if h_cycles is not None:
            stem += f"_H_cycles_{h_cycles}"
        if l_cycles is not None:
            stem += f"_L_cycles_{l_cycles}"
        stem += threshold_suffix
        stem += strategy_suffix
        return Path(f"{stem}.json")
    checkpoint_file = Path(resolved_path)
    stem = f"{checkpoint_file.stem}_evaluation_{split}"
    if max_batches is not None:
        stem += f"_max_batches_{max_batches}"
    if max_problems is not None:
        stem += f"_max_problems_{max_problems}"
    if loops is not None:
        stem += f"_loops_{loops}"
    if h_cycles is not None:
        stem += f"_H_cycles_{h_cycles}"
    if l_cycles is not None:
        stem += f"_L_cycles_{l_cycles}"
    stem += threshold_suffix
    stem += strategy_suffix
    return checkpoint_file.parent / f"{stem}.json"


def _sidecar_predictions_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_outputs.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on a dataset split.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/URM-sudoku-base-good",
        help="Checkpoint file or checkpoint directory.",
    )
    parser.add_argument("--data_path", type=str, default=None, help="Override dataset path from the checkpoint config.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override the global batch size for evaluation.")
    parser.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help="Evaluate at most this many problems. Stops exactly at the requested number.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the evaluation summary as JSON.",
    )
    parser.add_argument("--max_batches", type=int, default=None, help="Stop after this many batches for debugging.")
    parser.add_argument(
        "--loops",
        type=int,
        default=None,
        help="Override the model loop count used while solving each problem.",
    )
    parser.add_argument(
        "--H_cycles",
        type=int,
        default=None,
        help="Override H_cycles for evaluation.",
    )
    parser.add_argument(
        "--L_cycles",
        type=int,
        default=None,
        help="Override L_cycles for evaluation.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use, for example 'cuda' or 'cpu'.")
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Also save inputs, labels, preds, logits, and halt logits to a sidecar .pt file.",
    )
    parser.add_argument(
        "--hidden_diff_threshold",
        type=float,
        default=None,
        help="Prune examples from the carry during evaluation when the relative hidden-state change is at or below this threshold.",
    )
    parser.add_argument(
        "--active_batch_strategy",
        type=str,
        default="refill",
        choices=("refill", "shrink"),
        help="How to manage active evaluation slots: keep refilling finished slots, or preserve the legacy shrinking batch.",
    )
    parser.add_argument(
        "--no_strict_load",
        action="store_true",
        help="Allow missing or unexpected keys when loading the checkpoint.",
    )
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    resolved_checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    if resolved_checkpoint_path is None:
        raise FileNotFoundError(f"Could not resolve checkpoint path from: {args.checkpoint}")

    print(f"Loading config from: {args.checkpoint}")
    config = load_config_from_checkpoint_path(args.checkpoint)
    if args.data_path is not None:
        config.data_path = args.data_path
    if args.batch_size is not None:
        config.global_batch_size = args.batch_size
    loop_override_key: Optional[str] = None
    if args.loops is not None:
        loop_override_key = _set_effective_loop_count(config, args.loops)
    if args.H_cycles is not None:
        if config.arch.__pydantic_extra__ is None:
            config.arch.__pydantic_extra__ = {}
        config.arch.__pydantic_extra__["H_cycles"] = args.H_cycles
    if args.L_cycles is not None:
        if config.arch.__pydantic_extra__ is None:
            config.arch.__pydantic_extra__ = {}
        config.arch.__pydantic_extra__["L_cycles"] = args.L_cycles

    print(f"Using device: {device}")
    print(f"Using dataset: {config.data_path} ({args.split})")
    print(f"Using batch size: {config.global_batch_size}")
    if args.max_problems is not None:
        print(f"Evaluating at most {args.max_problems} problems")
    if args.loops is not None:
        if loop_override_key == "halt_max_steps":
            print(f"Overriding loops to: {args.loops} (via halt_max_steps)")
        else:
            print(f"Overriding loops to: {args.loops}")
    if args.H_cycles is not None:
        print(f"Overriding H_cycles to: {args.H_cycles}")
    if args.L_cycles is not None:
        print(f"Overriding L_cycles to: {args.L_cycles}")
    if args.hidden_diff_threshold is not None and args.hidden_diff_threshold > 0:
        print(f"Using hidden diff pruning threshold: {args.hidden_diff_threshold}")
    print(f"Using active batch strategy: {args.active_batch_strategy}")

    dataloader = create_test_dataloader(config, args.split, config.global_batch_size)
    metadata = dataloader.dataset.metadata
    effective_loops = _get_effective_loop_count(config)
    loop_checkpoints = _power_of_two_loop_checkpoints(effective_loops)
    if loop_checkpoints:
        print(f"Reporting intermediate metrics at power-of-two loops: {loop_checkpoints}")

    print("Reconstructing model...")
    model = create_model_for_evaluation(config, metadata, device=device)

    print(f"Loading weights from: {resolved_checkpoint_path}")
    step = load_model_weights(
        model,
        resolved_checkpoint_path,
        device=device,
        strict=not args.no_strict_load,
    )
    start_time = time.time()
    metrics_by_set, loop_metrics_by_step, saved_outputs, processed_batches, processed_problems = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        max_problems=args.max_problems,
        max_batches=args.max_batches,
        save_predictions=args.save_predictions,
        hidden_diff_threshold=args.hidden_diff_threshold,
        loop_checkpoints=loop_checkpoints,
        active_batch_strategy=args.active_batch_strategy,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("")
    print(f"Evaluation completed in {elapsed_time:.2f} seconds.")
    print(f"Processed batches: {processed_batches}")
    print(f"Processed problems: {processed_problems}")
    if step is not None:
        print(f"Checkpoint step: {step}")
    for set_name, metrics in metrics_by_set.items():
        print(f"[{set_name}]")
        for key, value in sorted(metrics.items()):
            print(f"  {key}: {value:.6f}")

    if loop_metrics_by_step:
        print("")
        print("Intermediate metrics at power-of-two loops:")
        for loop_step, metrics_per_set in loop_metrics_by_step.items():
            print(f"[loop={loop_step}]")
            for set_name, metrics in metrics_per_set.items():
                print(f"  [{set_name}]")
                for key in ("accuracy", "exact_accuracy", "lm_loss"):
                    if key in metrics:
                        print(f"    {key}: {metrics[key]:.6f}")
                if "finished_fraction" in metrics:
                    print(f"    finished_pct: {metrics['finished_fraction'] * 100:.2f}%")

    output_path = (
        Path(args.output)
        if args.output is not None
        else _default_output_path(
            args.checkpoint,
            args.split,
            args.max_batches,
            args.max_problems,
            args.loops,
            args.H_cycles,
            args.L_cycles,
            args.hidden_diff_threshold,
            args.active_batch_strategy,
        )
    )
    if output_path.suffix.lower() != ".json":
        print(f"Warning: writing JSON summary to a path without a .json suffix: {output_path}")

    payload = {
        "checkpoint": resolved_checkpoint_path,
        "checkpoint_step": step,
        "data_path": config.data_path,
        "split": args.split,
        "batch_size": config.global_batch_size,
        "loops": effective_loops,
        "H_cycles": (config.arch.__pydantic_extra__ or {}).get("H_cycles"),
        "L_cycles": (config.arch.__pydantic_extra__ or {}).get("L_cycles"),
        "hidden_diff_threshold": args.hidden_diff_threshold,
        "active_batch_strategy": args.active_batch_strategy,
        "max_batches": args.max_batches,
        "max_problems": args.max_problems,
        "elapsed_time_sec": elapsed_time,
        "processed_batches": processed_batches,
        "processed_problems": processed_problems,
        "metrics": metrics_by_set,
        "power_of_two_loop_metrics": loop_metrics_by_step,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved evaluation summary to: {output_path}")
    if args.save_predictions:
        predictions_output_path = _sidecar_predictions_output_path(output_path)
        torch.save(saved_outputs, predictions_output_path)
        print(f"Saved prediction tensors to: {predictions_output_path}")


if __name__ == "__main__":
    main()
