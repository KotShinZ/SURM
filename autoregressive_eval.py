from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from data.common import PuzzleDatasetMetadata
from models.losses import IGNORE_LABEL_ID


def _unwrap_base_model(model: Any) -> Any:
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return getattr(model, "model", model)


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _sample_ranges(batch: Dict[str, torch.Tensor], sample_idx: int) -> Tuple[int, int, int, int]:
    seq_offsets = batch["seq_offsets"]
    label_offsets = batch.get("label_seq_offsets", seq_offsets)
    return (
        int(seq_offsets[sample_idx].item()),
        int(seq_offsets[sample_idx + 1].item()),
        int(label_offsets[sample_idx].item()),
        int(label_offsets[sample_idx + 1].item()),
    )


def _target_tokens_and_positions(
    batch: Dict[str, torch.Tensor],
    sample_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start, end, label_start, label_end = _sample_ranges(batch, sample_idx)
    sample_answer_mask = batch["answer_mask"][start:end].to(torch.bool)
    sample_positions = batch["position_ids"][start:end]
    target_positions = sample_positions[sample_answer_mask]

    labels = batch.get("labels")
    if labels is None:
        target_tokens = batch.get("source_inputs", batch["inputs"])[start:end][sample_answer_mask]
    elif labels.ndim == 1 and labels.numel() != batch["inputs"].numel():
        target_tokens = labels[label_start:label_end]
    else:
        target_tokens = labels[start:end][sample_answer_mask]

    return target_tokens, target_positions, sample_answer_mask


def _offsets(length: int, *, device: torch.device) -> torch.Tensor:
    return torch.tensor([0, length], dtype=torch.int32, device=device)


def _single_generation_batch(
    batch: Dict[str, torch.Tensor],
    sample_idx: int,
    generated_tokens: torch.Tensor,
    target_positions: torch.Tensor,
    sample_answer_mask: torch.Tensor,
    start_token_id: int,
) -> Dict[str, torch.Tensor]:
    start, end, _label_start, _label_end = _sample_ranges(batch, sample_idx)
    sample_inputs = batch["inputs"][start:end]
    sample_source_inputs = batch.get("source_inputs", batch["inputs"])[start:end]
    sample_positions = batch["position_ids"][start:end]
    context_mask = ~sample_answer_mask

    answer_len = int(generated_tokens.numel()) + 1
    answer_input = torch.empty((answer_len,), dtype=sample_inputs.dtype, device=sample_inputs.device)
    answer_input[0] = int(start_token_id)
    if generated_tokens.numel() > 0:
        answer_input[1:] = generated_tokens.to(dtype=sample_inputs.dtype)

    answer_positions = target_positions[:answer_len]
    inputs = torch.cat([sample_inputs[context_mask], answer_input], dim=0)
    source_inputs = torch.cat([sample_source_inputs[context_mask], answer_input], dim=0)
    position_ids = torch.cat([sample_positions[context_mask], answer_positions], dim=0)
    answer_mask = torch.cat(
        [
            torch.zeros((int(context_mask.sum().item()),), dtype=torch.bool, device=sample_inputs.device),
            torch.ones((answer_len,), dtype=torch.bool, device=sample_inputs.device),
        ],
        dim=0,
    )

    gen_batch: Dict[str, torch.Tensor] = {
        "inputs": inputs,
        "labels": torch.full((answer_len,), IGNORE_LABEL_ID, dtype=torch.int32, device=sample_inputs.device),
        "answer_mask": answer_mask,
        "source_inputs": source_inputs,
        "position_ids": position_ids,
        "seq_lengths": torch.tensor([inputs.numel()], dtype=torch.int32, device=sample_inputs.device),
        "seq_offsets": _offsets(inputs.numel(), device=sample_inputs.device),
        "label_seq_lengths": torch.tensor([answer_len], dtype=torch.int32, device=sample_inputs.device),
        "label_seq_offsets": _offsets(answer_len, device=sample_inputs.device),
        "puzzle_identifiers": batch["puzzle_identifiers"][sample_idx : sample_idx + 1],
    }
    if "arc_identifiers" in batch:
        gen_batch["arc_identifiers"] = batch["arc_identifiers"][sample_idx : sample_idx + 1]
    return gen_batch


def _final_prediction_batch(
    batch: Dict[str, torch.Tensor],
    sample_idx: int,
    generated_tokens: torch.Tensor,
    target_positions: torch.Tensor,
    sample_answer_mask: torch.Tensor,
) -> Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, torch.Tensor]]:
    start, end, _label_start, _label_end = _sample_ranges(batch, sample_idx)
    sample_inputs = batch["inputs"][start:end]
    sample_source_inputs = batch.get("source_inputs", batch["inputs"])[start:end]
    sample_positions = batch["position_ids"][start:end]
    context_mask = ~sample_answer_mask
    answer_positions = target_positions[: generated_tokens.numel()]

    inputs = torch.cat([sample_inputs[context_mask], generated_tokens.to(sample_inputs.dtype)], dim=0)
    source_inputs = torch.cat(
        [sample_source_inputs[context_mask], generated_tokens.to(sample_source_inputs.dtype)],
        dim=0,
    )
    position_ids = torch.cat([sample_positions[context_mask], answer_positions], dim=0)
    answer_mask = torch.cat(
        [
            torch.zeros((int(context_mask.sum().item()),), dtype=torch.bool, device=sample_inputs.device),
            torch.ones((generated_tokens.numel(),), dtype=torch.bool, device=sample_inputs.device),
        ],
        dim=0,
    )
    preds = torch.cat(
        [
            torch.zeros((int(context_mask.sum().item()),), dtype=torch.long, device=sample_inputs.device),
            generated_tokens.to(torch.long),
        ],
        dim=0,
    )

    final_batch: Dict[str, Optional[torch.Tensor]] = {
        "inputs": inputs,
        "labels": None,
        "answer_mask": answer_mask,
        "source_inputs": source_inputs,
        "position_ids": position_ids,
        "seq_lengths": torch.tensor([inputs.numel()], dtype=torch.int32, device=sample_inputs.device),
        "seq_offsets": _offsets(inputs.numel(), device=sample_inputs.device),
        "puzzle_identifiers": batch["puzzle_identifiers"][sample_idx : sample_idx + 1],
    }
    if "arc_identifiers" in batch:
        final_batch["arc_identifiers"] = batch["arc_identifiers"][sample_idx : sample_idx + 1]

    return final_batch, {"preds": preds}


def _run_base_model_to_halt(base_model: Any, batch: Dict[str, torch.Tensor]) -> Tuple[Any, Dict[str, torch.Tensor]]:
    carry = base_model.initial_carry(batch)
    max_steps = int(getattr(getattr(base_model, "config", None), "loops", 1)) + 1
    outputs: Dict[str, torch.Tensor] = {}
    for _ in range(max_steps):
        carry, outputs = base_model(carry=carry, batch=batch)
        if bool(carry.halted.all().item()):
            break
    return carry, outputs


def _generate_sample(
    base_model: Any,
    batch: Dict[str, torch.Tensor],
    sample_idx: int,
    *,
    start_token_id: int,
) -> Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, torch.Tensor], Dict[str, float]]:
    target_tokens, target_positions, sample_answer_mask = _target_tokens_and_positions(batch, sample_idx)
    device = batch["inputs"].device
    generated = torch.empty((0,), dtype=torch.long, device=device)
    final_carry = None
    final_outputs: Dict[str, torch.Tensor] = {}

    for _ in range(int(target_tokens.numel())):
        gen_batch = _single_generation_batch(
            batch,
            sample_idx,
            generated,
            target_positions,
            sample_answer_mask,
            start_token_id,
        )
        final_carry, final_outputs = _run_base_model_to_halt(base_model, gen_batch)
        logits = final_outputs["logits"]
        next_token = torch.argmax(logits[-1], dim=-1).view(1)
        generated = torch.cat([generated, next_token.to(torch.long)], dim=0)

    final_batch, preds = _final_prediction_batch(
        batch,
        sample_idx,
        generated,
        target_positions,
        sample_answer_mask,
    )
    q_halt_logits = final_outputs.get("q_halt_logits")
    if q_halt_logits is None:
        q_halt_logits = torch.zeros((1,), dtype=torch.float32, device=device)
    preds["q_halt_logits"] = q_halt_logits

    valid = target_tokens != IGNORE_LABEL_ID
    correct = (generated.to(target_tokens.dtype) == target_tokens) & valid
    valid_count = int(valid.sum().item())
    exact = bool(valid_count > 0 and correct.sum().item() == valid_count)
    steps = 0.0
    if final_carry is not None and final_carry.steps is not None:
        steps = float(final_carry.steps[0].item())

    metrics = {
        "count": 1.0 if valid_count > 0 else 0.0,
        "accuracy": float(correct.sum().item()) / max(valid_count, 1),
        "exact_accuracy": 1.0 if exact else 0.0,
        "steps": steps if valid_count > 0 else 0.0,
    }
    return final_batch, preds, metrics


def evaluate_autoregressive(
    config: Any,
    train_state: Any,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
    early_eval: bool = False,
):
    device = torch.device("cuda")
    base_model = _unwrap_base_model(train_state.model)
    start_token_id = int(getattr(config.arch, "causal_lm_start_token_id", 1))

    for evaluator in evaluators:
        evaluator.begin_eval()

    set_ids = {name: idx for idx, name in enumerate(eval_metadata.sets)}
    metric_keys = ["count", "accuracy", "exact_accuracy", "steps"]
    metric_values = torch.zeros((len(set_ids), len(metric_keys)), dtype=torch.float64, device=device)

    processed_batches = 0
    print("Starting autoregressive evaluation... len(eval_loader) =", len(eval_loader))
    with torch.inference_mode():
        for set_name, batch, _global_batch_size in eval_loader:
            if early_eval and processed_batches > 50:
                break
            processed_batches += 1
            if rank == 0:
                print(f"Processing autoregressive batch {processed_batches}: {set_name}")

            batch = _to_device(batch, device)
            sample_count = int(batch["puzzle_identifiers"].shape[0])
            set_id = set_ids[set_name]

            for sample_idx in range(sample_count):
                final_batch, preds, sample_metrics = _generate_sample(
                    base_model,
                    batch,
                    sample_idx,
                    start_token_id=start_token_id,
                )
                for evaluator in evaluators:
                    evaluator.update_batch(final_batch, preds)
                metric_values[set_id] += torch.tensor(
                    [sample_metrics[key] for key in metric_keys],
                    dtype=torch.float64,
                    device=device,
                )

    if world_size > 1:
        dist.reduce(metric_values, dst=0)

    reduced_metrics = None
    if rank == 0:
        reduced_metrics = {}
        values = metric_values.cpu()
        for set_name, set_id in set_ids.items():
            count = float(values[set_id, 0].item())
            if count <= 0:
                reduced_metrics[set_name] = {key: 0.0 for key in metric_keys[1:]}
            else:
                reduced_metrics[set_name] = {
                    key: float(values[set_id, idx].item()) / count
                    for idx, key in enumerate(metric_keys[1:], start=1)
                }

    if rank == 0:
        print(f"\nRunning {len(evaluators)} autoregressive evaluator(s)...")

    for i, evaluator in enumerate(evaluators):
        if rank == 0:
            print(f"Running evaluator {i + 1}/{len(evaluators)}: {evaluator.__class__.__name__}")

        evaluator_save_path = None
        if config.checkpoint_path is not None:
            evaluator_save_path = os.path.join(
                config.checkpoint_path,
                f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
            )
            os.makedirs(evaluator_save_path, exist_ok=True)

        metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
        if rank == 0 and metrics is not None:
            if reduced_metrics is None:
                reduced_metrics = {}
            reduced_metrics.update(metrics)
            print(f"  Completed {evaluator.__class__.__name__}")

    if rank == 0:
        print("All autoregressive evaluators completed!")
    return reduced_metrics
