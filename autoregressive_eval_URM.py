from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from data.common import PuzzleDatasetMetadata
from models.urm.urm import URMInferenceCache


ARC_NEWLINE_TOKEN_ID = 1
DEFAULT_CASUAL_MAX_NEW_TOKENS = 931
DEFAULT_CACHE_CHUNK_SIZE = 64


def _unwrap_base_model(model: Any) -> Any:
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return getattr(model, "model", model)


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _resolve_max_new_tokens(config: Any) -> int:
    value = getattr(config, "autoregressive_eval_max_new_tokens", None)
    if value is None:
        value = getattr(getattr(config, "arch", None), "autoregressive_eval_max_new_tokens", None)
    if value is None:
        return DEFAULT_CASUAL_MAX_NEW_TOKENS
    return max(1, int(value))


def _resolve_cache_chunk_size(config: Any) -> int:
    value = getattr(config, "autoregressive_eval_cache_chunk_size", None)
    if value is None:
        value = getattr(getattr(config, "arch", None), "autoregressive_eval_cache_chunk_size", None)
    if value is None:
        return DEFAULT_CACHE_CHUNK_SIZE
    return max(1, int(value))


def _resolve_casual_token_ids(
    config: Any,
    base_model: Any,
    eval_metadata: PuzzleDatasetMetadata,
) -> Tuple[int, int]:
    arch_config = getattr(config, "arch", None)
    model_config = getattr(base_model, "config", None)
    vocab_size = int(getattr(model_config, "vocab_size", eval_metadata.vocab_size))

    configured_start_token_id = getattr(arch_config, "causal_lm_start_token_id", None)
    if configured_start_token_id is None:
        configured_start_token_id = getattr(model_config, "causal_lm_start_token_id", None)

    configured_end_token_id = getattr(arch_config, "casual_lm_end_token_id", None)
    if configured_end_token_id is None:
        configured_end_token_id = getattr(model_config, "casual_lm_end_token_id", None)

    if configured_start_token_id is None and configured_end_token_id is None:
        end_token_id = vocab_size - 2
        start_token_id = vocab_size - 1
    elif configured_start_token_id is None:
        end_token_id = int(configured_end_token_id)
        start_token_id = end_token_id + 1
    elif configured_end_token_id is None:
        start_token_id = int(configured_start_token_id)
        end_token_id = start_token_id - 1
    else:
        start_token_id = int(configured_start_token_id)
        end_token_id = int(configured_end_token_id)

    if end_token_id < 0 or start_token_id < 0 or end_token_id >= vocab_size or start_token_id >= vocab_size:
        raise ValueError(
            "Resolved casual token ids are outside the model vocabulary: "
            f"START={start_token_id}, END={end_token_id}, vocab_size={vocab_size}."
        )
    return start_token_id, end_token_id


def _sample_count(batch: Dict[str, torch.Tensor]) -> int:
    if "puzzle_identifiers" in batch:
        return int(batch["puzzle_identifiers"].shape[0])
    return int(batch["inputs"].shape[0])


def _position_dim_for_model(base_model: Any, batch: Dict[str, torch.Tensor]) -> int:
    prompt_positions = batch.get("prompt_position_ids")
    if prompt_positions is not None and prompt_positions.ndim >= 2:
        return int(prompt_positions.shape[-1])

    config = getattr(base_model, "config", None)
    grid_depth = int(getattr(config, "grid_depth", 0))
    grid_io = int(getattr(config, "grid_io", 0))
    grid_height = int(getattr(config, "grid_height", 0))
    grid_width = int(getattr(config, "grid_width", 0))
    if grid_depth > 0 and grid_io > 0 and grid_height > 0 and grid_width > 0:
        return 4
    if grid_depth > 0 and grid_height > 0 and grid_width > 0:
        return 3
    if grid_height > 0 and grid_width > 0:
        return 2
    return 4


def _advance_positions(
    positions: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    newline_token_id: int,
    end_token_id: int,
    grid_height: int,
    grid_width: int,
    row_widths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    next_positions = positions.clone()
    if positions.shape[-1] < 2:
        return next_positions

    rows = positions[:, -2]
    cols = positions[:, -1]
    is_newline = token_ids == int(newline_token_id)
    is_end = token_ids == int(end_token_id)

    if row_widths is None:
        next_rows = torch.where(is_newline, rows + 1, rows)
        next_cols = torch.where(is_newline, torch.zeros_like(cols), cols + 1)
    else:
        widths = row_widths.to(device=positions.device, dtype=cols.dtype)
        has_width = widths > 0

        row_major_cols = cols + 1
        row_major_wrap = row_major_cols >= widths.clamp_min(1)
        row_major_rows = torch.where(row_major_wrap, rows + 1, rows)
        row_major_cols = torch.where(row_major_wrap, torch.zeros_like(cols), row_major_cols)

        token_rows = torch.where(is_newline, rows + 1, rows)
        token_cols = torch.where(is_newline, torch.zeros_like(cols), cols + 1)
        next_rows = torch.where(has_width, row_major_rows, token_rows)
        next_cols = torch.where(has_width, row_major_cols, token_cols)

    next_rows = torch.where(is_end, rows, next_rows).clamp(min=0, max=max(grid_height - 1, 0))
    next_cols = torch.where(is_end, cols, next_cols).clamp(min=0, max=max(grid_width - 1, 0))
    next_positions[:, -2] = next_rows
    next_positions[:, -1] = next_cols
    return next_positions


def _synthesize_positions(
    tokens: torch.Tensor,
    *,
    position_dim: int,
    newline_token_id: int,
) -> torch.Tensor:
    positions = torch.zeros((int(tokens.numel()), position_dim), dtype=torch.int32, device=tokens.device)
    row = 0
    col = 0
    for idx, token in enumerate(tokens.to(torch.long).tolist()):
        if position_dim >= 2:
            positions[idx, -2] = row
            positions[idx, -1] = col
        if int(token) == int(newline_token_id):
            row += 1
            col = 0
        else:
            col += 1
    return positions


def _prepare_prompt_batch(
    batch: Dict[str, torch.Tensor],
    base_model: Any,
    *,
    newline_token_id: int,
) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    sample_count = _sample_count(batch)
    device = batch["inputs"].device
    position_dim = _position_dim_for_model(base_model, batch)
    source_inputs_all = batch.get("source_inputs", batch["inputs"])
    prompt_positions_all = batch.get("prompt_position_ids")

    input_chunks: List[torch.Tensor] = []
    source_chunks: List[torch.Tensor] = []
    position_chunks: List[torch.Tensor] = []
    final_prompt_positions: List[torch.Tensor] = []
    seq_lengths: List[int] = []

    if "seq_offsets" in batch:
        seq_offsets = batch["seq_offsets"].to(torch.long)
        for sample_idx in range(sample_count):
            start = int(seq_offsets[sample_idx].item())
            end = int(seq_offsets[sample_idx + 1].item())
            prompt_inputs = batch["inputs"][start:end]
            prompt_source_inputs = source_inputs_all[start:end]
            if prompt_positions_all is not None:
                prompt_positions = prompt_positions_all[start:end].to(device=device, dtype=torch.int32)
            else:
                prompt_positions = _synthesize_positions(
                    prompt_inputs,
                    position_dim=position_dim,
                    newline_token_id=newline_token_id,
                )
            input_chunks.append(prompt_inputs)
            source_chunks.append(prompt_source_inputs)
            position_chunks.append(prompt_positions)
            final_prompt_positions.append(prompt_positions)
            seq_lengths.append(int(prompt_inputs.numel()))
    else:
        inputs = batch["inputs"]
        source_inputs = source_inputs_all
        for sample_idx in range(sample_count):
            prompt_inputs = inputs[sample_idx].reshape(-1)
            prompt_source_inputs = source_inputs[sample_idx].reshape(-1)
            prompt_positions = _synthesize_positions(
                prompt_inputs,
                position_dim=position_dim,
                newline_token_id=newline_token_id,
            )
            input_chunks.append(prompt_inputs)
            source_chunks.append(prompt_source_inputs)
            position_chunks.append(prompt_positions)
            final_prompt_positions.append(prompt_positions)
            seq_lengths.append(int(prompt_inputs.numel()))

    seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.int32, device=device)
    prompt_batch: Dict[str, torch.Tensor] = {
        "inputs": torch.cat(input_chunks, dim=0) if input_chunks else batch["inputs"].new_empty((0,)),
        "source_inputs": torch.cat(source_chunks, dim=0) if source_chunks else batch["inputs"].new_empty((0,)),
        "position_ids": (
            torch.cat(position_chunks, dim=0)
            if position_chunks
            else torch.empty((0, position_dim), dtype=torch.int32, device=device)
        ),
        "seq_lengths": seq_lengths_tensor,
        "seq_offsets": F.pad(torch.cumsum(seq_lengths_tensor, dim=0), (1, 0)).to(torch.int32),
        "puzzle_identifiers": batch["puzzle_identifiers"],
    }
    if "arc_identifiers" in batch:
        prompt_batch["arc_identifiers"] = batch["arc_identifiers"]
    return prompt_batch, source_chunks, final_prompt_positions


def _target_problem_position(
    positions: torch.Tensor,
    *,
    skip_trailing_start_position: bool,
    position_dim: int,
) -> torch.Tensor:
    fallback = torch.zeros((position_dim,), dtype=torch.int32, device=positions.device)
    if positions.ndim != 2 or positions.shape[0] == 0:
        return fallback

    candidates = positions.to(dtype=torch.int32)
    if skip_trailing_start_position and candidates.shape[0] > 1:
        candidates = candidates[:-1]
    if candidates.shape[0] == 0:
        return fallback

    if position_dim >= 4:
        input_side = candidates[:, 1] == 0
        if bool(input_side.any().item()):
            return candidates[input_side][-1].clone()

    return candidates[-1].clone()


def _initial_decode_positions(
    prompt_positions: List[torch.Tensor],
    *,
    position_dim: int,
    io_id: int,
) -> torch.Tensor:
    rows = []
    for positions in prompt_positions:
        decode_position = torch.zeros((position_dim,), dtype=torch.int32, device=positions.device)
        target_problem_position = _target_problem_position(
            positions,
            skip_trailing_start_position=True,
            position_dim=position_dim,
        )
        if position_dim >= 4:
            decode_position[:-2] = target_problem_position[:-2]
            decode_position[1] = int(io_id)
        elif position_dim > 2:
            decode_position[:-2] = target_problem_position[:-2]
        rows.append(decode_position)
    if rows:
        return torch.stack(rows, dim=0)
    return torch.empty((0, position_dim), dtype=torch.int32)


def _initial_special_positions(prompt_positions: List[torch.Tensor], position_dim: int) -> torch.Tensor:
    return _initial_decode_positions(prompt_positions, position_dim=position_dim, io_id=0)


def _initial_answer_positions(prompt_positions: List[torch.Tensor], position_dim: int) -> torch.Tensor:
    return _initial_decode_positions(prompt_positions, position_dim=position_dim, io_id=1)


def _run_prefill_to_cache(
    base_model: Any,
    prompt_batch: Dict[str, torch.Tensor],
    *,
    max_cache_len: int,
    cache_chunk_size: int,
) -> Tuple[Dict[str, torch.Tensor], URMInferenceCache]:
    carry = base_model.initial_carry(prompt_batch)
    aggregate_layers = []
    outputs: Dict[str, torch.Tensor] = {}
    loops = int(getattr(getattr(base_model, "config", None), "loops", 1))
    for _ in range(max(1, loops)):
        carry, outputs, step_cache = base_model(
            carry=carry,
            batch=prompt_batch,
            return_cache=True,
            max_cache_len=max_cache_len,
            cache_chunk_size=cache_chunk_size,
        )
        if step_cache is not None:
            aggregate_layers.extend(step_cache.layers)
        if bool(carry.halted.all().item()):
            break

    cache = URMInferenceCache(
        layers=aggregate_layers,
        batch_size=int(prompt_batch["puzzle_identifiers"].shape[0]),
        max_cache_len=max_cache_len,
        cache_chunk_size=cache_chunk_size,
    )
    return outputs, cache


def _decode_one_step_to_halt(
    base_model: Any,
    cache: URMInferenceCache,
    input_tokens: torch.Tensor,
    position_ids: torch.Tensor,
    puzzle_identifiers: torch.Tensor,
    *,
    max_cache_len: int,
    cache_chunk_size: int,
) -> Tuple[Dict[str, torch.Tensor], URMInferenceCache]:
    batch_size = int(input_tokens.shape[0])
    device = input_tokens.device
    seq_lengths = torch.ones((batch_size,), dtype=torch.int32, device=device)
    decode_batch = {
        "inputs": input_tokens.to(torch.int32),
        "position_ids": position_ids.to(device=device, dtype=torch.int32),
        "seq_lengths": seq_lengths,
        "seq_offsets": torch.arange(batch_size + 1, dtype=torch.int32, device=device),
        "puzzle_identifiers": puzzle_identifiers,
    }
    carry = base_model.initial_carry(decode_batch)
    outputs: Dict[str, torch.Tensor] = {}
    updated_cache = cache
    loops = int(getattr(getattr(base_model, "config", None), "loops", 1))
    for _ in range(max(1, loops)):
        carry, outputs, updated_cache = base_model(
            carry=carry,
            batch=decode_batch,
            cache=updated_cache,
            max_cache_len=max_cache_len,
            cache_chunk_size=cache_chunk_size,
        )
        if bool(carry.halted.all().item()):
            break
    return outputs, updated_cache


def _last_prompt_token_logits(
    logits: torch.Tensor,
    prompt_batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    seq_lengths = prompt_batch["seq_lengths"].to(device=logits.device, dtype=torch.long)
    if bool((seq_lengths <= 0).any().item()):
        raise ValueError("Cannot generate from an empty prompt sequence.")

    if logits.ndim == 3:
        batch_indices = torch.arange(seq_lengths.shape[0], dtype=torch.long, device=logits.device)
        return logits[batch_indices, seq_lengths - 1]

    if logits.ndim == 2:
        seq_offsets = prompt_batch["seq_offsets"].to(device=logits.device, dtype=torch.long)
        return logits[seq_offsets[1:] - 1]

    raise ValueError(f"Expected 2D or 3D logits, got shape={tuple(logits.shape)}.")


def _prompt_ends_with_token(prompt_batch: Dict[str, torch.Tensor], token_id: int) -> bool:
    seq_offsets = prompt_batch["seq_offsets"].to(device=prompt_batch["inputs"].device, dtype=torch.long)
    if seq_offsets.numel() <= 1:
        return False
    lengths = seq_offsets[1:] - seq_offsets[:-1]
    if bool((lengths <= 0).any().item()):
        return False
    last_tokens = prompt_batch["inputs"][seq_offsets[1:] - 1].to(torch.long)
    return bool(torch.all(last_tokens == int(token_id)).item())


def _drop_eval_targets(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    label_keys = {"labels", "answer_mask", "label_seq_lengths", "label_seq_offsets", "label_seq_shapes", "seq_shapes"}
    return {key: value for key, value in batch.items() if key not in label_keys}


def _final_sample_result(
    prompt_batch: Dict[str, torch.Tensor],
    source_chunks: List[torch.Tensor],
    prompt_positions: List[torch.Tensor],
    generated_tokens: torch.Tensor,
    generated_positions: torch.Tensor,
    generated_lengths: torch.Tensor,
    q_halt_logits: torch.Tensor,
    sample_idx: int,
) -> Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, torch.Tensor], Dict[str, float]]:
    prompt_start = int(prompt_batch["seq_offsets"][sample_idx].item())
    prompt_end = int(prompt_batch["seq_offsets"][sample_idx + 1].item())
    prompt_inputs = prompt_batch["inputs"][prompt_start:prompt_end]
    prompt_source_inputs = source_chunks[sample_idx]
    prompt_pos = prompt_positions[sample_idx]
    gen_len = int(generated_lengths[sample_idx].item())
    sample_generated = generated_tokens[sample_idx, :gen_len].to(prompt_inputs.dtype)
    sample_generated_positions = generated_positions[sample_idx, :gen_len].to(prompt_pos.dtype)

    inputs = torch.cat([prompt_inputs, sample_generated], dim=0)
    source_inputs = torch.cat([prompt_source_inputs, sample_generated.to(prompt_source_inputs.dtype)], dim=0)
    position_ids = torch.cat([prompt_pos, sample_generated_positions], dim=0)
    answer_mask = torch.cat(
        [
            torch.zeros((int(prompt_inputs.numel()),), dtype=torch.bool, device=inputs.device),
            torch.ones((gen_len,), dtype=torch.bool, device=inputs.device),
        ],
        dim=0,
    )
    preds = torch.cat(
        [
            torch.zeros((int(prompt_inputs.numel()),), dtype=torch.long, device=inputs.device),
            sample_generated.to(torch.long),
        ],
        dim=0,
    )
    final_batch: Dict[str, Optional[torch.Tensor]] = {
        "inputs": inputs,
        "labels": None,
        "answer_mask": answer_mask,
        "source_inputs": source_inputs,
        "position_ids": position_ids,
        "seq_lengths": torch.tensor([inputs.numel()], dtype=torch.int32, device=inputs.device),
        "seq_offsets": torch.tensor([0, inputs.numel()], dtype=torch.int32, device=inputs.device),
        "puzzle_identifiers": prompt_batch["puzzle_identifiers"][sample_idx : sample_idx + 1],
    }
    if "arc_identifiers" in prompt_batch:
        final_batch["arc_identifiers"] = prompt_batch["arc_identifiers"][sample_idx : sample_idx + 1]

    preds_dict = {
        "preds": preds,
        "q_halt_logits": q_halt_logits[sample_idx : sample_idx + 1].to(torch.float32),
    }
    return final_batch, preds_dict, {"count": 0.0, "accuracy": 0.0, "exact_accuracy": 0.0, "steps": 0.0}


def _generate_urm_batch(
    base_model: Any,
    batch: Dict[str, torch.Tensor],
    *,
    start_token_id: int,
    end_token_id: int,
    newline_token_id: int,
    max_new_tokens: int,
    cache_chunk_size: int,
) -> List[Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, torch.Tensor], Dict[str, float]]]:
    prompt_batch, source_chunks, prompt_positions = _prepare_prompt_batch(
        batch,
        base_model,
        newline_token_id=newline_token_id,
    )
    batch_size = int(prompt_batch["puzzle_identifiers"].shape[0])
    device = prompt_batch["inputs"].device
    max_prompt_len = int(prompt_batch["seq_lengths"].max().item()) if batch_size > 0 else 0
    prefix_seq_len = int(getattr(getattr(base_model, "inner", None), "prefix_seq_len", 0))
    max_cache_len = max_prompt_len + prefix_seq_len
    prefill_outputs, cache = _run_prefill_to_cache(
        base_model,
        prompt_batch,
        max_cache_len=max_cache_len,
        cache_chunk_size=cache_chunk_size,
    )

    position_dim = int(prompt_batch["position_ids"].shape[-1]) if prompt_batch["position_ids"].ndim == 2 else 4
    current_positions = _initial_special_positions(prompt_positions, position_dim).to(device=device, dtype=torch.int32)
    next_answer_positions = _initial_answer_positions(prompt_positions, position_dim).to(device=device, dtype=torch.int32)
    input_tokens = torch.full((batch_size,), int(start_token_id), dtype=torch.long, device=device)
    generated_tokens = torch.full((batch_size, max_new_tokens), int(end_token_id), dtype=torch.long, device=device)
    generated_positions = torch.zeros((batch_size, max_new_tokens, position_dim), dtype=torch.int32, device=device)
    generated_lengths = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    answer_row_widths = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    finished = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    q_halt_logits = prefill_outputs.get(
        "q_halt_logits",
        torch.zeros((batch_size,), dtype=torch.float32, device=device),
    ).to(torch.float32)

    config = getattr(base_model, "config", None)
    grid_height = int(getattr(config, "grid_height", 30) or 30)
    grid_width = int(getattr(config, "grid_width", 30) or 30)

    decode_start = 0
    if int(max_new_tokens) > 0 and _prompt_ends_with_token(prompt_batch, start_token_id):
        logits = _last_prompt_token_logits(prefill_outputs["logits"], prompt_batch)
        next_tokens = torch.argmax(logits, dim=-1).to(torch.long)
        is_end = next_tokens == int(end_token_id)
        append_mask = ~finished & ~is_end
        generated_tokens[:, 0] = next_tokens
        generated_positions[:, 0] = torch.where(
            is_end.unsqueeze(-1),
            torch.zeros_like(next_answer_positions),
            next_answer_positions,
        )
        generated_lengths = generated_lengths + append_mask.to(torch.int32)
        finished = finished | is_end
        current_positions = torch.where(
            is_end.unsqueeze(-1),
            torch.zeros_like(next_answer_positions),
            next_answer_positions,
        )
        first_newline_widths = next_answer_positions[:, -1] + 1
        answer_row_widths = torch.where(
            (answer_row_widths <= 0) & append_mask & (next_tokens == int(newline_token_id)),
            first_newline_widths.to(answer_row_widths.dtype),
            answer_row_widths,
        )
        next_answer_positions = _advance_positions(
            next_answer_positions,
            next_tokens,
            newline_token_id=newline_token_id,
            end_token_id=end_token_id,
            grid_height=grid_height,
            grid_width=grid_width,
            row_widths=answer_row_widths,
        )
        input_tokens = next_tokens
        decode_start = 1

    for decode_step in range(decode_start, int(max_new_tokens)):
        if bool(finished.all().item()):
            break

        outputs, cache = _decode_one_step_to_halt(
            base_model,
            cache,
            input_tokens,
            current_positions,
            prompt_batch["puzzle_identifiers"],
            max_cache_len=max_cache_len,
            cache_chunk_size=cache_chunk_size,
        )
        logits = outputs["logits"]
        next_tokens = torch.argmax(logits, dim=-1).to(torch.long)
        next_tokens = torch.where(finished, torch.full_like(next_tokens, int(end_token_id)), next_tokens)
        q_halt_logits = outputs.get("q_halt_logits", q_halt_logits).to(torch.float32)

        is_end = next_tokens == int(end_token_id)
        append_mask = ~finished & ~is_end
        generated_tokens[:, decode_step] = next_tokens
        generated_positions[:, decode_step] = torch.where(
            is_end.unsqueeze(-1),
            torch.zeros_like(next_answer_positions),
            next_answer_positions,
        )
        generated_lengths = generated_lengths + append_mask.to(torch.int32)
        finished = finished | is_end
        current_positions = torch.where(
            is_end.unsqueeze(-1),
            torch.zeros_like(next_answer_positions),
            next_answer_positions,
        )
        first_newline_widths = next_answer_positions[:, -1] + 1
        answer_row_widths = torch.where(
            (answer_row_widths <= 0) & append_mask & (next_tokens == int(newline_token_id)),
            first_newline_widths.to(answer_row_widths.dtype),
            answer_row_widths,
        )
        next_answer_positions = _advance_positions(
            next_answer_positions,
            next_tokens,
            newline_token_id=newline_token_id,
            end_token_id=end_token_id,
            grid_height=grid_height,
            grid_width=grid_width,
            row_widths=answer_row_widths,
        )
        input_tokens = next_tokens

    return [
        _final_sample_result(
            prompt_batch,
            source_chunks,
            prompt_positions,
            generated_tokens,
            generated_positions,
            generated_lengths,
            q_halt_logits,
            sample_idx,
        )
        for sample_idx in range(batch_size)
    ]


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
    train_state.model.eval()
    base_model.eval()

    start_token_id, end_token_id = _resolve_casual_token_ids(config, base_model, eval_metadata)
    newline_token_id = int(getattr(config.arch, "casual_lm_newline_token_id", ARC_NEWLINE_TOKEN_ID))
    max_new_tokens = _resolve_max_new_tokens(config)
    cache_chunk_size = _resolve_cache_chunk_size(config)

    if rank == 0:
        print(
            "Autoregressive URM evaluation uses URM.forward prefill/decode KV cache "
            f"(START_TOKEN={start_token_id}, END_TOKEN={end_token_id}, "
            f"NEWLINE_TOKEN={newline_token_id}, max_new_tokens={max_new_tokens}, "
            f"cache_chunk_size={cache_chunk_size})."
        )

    for evaluator in evaluators:
        evaluator.begin_eval()

    set_ids = {name: idx for idx, name in enumerate(eval_metadata.sets)}
    metric_keys = ["count", "accuracy", "exact_accuracy", "steps"]
    metric_values = torch.zeros((len(set_ids), len(metric_keys)), dtype=torch.float64, device=device)

    processed_batches = 0
    print("Starting autoregressive URM evaluation... len(eval_loader) =", len(eval_loader))
    with torch.inference_mode():
        for set_name, batch, _global_batch_size in eval_loader:
            if early_eval and processed_batches > 50:
                break
            processed_batches += 1
            if rank == 0:
                print(f"Processing autoregressive URM batch {processed_batches}: {set_name}")

            batch = _to_device(batch, device)
            batch = _drop_eval_targets(batch)
            set_id = set_ids[set_name]
            sample_count = _sample_count(batch)
            if sample_count == 0:
                continue
            if rank == 0:
                sample_label = "sample" if sample_count == 1 else "samples"
                print(f"    Generating {sample_count} {sample_label} with URM KV-cache decode...")

            sample_results = _generate_urm_batch(
                base_model,
                batch,
                start_token_id=start_token_id,
                end_token_id=end_token_id,
                newline_token_id=newline_token_id,
                max_new_tokens=max_new_tokens,
                cache_chunk_size=cache_chunk_size,
            )

            for final_batch, preds, sample_metrics in sample_results:
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
        print(f"\nRunning {len(evaluators)} autoregressive URM evaluator(s)...")

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
        print("All autoregressive URM evaluators completed!")
    return reduced_metrics
