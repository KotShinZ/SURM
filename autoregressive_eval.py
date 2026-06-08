from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from data.common import PuzzleDatasetMetadata
from models.layers import apply_rotary_pos_emb_one, flash_attn_varlen_func, rms_norm
from models.losses import IGNORE_LABEL_ID


ARC_EOS_TOKEN_ID = 1
DEFAULT_CASUAL_MAX_NEW_TOKENS = 931


def _unwrap_base_model(model: Any) -> Any:
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return getattr(model, "model", model)


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _resolve_casual_max_new_tokens(config: Any) -> int:
    value = getattr(config, "autoregressive_eval_max_new_tokens", None)
    if value is None:
        value = getattr(getattr(config, "arch", None), "autoregressive_eval_max_new_tokens", None)
    if value is None:
        return DEFAULT_CASUAL_MAX_NEW_TOKENS

    max_new_tokens = int(value)
    if max_new_tokens <= 0:
        raise ValueError("autoregressive_eval_max_new_tokens must be positive when provided.")
    return max_new_tokens


def _sample_ranges(batch: Dict[str, torch.Tensor], sample_idx: int) -> Tuple[int, int, int, int]:
    seq_offsets = batch["seq_offsets"]
    label_offsets = batch.get("label_seq_offsets", seq_offsets)
    return (
        int(seq_offsets[sample_idx].item()),
        int(seq_offsets[sample_idx + 1].item()),
        int(label_offsets[sample_idx].item()),
        int(label_offsets[sample_idx + 1].item()),
    )


def _is_casual_mode(base_model: Any) -> bool:
    return getattr(getattr(base_model, "config", None), "forward_mode", None) == "casual"


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
    *,
    casual_lm: bool = False,
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

    labels = (
        torch.full((inputs.numel(),), IGNORE_LABEL_ID, dtype=torch.int32, device=sample_inputs.device)
        if casual_lm
        else torch.full((answer_len,), IGNORE_LABEL_ID, dtype=torch.int32, device=sample_inputs.device)
    )
    gen_batch: Dict[str, torch.Tensor] = {
        "inputs": inputs,
        "labels": labels,
        "answer_mask": answer_mask,
        "source_inputs": source_inputs,
        "position_ids": position_ids,
        "seq_lengths": torch.tensor([inputs.numel()], dtype=torch.int32, device=sample_inputs.device),
        "seq_offsets": _offsets(inputs.numel(), device=sample_inputs.device),
        "puzzle_identifiers": batch["puzzle_identifiers"][sample_idx : sample_idx + 1],
    }
    if not casual_lm:
        gen_batch["label_seq_lengths"] = torch.tensor([answer_len], dtype=torch.int32, device=sample_inputs.device)
        gen_batch["label_seq_offsets"] = _offsets(answer_len, device=sample_inputs.device)
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


@dataclass
class _KVCacheEntry:
    context_key: torch.Tensor
    context_value: torch.Tensor
    answer_key: torch.Tensor
    answer_value: torch.Tensor
    answer_ffn: torch.Tensor
    context_ffn_tail: Optional[torch.Tensor] = None
    answer_len: int = 0

    def key_value_with_current(
        self,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_chunks = [self.context_key]
        value_chunks = [self.context_value]
        if self.answer_len > 0:
            key_chunks.append(self.answer_key[: self.answer_len])
            value_chunks.append(self.answer_value[: self.answer_len])
        key_chunks.append(current_key)
        value_chunks.append(current_value)
        return torch.cat(key_chunks, dim=0), torch.cat(value_chunks, dim=0)

    def previous_ffn(self) -> Optional[torch.Tensor]:
        if self.answer_len == 0:
            return self.context_ffn_tail
        return self.answer_ffn[self.answer_len - 1 : self.answer_len]

    def append(
        self,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        current_ffn: torch.Tensor,
    ) -> None:
        self.answer_key[self.answer_len : self.answer_len + 1] = current_key
        self.answer_value[self.answer_len : self.answer_len + 1] = current_value
        self.answer_ffn[self.answer_len : self.answer_len + 1] = current_ffn
        self.answer_len += 1


@dataclass
class _ParallelSampleState:
    sample_idx: int
    target_tokens: torch.Tensor
    target_positions: torch.Tensor
    sample_answer_mask: torch.Tensor
    generated_tokens: torch.Tensor
    q_halt_logits: torch.Tensor
    steps: float = 0.0


@dataclass
class _CasualNoSizeState:
    sample_idx: int
    prompt_inputs: torch.Tensor
    prompt_source_inputs: torch.Tensor
    prompt_positions: torch.Tensor
    puzzle_identifier: torch.Tensor
    arc_identifier: Optional[torch.Tensor]
    generated_tokens: torch.Tensor
    q_halt_logits: torch.Tensor
    finished: bool = False
    steps: float = 0.0


def _advance_arc_position(row: int, col: int, token_id: int) -> Tuple[int, int]:
    if int(token_id) == ARC_EOS_TOKEN_ID:
        return min(row + 1, 30 - 1), 0
    return row, min(col + 1, 30 - 1)


def _answer_position_prefix(prompt_positions: torch.Tensor, *, io_id: int = 1) -> torch.Tensor:
    if prompt_positions.ndim != 2 or prompt_positions.shape[0] == 0:
        return torch.tensor([0, int(io_id)], dtype=torch.int32, device=prompt_positions.device)

    position_dim = int(prompt_positions.shape[1])
    prefix = torch.zeros((position_dim - 2,), dtype=prompt_positions.dtype, device=prompt_positions.device)
    candidates = prompt_positions
    if candidates.shape[0] > 1:
        candidates = candidates[:-1]

    if position_dim >= 4:
        input_side = candidates[:, 1] == 0
        if bool(input_side.any().item()):
            target_problem_position = candidates[input_side][-1]
        else:
            target_problem_position = candidates[-1]
        prefix = target_problem_position[:-2].clone()
        prefix[1] = int(io_id)
    elif position_dim > 2:
        prefix = candidates[-1, :-2].clone()
    return prefix


def _make_answer_positions_from_tokens(
    generated_tokens: torch.Tensor,
    prompt_positions: torch.Tensor,
    *,
    shifted: bool,
) -> torch.Tensor:
    if prompt_positions.ndim == 2 and prompt_positions.shape[1] > 0:
        position_dim = int(prompt_positions.shape[1])
        dtype = prompt_positions.dtype
        device = prompt_positions.device
    else:
        position_dim = 4
        dtype = torch.int32
        device = generated_tokens.device

    answer_prefix = _answer_position_prefix(
        prompt_positions
        if prompt_positions.ndim == 2
        else torch.empty((0, position_dim), dtype=dtype, device=device),
        io_id=1,
    )
    start_prefix = _answer_position_prefix(
        prompt_positions
        if prompt_positions.ndim == 2
        else torch.empty((0, position_dim), dtype=dtype, device=device),
        io_id=0,
    )
    if answer_prefix.numel() + 2 != position_dim:
        answer_prefix = torch.zeros((max(position_dim - 2, 0),), dtype=dtype, device=device)
    if start_prefix.numel() + 2 != position_dim:
        start_prefix = torch.zeros((max(position_dim - 2, 0),), dtype=dtype, device=device)

    length = int(generated_tokens.numel()) + (1 if shifted else 0)
    positions = torch.zeros((length, position_dim), dtype=dtype, device=device)
    row = 0
    col = 0
    for idx in range(length):
        prefix = start_prefix if shifted and idx == 0 else answer_prefix
        if prefix.numel() > 0:
            positions[idx, :-2] = prefix
        positions[idx, -2] = row
        positions[idx, -1] = col
        if shifted:
            if idx < generated_tokens.numel():
                row, col = _advance_arc_position(row, col, int(generated_tokens[idx].item()))
        else:
            row, col = _advance_arc_position(row, col, int(generated_tokens[idx].item()))
    return positions


def _run_base_model_to_halt(base_model: Any, batch: Dict[str, torch.Tensor]) -> Tuple[Any, Dict[str, torch.Tensor]]:
    carry = base_model.initial_carry(batch)
    max_steps = int(getattr(getattr(base_model, "config", None), "loops", 1)) + 1
    outputs: Dict[str, torch.Tensor] = {}
    for _ in range(max_steps):
        carry, outputs = base_model(carry=carry, batch=batch)
        if bool(carry.halted.all().item()):
            break
    return carry, outputs


def _project_query_key_value(layer: Any, states: torch.Tensor, cos_sin: Tuple[torch.Tensor, ...]):
    attn = layer.self_attn
    qkv = attn.qkv_proj(states)
    qkv = qkv.view(
        states.shape[0],
        attn.num_heads + 2 * attn.num_key_value_heads,
        attn.head_dim,
    )
    query = qkv[:, : attn.num_heads]
    key = qkv[:, attn.num_heads : attn.num_heads + attn.num_key_value_heads]
    value = qkv[:, attn.num_heads + attn.num_key_value_heads :]
    return (
        apply_rotary_pos_emb_one(query, cos_sin),
        apply_rotary_pos_emb_one(key, cos_sin),
        value,
    )


def _project_key_value(layer: Any, states: torch.Tensor, cos_sin: Tuple[torch.Tensor, ...]):
    _query, key, value = _project_query_key_value(layer, states, cos_sin)
    return key, value


def _new_cache_entry_from_key_value(
    layer: Any,
    context_key: torch.Tensor,
    context_value: torch.Tensor,
    max_answer_len: int,
    *,
    context_ffn_tail: Optional[torch.Tensor] = None,
) -> _KVCacheEntry:
    attn = layer.self_attn
    mlp = layer.mlp
    return _KVCacheEntry(
        context_key=context_key,
        context_value=context_value,
        answer_key=context_key.new_empty((max_answer_len, attn.num_key_value_heads, attn.head_dim)),
        answer_value=context_value.new_empty((max_answer_len, attn.num_key_value_heads, attn.head_dim)),
        answer_ffn=context_key.new_empty((max_answer_len, mlp.inter)),
        context_ffn_tail=context_ffn_tail,
    )


def _new_cache_entry(
    layer: Any,
    context_states: torch.Tensor,
    context_cos_sin: Tuple[torch.Tensor, ...],
    max_answer_len: int,
    *,
    context_ffn_tail: Optional[torch.Tensor] = None,
) -> _KVCacheEntry:
    context_key, context_value = _project_key_value(layer, context_states, context_cos_sin)
    return _new_cache_entry_from_key_value(
        layer,
        context_key,
        context_value,
        max_answer_len,
        context_ffn_tail=context_ffn_tail,
    )


def _run_mlp_packed_with_ffn(
    layer: Any,
    states: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mlp = layer.mlp
    gate, up = mlp.gate_up_proj(states).chunk(2, dim=-1)
    ffn = F.silu(gate) * up

    gap = max(0, mlp.conv_kernel - 1)
    if gap > 0 and ffn.shape[0] > 0:
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device=ffn.device, dtype=torch.long)
        seq_ids = torch.repeat_interleave(
            torch.arange(lengths.shape[0], device=ffn.device, dtype=torch.long),
            lengths,
        )
        token_positions = torch.arange(ffn.shape[0], device=ffn.device, dtype=torch.long)
        expanded_positions = token_positions + seq_ids * gap
        expanded_len = ffn.shape[0] + lengths.shape[0] * gap
        expanded = ffn.new_zeros((expanded_len, ffn.shape[-1]))
        expanded[expanded_positions] = ffn
    else:
        expanded_positions = torch.arange(ffn.shape[0], device=ffn.device, dtype=torch.long)
        expanded = ffn

    conv_output = mlp.dwconv(expanded.unsqueeze(0).transpose(1, 2).to(mlp.dwconv.weight.dtype))
    conv_output = conv_output[..., : expanded.size(0)]
    conv_output = mlp.act(conv_output).transpose(1, 2).squeeze(0).contiguous()
    conv_output = conv_output[expanded_positions]
    return mlp.down_proj(mlp.mlp_dropout(conv_output)), ffn


def _run_cached_mlp(layer: Any, states: torch.Tensor, entry: _KVCacheEntry) -> Tuple[torch.Tensor, torch.Tensor]:
    mlp = layer.mlp
    gate, up = mlp.gate_up_proj(states).chunk(2, dim=-1)
    current_ffn = F.silu(gate) * up
    previous_ffn = entry.previous_ffn()
    conv_input = current_ffn if previous_ffn is None else torch.cat([previous_ffn, current_ffn], dim=0)
    conv_output = mlp.dwconv(conv_input.unsqueeze(0).transpose(1, 2).to(mlp.dwconv.weight.dtype))
    conv_output = conv_output[..., : conv_input.shape[0]]
    conv_output = mlp.act(conv_output[..., -1:].transpose(1, 2)).squeeze(0).contiguous()
    return mlp.down_proj(mlp.mlp_dropout(conv_output)), current_ffn


def _run_cached_answer_layer(
    layer: Any,
    states: torch.Tensor,
    token_cos_sin: Tuple[torch.Tensor, ...],
    entry: _KVCacheEntry,
) -> torch.Tensor:
    query, current_key, current_value = _project_query_key_value(layer, states, token_cos_sin)
    key, value = entry.key_value_with_current(current_key, current_value)
    cu_q = torch.tensor([0, query.shape[0]], dtype=torch.int32, device=query.device)
    cu_k = torch.tensor([0, key.shape[0]], dtype=torch.int32, device=query.device)
    attn_output = flash_attn_varlen_func(
        q=query.contiguous(),
        k=key.contiguous(),
        v=value.contiguous(),
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=query.shape[0],
        max_seqlen_k=key.shape[0],
        causal=False,
        window_size=(-1, -1),
        dropout_p=0.0,
    )
    if isinstance(attn_output, tuple):
        attn_output = attn_output[0]

    states = rms_norm(
        states + layer.self_attn.o_proj(attn_output.reshape(states.shape[0], layer.self_attn.output_size)),
        variance_epsilon=layer.norm_eps,
    )
    mlp_output, current_ffn = _run_cached_mlp(layer, states, entry)
    states = rms_norm(states + mlp_output, variance_epsilon=layer.norm_eps)
    entry.append(current_key, current_value, current_ffn)
    return states


def _supports_autoregressive_kv_cache(base_model: Any) -> bool:
    config = getattr(base_model, "config", None)
    inner = getattr(base_model, "inner", None)
    if config is None or inner is None:
        return False
    if getattr(config, "forward_mode", None) not in {"prefix_lm", "casual"}:
        return False
    if not getattr(config, "variable_seq_lengths", False) or inner.use_hrm:
        return False
    if getattr(config, "patch_io_enabled", False):
        return False
    if getattr(config, "answer_initial_mode", "default") != "default":
        return False
    if float(getattr(config, "noise_size", 0.0)) != 0.0:
        return False
    if float(getattr(config, "input_embedding_noise_size", 0.0)) != 0.0:
        return False
    if len(inner.prelude_layers) != 0 or len(inner.coda_layers) != 0 or len(inner.context_layers) != 0:
        return False
    if inner._label_separate_C_enabled():
        return False
    for layer in inner.layers:
        if layer.self_attn.attention_type != "full":
            return False
        if layer.mlp.conv_kernel != 2:
            return False
    return True


def _supports_prefix_lm_kv_cache(base_model: Any) -> bool:
    return _supports_autoregressive_kv_cache(base_model)


def _build_context_batch(
    batch: Dict[str, torch.Tensor],
    sample_idx: int,
    sample_answer_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    start, end, _label_start, _label_end = _sample_ranges(batch, sample_idx)
    context_mask = ~sample_answer_mask
    context_inputs = batch["inputs"][start:end][context_mask]
    context_positions = batch["position_ids"][start:end][context_mask]
    context_len = int(context_inputs.numel())
    context_batch = {
        "inputs": context_inputs,
        "labels": torch.full((context_len,), IGNORE_LABEL_ID, dtype=torch.int32, device=context_inputs.device),
        "answer_mask": torch.zeros((context_len,), dtype=torch.bool, device=context_inputs.device),
        "position_ids": context_positions,
        "seq_lengths": torch.tensor([context_len], dtype=torch.int32, device=context_inputs.device),
        "seq_offsets": _offsets(context_len, device=context_inputs.device),
        "puzzle_identifiers": batch["puzzle_identifiers"][sample_idx : sample_idx + 1],
    }
    if "source_inputs" in batch:
        context_batch["source_inputs"] = batch["source_inputs"][start:end][context_mask]
    return context_batch


def _build_full_position_batch(
    batch: Dict[str, torch.Tensor],
    sample_idx: int,
    target_positions: torch.Tensor,
    sample_answer_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    start, end, _label_start, _label_end = _sample_ranges(batch, sample_idx)
    context_mask = ~sample_answer_mask
    context_inputs = batch["inputs"][start:end][context_mask]
    context_positions = batch["position_ids"][start:end][context_mask]
    answer_placeholders = torch.zeros(
        (int(target_positions.shape[0]),),
        dtype=context_inputs.dtype,
        device=context_inputs.device,
    )
    inputs = torch.cat([context_inputs, answer_placeholders], dim=0)
    position_ids = torch.cat([context_positions, target_positions], dim=0)
    return {
        "inputs": inputs,
        "labels": torch.full((inputs.numel(),), IGNORE_LABEL_ID, dtype=torch.int32, device=inputs.device),
        "answer_mask": torch.cat(
            [
                torch.zeros((context_inputs.numel(),), dtype=torch.bool, device=inputs.device),
                torch.ones((target_positions.shape[0],), dtype=torch.bool, device=inputs.device),
            ],
            dim=0,
        ),
        "position_ids": position_ids,
        "seq_lengths": torch.tensor([inputs.numel()], dtype=torch.int32, device=inputs.device),
        "seq_offsets": _offsets(inputs.numel(), device=inputs.device),
        "puzzle_identifiers": batch["puzzle_identifiers"][sample_idx : sample_idx + 1],
    }


def _build_parallel_context_batch(
    batch: Dict[str, torch.Tensor],
    states: List[_ParallelSampleState],
) -> Dict[str, torch.Tensor]:
    input_chunks = []
    source_chunks = []
    position_chunks = []
    seq_lengths = []
    puzzle_identifiers = []

    for state in states:
        start, end, _label_start, _label_end = _sample_ranges(batch, state.sample_idx)
        context_mask = ~state.sample_answer_mask
        context_inputs = batch["inputs"][start:end][context_mask]
        input_chunks.append(context_inputs)
        source_chunks.append(batch.get("source_inputs", batch["inputs"])[start:end][context_mask])
        position_chunks.append(batch["position_ids"][start:end][context_mask])
        seq_lengths.append(context_inputs.numel())
        puzzle_identifiers.append(batch["puzzle_identifiers"][state.sample_idx : state.sample_idx + 1])

    device = batch["inputs"].device
    seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.int32, device=device)
    inputs = torch.cat(input_chunks, dim=0) if input_chunks else batch["inputs"].new_empty((0,))
    context_batch = {
        "inputs": inputs,
        "labels": torch.full((inputs.numel(),), IGNORE_LABEL_ID, dtype=torch.int32, device=device),
        "answer_mask": torch.zeros((inputs.numel(),), dtype=torch.bool, device=device),
        "source_inputs": torch.cat(source_chunks, dim=0) if source_chunks else batch["inputs"].new_empty((0,)),
        "position_ids": (
            torch.cat(position_chunks, dim=0)
            if position_chunks
            else batch["position_ids"].new_empty((0,) + batch["position_ids"].shape[1:])
        ),
        "seq_lengths": seq_lengths_tensor,
        "seq_offsets": F.pad(torch.cumsum(seq_lengths_tensor, dim=0), (1, 0)).to(torch.int32),
        "puzzle_identifiers": (
            torch.cat(puzzle_identifiers, dim=0)
            if puzzle_identifiers
            else batch["puzzle_identifiers"].new_empty((0,))
        ),
    }
    return context_batch


def _build_parallel_full_position_batch(
    batch: Dict[str, torch.Tensor],
    states: List[_ParallelSampleState],
) -> Dict[str, torch.Tensor]:
    input_chunks = []
    position_chunks = []
    answer_mask_chunks = []
    seq_lengths = []
    puzzle_identifiers = []

    for state in states:
        start, end, _label_start, _label_end = _sample_ranges(batch, state.sample_idx)
        context_mask = ~state.sample_answer_mask
        context_inputs = batch["inputs"][start:end][context_mask]
        context_positions = batch["position_ids"][start:end][context_mask]
        answer_placeholders = torch.zeros(
            (int(state.target_positions.shape[0]),),
            dtype=context_inputs.dtype,
            device=context_inputs.device,
        )
        inputs = torch.cat([context_inputs, answer_placeholders], dim=0)
        input_chunks.append(inputs)
        position_chunks.append(torch.cat([context_positions, state.target_positions], dim=0))
        answer_mask_chunks.append(
            torch.cat(
                [
                    torch.zeros((context_inputs.numel(),), dtype=torch.bool, device=context_inputs.device),
                    torch.ones((state.target_positions.shape[0],), dtype=torch.bool, device=context_inputs.device),
                ],
                dim=0,
            )
        )
        seq_lengths.append(inputs.numel())
        puzzle_identifiers.append(batch["puzzle_identifiers"][state.sample_idx : state.sample_idx + 1])

    device = batch["inputs"].device
    seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.int32, device=device)
    inputs = torch.cat(input_chunks, dim=0) if input_chunks else batch["inputs"].new_empty((0,))
    return {
        "inputs": inputs,
        "labels": torch.full((inputs.numel(),), IGNORE_LABEL_ID, dtype=torch.int32, device=device),
        "answer_mask": (
            torch.cat(answer_mask_chunks, dim=0)
            if answer_mask_chunks
            else torch.empty((0,), dtype=torch.bool, device=device)
        ),
        "position_ids": (
            torch.cat(position_chunks, dim=0)
            if position_chunks
            else batch["position_ids"].new_empty((0,) + batch["position_ids"].shape[1:])
        ),
        "seq_lengths": seq_lengths_tensor,
        "seq_offsets": F.pad(torch.cumsum(seq_lengths_tensor, dim=0), (1, 0)).to(torch.int32),
        "puzzle_identifiers": (
            torch.cat(puzzle_identifiers, dim=0)
            if puzzle_identifiers
            else batch["puzzle_identifiers"].new_empty((0,))
        ),
    }


def _precompute_context_cache(
    base_model: Any,
    context_batch: Dict[str, torch.Tensor],
    context_cos_sin: Tuple[torch.Tensor, ...],
    max_answer_len: int,
) -> Tuple[torch.Tensor, List[_KVCacheEntry]]:
    inner = base_model.inner
    casual_lm = _is_casual_mode(base_model)
    context_embeddings, _token_indices = inner._input_embeddings_packed(context_batch)
    context_states = base_model._make_packed_initial_hidden(context_batch)
    context_len = int(context_states.shape[0])
    cu_context = torch.tensor([0, context_len], dtype=torch.int32, device=context_states.device)
    caches: List[_KVCacheEntry] = []

    for _outer_step in range(int(base_model.config.loops)):
        for _h_cycle in range(int(base_model.config.H_cycles)):
            for _l_cycle in range(int(base_model.config.L_cycles)):
                context_states = inner._inject_inputs(context_states, context_embeddings)
                for layer in inner.layers:
                    if casual_lm:
                        context_key, context_value = _project_key_value(layer, context_states, context_cos_sin)
                        attn_output = layer.self_attn.forward_packed(
                            cos_sin=context_cos_sin,
                            hidden_states=context_states,
                            cu_seqlens=cu_context,
                            max_seqlen=context_len,
                        )
                        attn_states = rms_norm(
                            context_states + attn_output,
                            variance_epsilon=layer.norm_eps,
                        )
                        mlp_output, context_ffn = _run_mlp_packed_with_ffn(layer, attn_states, cu_context)
                        context_states = rms_norm(attn_states + mlp_output, variance_epsilon=layer.norm_eps)
                        context_ffn_tail = context_ffn[-1:] if context_ffn.numel() > 0 else None
                        caches.append(
                            _new_cache_entry_from_key_value(
                                layer,
                                context_key,
                                context_value,
                                max_answer_len=max_answer_len,
                                context_ffn_tail=context_ffn_tail,
                            )
                        )
                    else:
                        context_states = layer.forward_packed(
                            cos_sin=context_cos_sin,
                            hidden_states=context_states,
                            cu_seqlens=cu_context,
                            max_seqlen=context_len,
                        )
                        caches.append(
                            _new_cache_entry(
                                layer,
                                context_states,
                                context_cos_sin,
                                max_answer_len=max_answer_len,
                            )
                        )
    return context_states, caches


def _new_parallel_cache_entries(
    layer: Any,
    context_states: torch.Tensor,
    context_key: torch.Tensor,
    context_value: torch.Tensor,
    cu_context: torch.Tensor,
    max_answer_lens: List[int],
    context_ffn_tail: Optional[torch.Tensor] = None,
) -> List[_KVCacheEntry]:
    attn = layer.self_attn
    mlp = layer.mlp
    entries = []
    for sample_idx, max_answer_len in enumerate(max_answer_lens):
        start = int(cu_context[sample_idx].item())
        end = int(cu_context[sample_idx + 1].item())
        sample_context_states = context_states[start:end]
        sample_context_key = context_key[start:end]
        sample_context_value = context_value[start:end]
        entries.append(
            _KVCacheEntry(
                context_key=sample_context_key,
                context_value=sample_context_value,
                answer_key=sample_context_key.new_empty((max_answer_len, attn.num_key_value_heads, attn.head_dim)),
                answer_value=sample_context_value.new_empty((max_answer_len, attn.num_key_value_heads, attn.head_dim)),
                answer_ffn=sample_context_states.new_empty((max_answer_len, mlp.inter)),
                context_ffn_tail=(
                    context_ffn_tail[sample_idx : sample_idx + 1]
                    if context_ffn_tail is not None
                    else None
                ),
            )
        )
    return entries


def _precompute_context_cache_parallel(
    base_model: Any,
    context_batch: Dict[str, torch.Tensor],
    context_cos_sin: Tuple[torch.Tensor, ...],
    max_answer_lens: List[int],
) -> Tuple[torch.Tensor, List[List[_KVCacheEntry]], torch.Tensor]:
    inner = base_model.inner
    casual_lm = _is_casual_mode(base_model)
    context_embeddings, _token_indices = inner._input_embeddings_packed(context_batch)
    context_states = base_model._make_packed_initial_hidden(context_batch)
    context_lengths = context_batch["seq_lengths"].to(torch.int32) + int(inner.prefix_seq_len)
    cu_context = F.pad(torch.cumsum(context_lengths, dim=0), (1, 0)).to(torch.int32)
    max_context_len = int(context_lengths.max().item()) if context_lengths.numel() else 0
    caches: List[List[_KVCacheEntry]] = []

    for _outer_step in range(int(base_model.config.loops)):
        for _h_cycle in range(int(base_model.config.H_cycles)):
            for _l_cycle in range(int(base_model.config.L_cycles)):
                context_states = inner._inject_inputs(context_states, context_embeddings)
                for layer in inner.layers:
                    if casual_lm:
                        context_key, context_value = _project_key_value(layer, context_states, context_cos_sin)
                        attn_output = layer.self_attn.forward_packed(
                            cos_sin=context_cos_sin,
                            hidden_states=context_states,
                            cu_seqlens=cu_context,
                            max_seqlen=max_context_len,
                        )
                        attn_states = rms_norm(
                            context_states + attn_output,
                            variance_epsilon=layer.norm_eps,
                        )
                        mlp_output, context_ffn = _run_mlp_packed_with_ffn(layer, attn_states, cu_context)
                        context_states = rms_norm(attn_states + mlp_output, variance_epsilon=layer.norm_eps)
                        tail_indices = cu_context[1:].to(torch.long) - 1
                        context_ffn_tail = context_ffn[tail_indices] if tail_indices.numel() > 0 else None
                        caches.append(
                            _new_parallel_cache_entries(
                                layer,
                                context_states,
                                context_key,
                                context_value,
                                cu_context,
                                max_answer_lens,
                                context_ffn_tail=context_ffn_tail,
                            )
                        )
                    else:
                        context_states = layer.forward_packed(
                            cos_sin=context_cos_sin,
                            hidden_states=context_states,
                            cu_seqlens=cu_context,
                            max_seqlen=max_context_len,
                        )
                        context_key, context_value = _project_key_value(layer, context_states, context_cos_sin)
                        caches.append(
                            _new_parallel_cache_entries(
                                layer,
                                context_states,
                                context_key,
                                context_value,
                                cu_context,
                                max_answer_lens,
                            )
                        )
    return context_states, caches, cu_context


def _answer_input_embedding(base_model: Any, token: torch.Tensor) -> torch.Tensor:
    inner = base_model.inner
    token = token.view(1).to(device=inner.embed_tokens.embedding_weight.device, dtype=torch.int32)
    return inner.embed_scale * inner.embed_tokens(token)


def _answer_input_embeddings(base_model: Any, tokens: torch.Tensor) -> torch.Tensor:
    inner = base_model.inner
    tokens = tokens.to(device=inner.embed_tokens.embedding_weight.device, dtype=torch.int32)
    return inner.embed_scale * inner.embed_tokens(tokens)


def _decode_next_token_logits(
    base_model: Any,
    caches: List[_KVCacheEntry],
    token: torch.Tensor,
    token_cos_sin: Tuple[torch.Tensor, ...],
) -> torch.Tensor:
    state = base_model._empty_initial_hidden(
        (1, base_model.config.hidden_size),
        token.device,
    )
    token_embedding = _answer_input_embedding(base_model, token)
    cache_idx = 0
    for _outer_step in range(int(base_model.config.loops)):
        for _h_cycle in range(int(base_model.config.H_cycles)):
            for _l_cycle in range(int(base_model.config.L_cycles)):
                state = base_model.inner._inject_inputs(state, token_embedding)
                for layer in base_model.inner.layers:
                    state = _run_cached_answer_layer(layer, state, token_cos_sin, caches[cache_idx])
                    cache_idx += 1
    return base_model.inner.lm_head(state).squeeze(0)


def _run_cached_mlp_batched(
    layer: Any,
    states: torch.Tensor,
    entries: List[_KVCacheEntry],
) -> Tuple[torch.Tensor, torch.Tensor]:
    mlp = layer.mlp
    gate, up = mlp.gate_up_proj(states).chunk(2, dim=-1)
    current_ffn = F.silu(gate) * up
    previous_ffns = [entry.previous_ffn() for entry in entries]
    if previous_ffns and all(previous_ffn is not None for previous_ffn in previous_ffns):
        previous_ffn = torch.cat([previous_ffn for previous_ffn in previous_ffns if previous_ffn is not None], dim=0)
        conv_input = torch.stack([previous_ffn, current_ffn], dim=1)
    else:
        conv_input = current_ffn.unsqueeze(1)

    conv_output = mlp.dwconv(conv_input.transpose(1, 2).to(mlp.dwconv.weight.dtype))
    conv_output = conv_output[..., : conv_input.shape[1]]
    conv_output = mlp.act(conv_output[..., -1:].transpose(1, 2)).squeeze(1).contiguous()
    return mlp.down_proj(mlp.mlp_dropout(conv_output)), current_ffn


def _run_cached_answer_layer_batched(
    layer: Any,
    states: torch.Tensor,
    token_cos_sin: Tuple[torch.Tensor, ...],
    entries: List[_KVCacheEntry],
) -> torch.Tensor:
    query, current_key, current_value = _project_query_key_value(layer, states, token_cos_sin)
    key_chunks = []
    value_chunks = []
    key_lengths = []
    for sample_idx, entry in enumerate(entries):
        key, value = entry.key_value_with_current(
            current_key[sample_idx : sample_idx + 1],
            current_value[sample_idx : sample_idx + 1],
        )
        key_chunks.append(key)
        value_chunks.append(value)
        key_lengths.append(key.shape[0])

    key = torch.cat(key_chunks, dim=0)
    value = torch.cat(value_chunks, dim=0)
    cu_q = torch.arange(query.shape[0] + 1, dtype=torch.int32, device=query.device)
    key_lengths_tensor = torch.tensor(key_lengths, dtype=torch.int32, device=query.device)
    cu_k = F.pad(torch.cumsum(key_lengths_tensor, dim=0), (1, 0)).to(torch.int32)
    attn_output = flash_attn_varlen_func(
        q=query.contiguous(),
        k=key.contiguous(),
        v=value.contiguous(),
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=1,
        max_seqlen_k=int(key_lengths_tensor.max().item()) if key_lengths_tensor.numel() else 0,
        causal=False,
        window_size=(-1, -1),
        dropout_p=0.0,
    )
    if isinstance(attn_output, tuple):
        attn_output = attn_output[0]

    states = rms_norm(
        states + layer.self_attn.o_proj(attn_output.reshape(states.shape[0], layer.self_attn.output_size)),
        variance_epsilon=layer.norm_eps,
    )
    mlp_output, current_ffn = _run_cached_mlp_batched(layer, states, entries)
    states = rms_norm(states + mlp_output, variance_epsilon=layer.norm_eps)
    for sample_idx, entry in enumerate(entries):
        entry.append(
            current_key[sample_idx : sample_idx + 1],
            current_value[sample_idx : sample_idx + 1],
            current_ffn[sample_idx : sample_idx + 1],
        )
    return states


def _decode_next_token_logits_batched(
    base_model: Any,
    caches: List[List[_KVCacheEntry]],
    cache_indices: List[int],
    tokens: torch.Tensor,
    token_cos_sin: Tuple[torch.Tensor, ...],
) -> torch.Tensor:
    state = base_model._empty_initial_hidden(
        (tokens.shape[0], base_model.config.hidden_size),
        tokens.device,
    )
    token_embeddings = _answer_input_embeddings(base_model, tokens)
    cache_idx = 0
    for _outer_step in range(int(base_model.config.loops)):
        for _h_cycle in range(int(base_model.config.H_cycles)):
            for _l_cycle in range(int(base_model.config.L_cycles)):
                state = base_model.inner._inject_inputs(state, token_embeddings)
                for layer in base_model.inner.layers:
                    entries = [caches[cache_idx][sample_idx] for sample_idx in cache_indices]
                    state = _run_cached_answer_layer_batched(layer, state, token_cos_sin, entries)
                    cache_idx += 1
    return base_model.inner.lm_head(state)


def _parallel_context_and_answer_indices(
    base_model: Any,
    states: List[_ParallelSampleState],
) -> Tuple[torch.Tensor, List[List[int]]]:
    device = states[0].target_tokens.device if states else torch.device("cpu")
    prefix_len = int(base_model.inner.prefix_seq_len)
    context_indices = []
    answer_indices_by_sample = []
    offset = 0
    for state in states:
        context_len = int((~state.sample_answer_mask).sum().item())
        answer_len = int(state.target_positions.shape[0])
        context_total_len = prefix_len + context_len
        if context_total_len > 0:
            context_indices.append(
                torch.arange(offset, offset + context_total_len, dtype=torch.long, device=device)
            )
        answer_start = offset + context_total_len
        answer_indices_by_sample.append(
            [answer_start + token_idx for token_idx in range(answer_len)]
        )
        offset += context_total_len + answer_len

    if context_indices:
        return torch.cat(context_indices, dim=0), answer_indices_by_sample
    return torch.empty((0,), dtype=torch.long, device=device), answer_indices_by_sample


def _synthesize_prompt_positions(tokens: torch.Tensor) -> torch.Tensor:
    positions = torch.zeros((int(tokens.numel()), 4), dtype=torch.int32, device=tokens.device)
    row = 0
    col = 0
    pair_id = 0
    io_id = 0
    for idx, token in enumerate(tokens.to(torch.long).tolist()):
        positions[idx, 0] = pair_id
        positions[idx, 1] = io_id
        positions[idx, 2] = row
        positions[idx, 3] = col
        if int(token) == ARC_EOS_TOKEN_ID:
            row, col = _advance_arc_position(row, col, int(token))
        elif token >= 0:
            row, col = _advance_arc_position(row, col, int(token))
    return positions


def _init_casual_no_size_states(batch: Dict[str, torch.Tensor]) -> List[_CasualNoSizeState]:
    states: List[_CasualNoSizeState] = []
    seq_offsets = batch["seq_offsets"].to(torch.long)
    prompt_positions_all = batch.get("prompt_position_ids")
    source_inputs_all = batch.get("source_inputs", batch["inputs"])
    has_arc_identifiers = "arc_identifiers" in batch

    for sample_idx in range(int(batch["puzzle_identifiers"].shape[0])):
        start = int(seq_offsets[sample_idx].item())
        end = int(seq_offsets[sample_idx + 1].item())
        prompt_inputs = batch["inputs"][start:end]
        prompt_source_inputs = source_inputs_all[start:end]
        if prompt_positions_all is not None:
            prompt_positions = prompt_positions_all[start:end]
        else:
            prompt_positions = _synthesize_prompt_positions(prompt_inputs)
        states.append(
            _CasualNoSizeState(
                sample_idx=sample_idx,
                prompt_inputs=prompt_inputs,
                prompt_source_inputs=prompt_source_inputs,
                prompt_positions=prompt_positions,
                puzzle_identifier=batch["puzzle_identifiers"][sample_idx : sample_idx + 1],
                arc_identifier=(
                    batch["arc_identifiers"][sample_idx : sample_idx + 1]
                    if has_arc_identifiers
                    else None
                ),
                generated_tokens=torch.empty((0,), dtype=torch.long, device=batch["inputs"].device),
                q_halt_logits=torch.zeros((1,), dtype=torch.float32, device=batch["inputs"].device),
            )
        )
    return states


def _build_casual_no_size_generation_batch(
    states: List[_CasualNoSizeState],
    *,
    start_token_id: int,
) -> Dict[str, torch.Tensor]:
    input_chunks = []
    source_chunks = []
    position_chunks = []
    seq_lengths = []
    puzzle_identifiers = []
    arc_identifiers = []

    for state in states:
        answer_inputs = torch.cat(
            [
                torch.tensor([start_token_id], dtype=state.prompt_inputs.dtype, device=state.prompt_inputs.device),
                state.generated_tokens.to(dtype=state.prompt_inputs.dtype),
            ],
            dim=0,
        )
        answer_positions = _make_answer_positions_from_tokens(
            state.generated_tokens,
            state.prompt_positions,
            shifted=True,
        ).to(device=state.prompt_positions.device, dtype=state.prompt_positions.dtype)
        inputs = torch.cat([state.prompt_inputs, answer_inputs], dim=0)
        source_inputs = torch.cat([state.prompt_source_inputs, answer_inputs], dim=0)
        position_ids = torch.cat([state.prompt_positions, answer_positions], dim=0)
        input_chunks.append(inputs)
        source_chunks.append(source_inputs)
        position_chunks.append(position_ids)
        seq_lengths.append(int(inputs.numel()))
        puzzle_identifiers.append(state.puzzle_identifier)
        if state.arc_identifier is not None:
            arc_identifiers.append(state.arc_identifier)

    device = input_chunks[0].device if input_chunks else torch.device("cpu")
    seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.int32, device=device)
    batch = {
        "inputs": torch.cat(input_chunks, dim=0),
        "source_inputs": torch.cat(source_chunks, dim=0),
        "position_ids": torch.cat(position_chunks, dim=0),
        "seq_lengths": seq_lengths_tensor,
        "seq_offsets": F.pad(torch.cumsum(seq_lengths_tensor, dim=0), (1, 0)).to(torch.int32),
        "puzzle_identifiers": torch.cat(puzzle_identifiers, dim=0),
    }
    if arc_identifiers:
        batch["arc_identifiers"] = torch.cat(arc_identifiers, dim=0)
    return batch


def _final_casual_no_size_batch(
    state: _CasualNoSizeState,
) -> Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, torch.Tensor], Dict[str, float]]:
    answer_positions = _make_answer_positions_from_tokens(
        state.generated_tokens,
        state.prompt_positions,
        shifted=False,
    ).to(device=state.prompt_positions.device, dtype=state.prompt_positions.dtype)
    inputs = torch.cat([state.prompt_inputs, state.generated_tokens.to(dtype=state.prompt_inputs.dtype)], dim=0)
    source_inputs = torch.cat(
        [state.prompt_source_inputs, state.generated_tokens.to(dtype=state.prompt_source_inputs.dtype)],
        dim=0,
    )
    position_ids = torch.cat([state.prompt_positions, answer_positions], dim=0)
    prompt_len = int(state.prompt_inputs.numel())
    answer_len = int(state.generated_tokens.numel())
    answer_mask = torch.cat(
        [
            torch.zeros((prompt_len,), dtype=torch.bool, device=inputs.device),
            torch.ones((answer_len,), dtype=torch.bool, device=inputs.device),
        ],
        dim=0,
    )
    preds = torch.cat(
        [
            torch.zeros((prompt_len,), dtype=torch.long, device=inputs.device),
            state.generated_tokens.to(torch.long),
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
        "seq_offsets": _offsets(inputs.numel(), device=inputs.device),
        "puzzle_identifiers": state.puzzle_identifier,
    }
    if state.arc_identifier is not None:
        final_batch["arc_identifiers"] = state.arc_identifier

    preds_dict = {"preds": preds, "q_halt_logits": state.q_halt_logits}
    return final_batch, preds_dict, {"count": 0.0, "accuracy": 0.0, "exact_accuracy": 0.0, "steps": 0.0}


def _generate_casual_no_size_batch(
    base_model: Any,
    batch: Dict[str, torch.Tensor],
    *,
    start_token_id: int,
    end_token_id: int,
    max_new_tokens: int,
) -> List[Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, torch.Tensor], Dict[str, float]]]:
    states = _init_casual_no_size_states(batch)
    for _decode_step in range(max_new_tokens):
        active_states = [state for state in states if not state.finished]
        if not active_states:
            break

        gen_batch = _build_casual_no_size_generation_batch(
            active_states,
            start_token_id=start_token_id,
        )
        final_carry, final_outputs = _run_base_model_to_halt(base_model, gen_batch)
        logits = final_outputs["logits"]
        logit_indices = gen_batch["seq_offsets"].to(torch.long)[1:] - 1
        next_tokens = torch.argmax(logits[logit_indices], dim=-1).to(torch.long)
        q_halt_logits = final_outputs.get("q_halt_logits")
        if q_halt_logits is None:
            q_halt_logits = torch.zeros((len(active_states),), dtype=torch.float32, device=batch["inputs"].device)
        steps = None if final_carry is None else final_carry.steps

        for local_idx, state in enumerate(active_states):
            next_token = next_tokens[local_idx : local_idx + 1]
            state.q_halt_logits = q_halt_logits[local_idx : local_idx + 1].to(torch.float32)
            if steps is not None:
                state.steps = float(steps[local_idx].item())
            if int(next_token.item()) == int(end_token_id):
                state.finished = True
                continue
            state.generated_tokens = torch.cat([state.generated_tokens, next_token], dim=0)

    return [_final_casual_no_size_batch(state) for state in states]


def _generate_sample_cached(
    base_model: Any,
    batch: Dict[str, torch.Tensor],
    sample_idx: int,
    *,
    start_token_id: int,
) -> Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, torch.Tensor], Dict[str, float]]:
    target_tokens, target_positions, sample_answer_mask = _target_tokens_and_positions(batch, sample_idx)
    device = batch["inputs"].device
    if target_tokens.numel() == 0:
        return _generate_sample_slow(base_model, batch, sample_idx, start_token_id=start_token_id)

    context_batch = _build_context_batch(batch, sample_idx, sample_answer_mask)
    full_position_batch = _build_full_position_batch(batch, sample_idx, target_positions, sample_answer_mask)
    token_indices = base_model.inner._packed_data_token_indices(full_position_batch)
    cos_sin = base_model.inner._rotary_cos_sin_packed(full_position_batch, token_indices)
    context_total_len = base_model.inner.prefix_seq_len + int(context_batch["inputs"].numel())
    context_indices = torch.arange(context_total_len, dtype=torch.long, device=device)
    answer_start = context_total_len
    context_cos_sin = base_model.inner._slice_packed_cos_sin(cos_sin, context_indices)
    answer_cos_sins = [
        base_model.inner._slice_packed_cos_sin(
            cos_sin,
            torch.tensor([answer_start + token_idx], dtype=torch.long, device=device),
        )
        for token_idx in range(int(target_tokens.numel()))
    ]

    context_states, caches = _precompute_context_cache(
        base_model,
        context_batch,
        context_cos_sin,
        max_answer_len=int(target_tokens.numel()),
    )

    generated = torch.empty((0,), dtype=torch.long, device=device)
    input_token = torch.tensor([start_token_id], dtype=torch.long, device=device)
    for token_cos_sin in answer_cos_sins:
        logits = _decode_next_token_logits(
            base_model,
            caches,
            input_token,
            token_cos_sin,
        )
        next_token = torch.argmax(logits, dim=-1).view(1)
        generated = torch.cat([generated, next_token.to(torch.long)], dim=0)
        input_token = next_token.to(torch.long)

    final_batch, preds = _final_prediction_batch(
        batch,
        sample_idx,
        generated,
        target_positions,
        sample_answer_mask,
    )
    q_logits = base_model.inner.q_head(context_states[0:1]).to(torch.float32)
    preds["q_halt_logits"] = q_logits[..., 0]

    valid = target_tokens != IGNORE_LABEL_ID
    correct = (generated.to(target_tokens.dtype) == target_tokens) & valid
    valid_count = int(valid.sum().item())
    exact = bool(valid_count > 0 and correct.sum().item() == valid_count)
    metrics = {
        "count": 1.0 if valid_count > 0 else 0.0,
        "accuracy": float(correct.sum().item()) / max(valid_count, 1),
        "exact_accuracy": 1.0 if exact else 0.0,
        "steps": float(base_model.config.loops) if valid_count > 0 else 0.0,
    }
    return final_batch, preds, metrics


def _generate_batch_parallel_cached(
    base_model: Any,
    batch: Dict[str, torch.Tensor],
    *,
    start_token_id: int,
) -> List[Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, torch.Tensor], Dict[str, float]]]:
    device = batch["inputs"].device
    states: List[_ParallelSampleState] = []
    sample_count = int(batch["puzzle_identifiers"].shape[0])
    for sample_idx in range(sample_count):
        target_tokens, target_positions, sample_answer_mask = _target_tokens_and_positions(batch, sample_idx)
        states.append(
            _ParallelSampleState(
                sample_idx=sample_idx,
                target_tokens=target_tokens,
                target_positions=target_positions,
                sample_answer_mask=sample_answer_mask,
                generated_tokens=torch.empty((0,), dtype=torch.long, device=device),
                q_halt_logits=torch.zeros((1,), dtype=torch.float32, device=device),
            )
        )

    max_answer_len = max((int(state.target_tokens.numel()) for state in states), default=0)
    if max_answer_len > 0:
        context_batch = _build_parallel_context_batch(batch, states)
        full_position_batch = _build_parallel_full_position_batch(batch, states)
        token_indices = base_model.inner._packed_data_token_indices(full_position_batch)
        cos_sin = base_model.inner._rotary_cos_sin_packed(full_position_batch, token_indices)
        context_indices, answer_indices_by_sample = _parallel_context_and_answer_indices(base_model, states)
        context_cos_sin = base_model.inner._slice_packed_cos_sin(cos_sin, context_indices)
        context_states, caches, cu_context = _precompute_context_cache_parallel(
            base_model,
            context_batch,
            context_cos_sin,
            max_answer_lens=[int(state.target_tokens.numel()) for state in states],
        )
        q_logits = base_model.inner.q_head(context_states[cu_context[:-1].to(torch.long)]).to(torch.float32)
        for state, q_logit in zip(states, q_logits[..., 0]):
            if state.target_tokens.numel() > 0:
                state.q_halt_logits = q_logit.view(1)
                state.steps = float(base_model.config.loops)

        for decode_step in range(max_answer_len):
            active = [
                (sample_idx, state)
                for sample_idx, state in enumerate(states)
                if state.generated_tokens.numel() < state.target_tokens.numel()
            ]
            if not active:
                break

            cache_indices = [sample_idx for sample_idx, _state in active]
            input_tokens = torch.cat(
                [
                    torch.tensor([start_token_id], dtype=torch.long, device=device)
                    if state.generated_tokens.numel() == 0
                    else state.generated_tokens[-1:]
                    for _sample_idx, state in active
                ],
                dim=0,
            )
            answer_indices = torch.tensor(
                [answer_indices_by_sample[sample_idx][decode_step] for sample_idx, _state in active],
                dtype=torch.long,
                device=device,
            )
            token_cos_sin = base_model.inner._slice_packed_cos_sin(cos_sin, answer_indices)
            logits = _decode_next_token_logits_batched(
                base_model,
                caches,
                cache_indices,
                input_tokens,
                token_cos_sin,
            )
            next_tokens = torch.argmax(logits, dim=-1).to(torch.long)
            for local_idx, (_sample_idx, state) in enumerate(active):
                state.generated_tokens = torch.cat(
                    [state.generated_tokens, next_tokens[local_idx : local_idx + 1]],
                    dim=0,
                )

    results = []
    for state in states:
        final_batch, preds = _final_prediction_batch(
            batch,
            state.sample_idx,
            state.generated_tokens,
            state.target_positions,
            state.sample_answer_mask,
        )
        preds["q_halt_logits"] = state.q_halt_logits
        results.append(
            (
                final_batch,
                preds,
                _sample_metrics_from_generation(
                    state.target_tokens,
                    state.generated_tokens,
                    steps=state.steps,
                ),
            )
        )
    return results


def _generate_sample_slow(
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
            casual_lm=_is_casual_mode(base_model),
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


def _parallel_generation_batch(
    base_model: Any,
    batch: Dict[str, torch.Tensor],
    states: List[_ParallelSampleState],
    *,
    start_token_id: int,
) -> Dict[str, torch.Tensor]:
    casual_lm = _is_casual_mode(base_model)
    input_chunks = []
    source_chunks = []
    position_chunks = []
    answer_mask_chunks = []
    label_chunks = []
    seq_lengths = []
    label_lengths = []
    puzzle_identifiers = []
    arc_identifiers = []
    has_arc_identifiers = "arc_identifiers" in batch

    for state in states:
        start, end, _label_start, _label_end = _sample_ranges(batch, state.sample_idx)
        sample_inputs = batch["inputs"][start:end]
        sample_source_inputs = batch.get("source_inputs", batch["inputs"])[start:end]
        sample_positions = batch["position_ids"][start:end]
        context_mask = ~state.sample_answer_mask

        answer_len = int(state.generated_tokens.numel()) + 1
        answer_input = torch.empty((answer_len,), dtype=sample_inputs.dtype, device=sample_inputs.device)
        answer_input[0] = int(start_token_id)
        if state.generated_tokens.numel() > 0:
            answer_input[1:] = state.generated_tokens.to(dtype=sample_inputs.dtype)

        answer_positions = state.target_positions[:answer_len]
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

        input_chunks.append(inputs)
        source_chunks.append(source_inputs)
        position_chunks.append(position_ids)
        answer_mask_chunks.append(answer_mask)
        label_len = inputs.numel() if casual_lm else answer_len
        label_chunks.append(torch.full((label_len,), IGNORE_LABEL_ID, dtype=torch.int32, device=sample_inputs.device))
        seq_lengths.append(inputs.numel())
        label_lengths.append(label_len)
        puzzle_identifiers.append(batch["puzzle_identifiers"][state.sample_idx : state.sample_idx + 1])
        if has_arc_identifiers:
            arc_identifiers.append(batch["arc_identifiers"][state.sample_idx : state.sample_idx + 1])

    device = batch["inputs"].device
    seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.int32, device=device)
    label_lengths_tensor = torch.tensor(label_lengths, dtype=torch.int32, device=device)
    gen_batch: Dict[str, torch.Tensor] = {
        "inputs": torch.cat(input_chunks, dim=0),
        "labels": torch.cat(label_chunks, dim=0),
        "answer_mask": torch.cat(answer_mask_chunks, dim=0),
        "source_inputs": torch.cat(source_chunks, dim=0),
        "position_ids": torch.cat(position_chunks, dim=0),
        "seq_lengths": seq_lengths_tensor,
        "seq_offsets": F.pad(torch.cumsum(seq_lengths_tensor, dim=0), (1, 0)).to(torch.int32),
        "puzzle_identifiers": torch.cat(puzzle_identifiers, dim=0),
    }
    if not casual_lm:
        gen_batch["label_seq_lengths"] = label_lengths_tensor
        gen_batch["label_seq_offsets"] = F.pad(torch.cumsum(label_lengths_tensor, dim=0), (1, 0)).to(torch.int32)
    if has_arc_identifiers:
        gen_batch["arc_identifiers"] = torch.cat(arc_identifiers, dim=0)
    return gen_batch


def _sample_metrics_from_generation(
    target_tokens: torch.Tensor,
    generated_tokens: torch.Tensor,
    *,
    steps: float,
) -> Dict[str, float]:
    valid = target_tokens != IGNORE_LABEL_ID
    correct = (generated_tokens.to(target_tokens.dtype) == target_tokens) & valid
    valid_count = int(valid.sum().item())
    exact = bool(valid_count > 0 and correct.sum().item() == valid_count)
    return {
        "count": 1.0 if valid_count > 0 else 0.0,
        "accuracy": float(correct.sum().item()) / max(valid_count, 1),
        "exact_accuracy": 1.0 if exact else 0.0,
        "steps": steps if valid_count > 0 else 0.0,
    }


def _generate_batch_parallel_slow(
    base_model: Any,
    batch: Dict[str, torch.Tensor],
    *,
    start_token_id: int,
) -> List[Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, torch.Tensor], Dict[str, float]]]:
    device = batch["inputs"].device
    states: List[_ParallelSampleState] = []
    sample_count = int(batch["puzzle_identifiers"].shape[0])
    for sample_idx in range(sample_count):
        target_tokens, target_positions, sample_answer_mask = _target_tokens_and_positions(batch, sample_idx)
        states.append(
            _ParallelSampleState(
                sample_idx=sample_idx,
                target_tokens=target_tokens,
                target_positions=target_positions,
                sample_answer_mask=sample_answer_mask,
                generated_tokens=torch.empty((0,), dtype=torch.long, device=device),
                q_halt_logits=torch.zeros((1,), dtype=torch.float32, device=device),
            )
        )

    max_answer_len = max((int(state.target_tokens.numel()) for state in states), default=0)
    for _decode_step in range(max_answer_len):
        active_states = [
            state
            for state in states
            if state.generated_tokens.numel() < state.target_tokens.numel()
        ]
        if not active_states:
            break

        gen_batch = _parallel_generation_batch(
            base_model,
            batch,
            active_states,
            start_token_id=start_token_id,
        )
        final_carry, final_outputs = _run_base_model_to_halt(base_model, gen_batch)
        logits = final_outputs["logits"]
        if _is_casual_mode(base_model):
            logit_indices = gen_batch["seq_offsets"].to(torch.long)[1:] - 1
        else:
            logit_indices = gen_batch["label_seq_offsets"].to(torch.long)[1:] - 1
        next_tokens = torch.argmax(logits[logit_indices], dim=-1).to(torch.long)
        q_halt_logits = final_outputs.get("q_halt_logits")
        if q_halt_logits is None:
            q_halt_logits = torch.zeros((len(active_states),), dtype=torch.float32, device=device)

        steps = None if final_carry is None else final_carry.steps
        for local_idx, state in enumerate(active_states):
            state.generated_tokens = torch.cat(
                [state.generated_tokens, next_tokens[local_idx : local_idx + 1]],
                dim=0,
            )
            state.q_halt_logits = q_halt_logits[local_idx : local_idx + 1].to(torch.float32)
            if steps is not None:
                state.steps = float(steps[local_idx].item())

    results = []
    for state in states:
        final_batch, preds = _final_prediction_batch(
            batch,
            state.sample_idx,
            state.generated_tokens,
            state.target_positions,
            state.sample_answer_mask,
        )
        preds["q_halt_logits"] = state.q_halt_logits
        results.append(
            (
                final_batch,
                preds,
                _sample_metrics_from_generation(
                    state.target_tokens,
                    state.generated_tokens,
                    steps=state.steps,
                ),
            )
        )
    return results


def _generate_batch_parallel(
    base_model: Any,
    batch: Dict[str, torch.Tensor],
    *,
    start_token_id: int,
    use_kv_cache: bool,
) -> List[Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, torch.Tensor], Dict[str, float]]]:
    if use_kv_cache and "position_ids" in batch:
        return _generate_batch_parallel_cached(base_model, batch, start_token_id=start_token_id)
    return _generate_batch_parallel_slow(base_model, batch, start_token_id=start_token_id)


def _generate_sample(
    base_model: Any,
    batch: Dict[str, torch.Tensor],
    sample_idx: int,
    *,
    start_token_id: int,
    use_kv_cache: bool,
) -> Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, torch.Tensor], Dict[str, float]]:
    if use_kv_cache and "position_ids" in batch:
        return _generate_sample_cached(base_model, batch, sample_idx, start_token_id=start_token_id)
    return _generate_sample_slow(base_model, batch, sample_idx, start_token_id=start_token_id)


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
    start_token_id = int(getattr(config.arch, "causal_lm_start_token_id", 1))
    casual_lm = _is_casual_mode(base_model)
    configured_end_token_id = getattr(config.arch, "casual_lm_end_token_id", None)
    if configured_end_token_id is None:
        configured_end_token_id = getattr(getattr(base_model, "config", None), "casual_lm_end_token_id", None)
    end_token_id = (
        int(configured_end_token_id)
        if configured_end_token_id is not None
        else int(getattr(getattr(base_model, "config", None), "vocab_size", eval_metadata.vocab_size)) - 1
    )
    max_new_tokens = _resolve_casual_max_new_tokens(config)
    use_kv_cache = _supports_autoregressive_kv_cache(base_model)
    forward_mode = getattr(getattr(base_model, "config", None), "forward_mode", "autoregressive")
    if rank == 0:
        if casual_lm:
            print(
                f"Autoregressive casual evaluation stops on END_TOKEN={end_token_id} "
                f"with max_new_tokens={max_new_tokens}."
            )
        if use_kv_cache:
            print(f"Autoregressive {forward_mode} evaluation uses batch-parallel KV-cache decode.")
        else:
            print(
                f"Autoregressive {forward_mode} KV cache is not supported for this model; "
                "using batch-parallel full-prefix decode."
            )

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
            sample_label = "sample" if sample_count == 1 else "samples"
            set_id = set_ids[set_name]

            casual_no_size = (
                casual_lm
                and "labels" not in batch
                and "answer_mask" not in batch
                and "position_ids" not in batch
            )
            if sample_count == 0:
                sample_results = []
            elif casual_no_size:
                if rank == 0:
                    print(
                        f"    Generating {sample_count} {sample_label} "
                        "with END-token casual decode..."
                    )
                sample_results = _generate_casual_no_size_batch(
                    base_model,
                    batch,
                    start_token_id=start_token_id,
                    end_token_id=end_token_id,
                    max_new_tokens=max_new_tokens,
                )
            elif sample_count > 1:
                if rank == 0:
                    decode_mode = "KV-cache" if use_kv_cache and "position_ids" in batch else "full-prefix"
                    print(
                        f"    Generating {sample_count} {sample_label} "
                        f"with batch-parallel {decode_mode} decode..."
                    )
                sample_results = _generate_batch_parallel(
                    base_model,
                    batch,
                    start_token_id=start_token_id,
                    use_kv_cache=use_kv_cache,
                )
            else:
                if rank == 0:
                    print("    Generating sample 1/1...")
                sample_results = [
                    _generate_sample(
                        base_model,
                        batch,
                        0,
                        start_token_id=start_token_id,
                        use_kv_cache=use_kv_cache,
                    )
                ]

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
