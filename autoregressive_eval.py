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


@dataclass
class _KVCacheEntry:
    context_key: torch.Tensor
    context_value: torch.Tensor
    answer_key: torch.Tensor
    answer_value: torch.Tensor
    answer_ffn: torch.Tensor
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
            return None
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


def _new_cache_entry(
    layer: Any,
    context_states: torch.Tensor,
    context_cos_sin: Tuple[torch.Tensor, ...],
    max_answer_len: int,
) -> _KVCacheEntry:
    context_key, context_value = _project_key_value(layer, context_states, context_cos_sin)
    attn = layer.self_attn
    mlp = layer.mlp
    return _KVCacheEntry(
        context_key=context_key,
        context_value=context_value,
        answer_key=context_key.new_empty((max_answer_len, attn.num_key_value_heads, attn.head_dim)),
        answer_value=context_value.new_empty((max_answer_len, attn.num_key_value_heads, attn.head_dim)),
        answer_ffn=context_states.new_empty((max_answer_len, mlp.inter)),
    )


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


def _supports_prefix_lm_kv_cache(base_model: Any) -> bool:
    config = getattr(base_model, "config", None)
    inner = getattr(base_model, "inner", None)
    if config is None or inner is None:
        return False
    if getattr(config, "forward_mode", None) != "prefix_lm":
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


def _precompute_context_cache(
    base_model: Any,
    context_batch: Dict[str, torch.Tensor],
    context_cos_sin: Tuple[torch.Tensor, ...],
    max_answer_len: int,
) -> Tuple[torch.Tensor, List[_KVCacheEntry]]:
    inner = base_model.inner
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


def _answer_input_embedding(base_model: Any, token: torch.Tensor) -> torch.Tensor:
    inner = base_model.inner
    token = token.view(1).to(device=inner.embed_tokens.embedding_weight.device, dtype=torch.int32)
    return inner.embed_scale * inner.embed_tokens(token)


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
    use_kv_cache = _supports_prefix_lm_kv_cache(base_model)
    if rank == 0:
        if use_kv_cache:
            print("Autoregressive prefix-LM evaluation uses KV cache.")
        else:
            print("Autoregressive prefix-LM KV cache is not supported for this model; using full-prefix decode.")

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
                print(f"    Generating sample {sample_idx + 1}/{sample_count}...")
                final_batch, preds, sample_metrics = _generate_sample(
                    base_model,
                    batch,
                    sample_idx,
                    start_token_id=start_token_id,
                    use_kv_cache=use_kv_cache,
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
