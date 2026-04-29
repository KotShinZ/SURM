# from typing import Any, Tuple, Dict, Sequence, Optional

# import torch
# import torch.nn.functional as F
# from torch import nn
# import math

# IGNORE_LABEL_ID = -100


# def s(x, epsilon=1e-30):
#     return torch.where(
#         x<0,
#         1/(1-x+ epsilon),
#         x + 1
#     )


# def log_stablemax(x, dim=-1):
#     s_x = s(x)
#     return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


# def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
#     logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

#     if valid_mask is None:
#         valid_mask = (labels != ignore_index)
#     transformed_labels = torch.where(valid_mask, labels, 0)
#     prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

#     return -torch.where(valid_mask, prediction_logprobs, 0)


# def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
#     # Cast logits to f32
#     # Flatten logits
#     return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


# class ACTLossHead(nn.Module):
#     def __init__(self, model: nn.Module, loss_type: str):
#         super().__init__()
#         self.model = model
#         self.loss_fn = globals()[loss_type]
        
#     def initial_carry(self, *args, **kwargs):
#         return self.model.initial_carry(*args, **kwargs)  # type: ignore

#     def forward(
#         self,
#         return_keys: Sequence[str],
#         # Model args
#         **model_kwargs,
#     ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
#         # Model logits
#         # B x SeqLen x D
#         new_carry, outputs = self.model(**model_kwargs)
#         labels = new_carry.current_data["labels"]

#         with torch.no_grad():
#             # Preds
#             outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

#             # Correctness
#             mask = (labels != IGNORE_LABEL_ID)
#             loss_counts = mask.sum(-1)
#             loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

#             is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
#             seq_is_correct = is_correct.sum(-1) == loss_counts
            
#             # Metrics (halted)
#             valid_metrics = new_carry.halted & (loss_counts > 0)
#             metrics = {
#                 "count": valid_metrics.sum(),
                
#                 "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
#                 "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

#                 "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
#                 "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
#             }

#         # Losses

#         lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
#         q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
#         metrics.update({
#             "lm_loss": lm_loss.detach(),
#             "q_halt_loss": q_halt_loss.detach(),
#         })
#         # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
#         q_continue_loss = 0
#         if "target_q_continue" in outputs:
#             q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

#             metrics["q_continue_loss"] = q_continue_loss.detach()
#         # Filter outputs for return
#         detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

#         return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()

from typing import Any, Tuple, Dict, Set, Optional

import torch
import torch.nn.functional as F
import torch._dynamo as dynamo
from torch import nn


IGNORE_LABEL_ID = -100


def _packed_segment_ids(lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
    lengths = lengths.to(device=device, dtype=torch.long)
    return torch.repeat_interleave(
        torch.arange(lengths.shape[0], device=device, dtype=torch.long),
        lengths,
    )


def _packed_segment_sum(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    totals = torch.zeros((num_segments,), device=values.device, dtype=dtype)
    totals.scatter_add_(0, segment_ids, values.to(dtype))
    return totals


def _align_packed_logits_to_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    current_data: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if labels.ndim != 1 or logits.ndim < 2 or logits.shape[0] == labels.shape[0]:
        return logits, labels, torch.ones_like(labels, dtype=torch.bool)
    if "label_seq_lengths" not in current_data or "seq_lengths" not in current_data:
        return logits, labels, torch.ones_like(labels, dtype=torch.bool)

    seq_lengths = current_data["seq_lengths"].to(device=logits.device, dtype=torch.long)
    label_lengths = current_data["label_seq_lengths"].to(device=logits.device, dtype=torch.long)
    seq_offsets = current_data.get(
        "seq_offsets",
        F.pad(torch.cumsum(seq_lengths.to(torch.int32), dim=0), (1, 0)),
    ).to(device=logits.device, dtype=torch.long)
    label_offsets = current_data.get(
        "label_seq_offsets",
        F.pad(torch.cumsum(label_lengths.to(torch.int32), dim=0), (1, 0)),
    ).to(device=logits.device, dtype=torch.long)

    aligned_logits = logits.new_zeros((labels.shape[0], *logits.shape[1:]))
    loss_labels = labels.clone()
    copied_mask = torch.zeros_like(labels, dtype=torch.bool)
    for idx in range(label_lengths.shape[0]):
        copy_len = int(torch.minimum(seq_lengths[idx], label_lengths[idx]).item())
        if copy_len <= 0:
            continue
        src_start = int(seq_offsets[idx].item())
        dst_start = int(label_offsets[idx].item())
        aligned_logits[dst_start : dst_start + copy_len] = logits[src_start : src_start + copy_len]
        copied_mask[dst_start : dst_start + copy_len] = True

    loss_labels[~copied_mask] = IGNORE_LABEL_ID
    return aligned_logits, loss_labels, copied_mask


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        loss_type: str,
        diff_L_loss_enabled: bool = False,
        diff_L_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.diff_L_loss_enabled = diff_L_loss_enabled
        self.diff_L_loss_weight = diff_L_loss_weight

        model_config = getattr(self.model, "config", None)
        if model_config is not None and hasattr(model_config, "diff_L_loss_enabled"):
            model_config.diff_L_loss_enabled = diff_L_loss_enabled

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Set[str],
        # Model args
        return_raw_outputs: bool = False,
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        profile = outputs.get("profile")
        labels = outputs.get("loss_labels", new_carry.current_data["labels"])
        loss_labels = labels
        copied_label_mask = torch.ones_like(labels, dtype=torch.bool) if labels.ndim == 1 else None
        if labels.ndim == 1:
            aligned_logits, loss_labels, copied_label_mask = _align_packed_logits_to_labels(
                outputs["logits"],
                labels,
                new_carry.current_data,
            )
            outputs["logits"] = aligned_logits

        # Correctness
        use_act = getattr(getattr(self.model, "config", None), "use_act", True)
        if labels.ndim == 1:
            with torch.no_grad():
                outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)
                mask = labels != IGNORE_LABEL_ID
                if copied_label_mask is not None:
                    outputs["preds"] = torch.where(
                        copied_label_mask,
                        outputs["preds"],
                        torch.full_like(outputs["preds"], IGNORE_LABEL_ID),
                    )
                is_correct = mask & (outputs["preds"] == labels)

                lengths = outputs.get(
                    "loss_seq_lengths",
                    new_carry.current_data.get("label_seq_lengths", new_carry.current_data["seq_lengths"]),
                ).to(device=labels.device, dtype=torch.long)
                segment_ids = _packed_segment_ids(lengths, labels.device)
                loss_counts = _packed_segment_sum(mask, segment_ids, lengths.shape[0], torch.long)
                correct_counts = _packed_segment_sum(is_correct, segment_ids, lengths.shape[0], torch.long)

                seq_is_correct = correct_counts == loss_counts
                loss_divisor = loss_counts.clamp_min(1)

                valid_metrics = new_carry.halted & (loss_counts > 0)
                metrics = {
                    "count": valid_metrics.sum(),
                    "accuracy": torch.where(valid_metrics, correct_counts.to(torch.float32) / loss_divisor, 0).sum(),
                    "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                    "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
                }
                if use_act:
                    metrics["q_halt_accuracy"] = (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum()

            loss_values = self.loss_fn(outputs["logits"], loss_labels, ignore_index=IGNORE_LABEL_ID)
            token_divisor = loss_divisor.to(loss_values.dtype)[segment_ids]
            lm_loss = (loss_values / token_divisor).sum()
        else:
            with torch.no_grad():
                # Preds
                outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

                # Correctness
                mask = labels != IGNORE_LABEL_ID
                loss_counts = mask.sum(-1)
                loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

                is_correct = mask & (outputs["preds"] == labels)
                seq_is_correct = is_correct.sum(-1) == loss_counts

                # Metrics (halted)
                valid_metrics = new_carry.halted & (loss_counts > 0)
                metrics = {
                    "count": valid_metrics.sum(),

                    "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                    "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                    "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
                }
                if use_act:
                    metrics["q_halt_accuracy"] = (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum()

            # Losses
            # FIXME: Assuming the batch is always full
            lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
        metrics["lm_loss"] = lm_loss.detach()

        q_halt_loss = 0
        q_continue_loss = 0
        if use_act:
            q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
            metrics["q_halt_loss"] = q_halt_loss.detach()

            # Q continue (bootstrapping target loss)
            if "target_q_continue" in outputs:
                q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
                metrics["q_continue_loss"] = q_continue_loss.detach()

        aux_loss = outputs.get("moe_aux_loss")
        if aux_loss is not None:
            metrics["moe_aux_loss"] = aux_loss.detach()

        router_metrics = outputs.get("router_metrics")
        if router_metrics is not None:
            for k, v in router_metrics.items():
                metrics[f"router/{k}"] = v.detach()

        if profile is not None:
            for name, duration in profile.items():
                metrics[f"profile/{name}"] = torch.tensor(duration, device=labels.device)

        # Filter outputs for return
        returned_outputs: Dict[str, torch.Tensor] = {}
        if return_raw_outputs:
            returned_outputs["raw_outputs"] = outputs

        for k in return_keys:
            if k in outputs:
                returned_outputs[k] = outputs[k].detach()

        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        if aux_loss is not None:
            total_loss = total_loss + aux_loss
            
        diff_L_loss = outputs.get("diff_L")
        if self.diff_L_loss_enabled and diff_L_loss is not None:
            diff_L_loss_mean = diff_L_loss.mean() * self.diff_L_loss_weight
            total_loss = total_loss + diff_L_loss_mean
            metrics["diff_L_loss"] = diff_L_loss_mean.detach()

        return (
            new_carry,
            total_loss,
            metrics,
            returned_outputs,
            new_carry.halted.all(),
        )
