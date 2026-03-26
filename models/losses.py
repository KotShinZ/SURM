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
import math


IGNORE_LABEL_ID = -100


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
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]

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
        labels = new_carry.current_data["labels"]

        B, SeqLen = labels.shape
        H = W = int(math.sqrt(SeqLen)) # 9x9グリッドを想定
        V = outputs["logits"].shape[-1] // 9 # 語彙数 (vocab_size)

        # ロジットを [B, SeqLen, 9マス, V] に変形
        logits_3x3 = outputs["logits"].view(B, SeqLen, 9, V) # (B, SeqLen, 9, V)
        # 代表値として「自分自身のマス（3x3の中心＝インデックス4）」のロジットを使用
        center_logits = logits_3x3[:, :, 4, :] # (B, SeqLen, V)

        # Correctness
        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(center_logits, dim=-1)

            # Correctness
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            use_act = getattr(getattr(self.model, "config", None), "use_act", True)
            metrics = {
                "count": valid_metrics.sum(),

                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }
            if use_act:
                metrics["q_halt_accuracy"] = (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum()

        # ==========================================
        # 1. 3x3 周囲マスの予測ロス (LM Loss)
        # ==========================================
        labels_2d = labels.view(B, H, W).float()
        # 画面外を IGNORE_LABEL_ID でパディングしてからUnfold
        labels_pad = F.pad(labels_2d.unsqueeze(1), (1, 1, 1, 1), value=IGNORE_LABEL_ID)
        labels_unfold = F.unfold(labels_pad, kernel_size=3).transpose(1, 2).long() # [B, SeqLen, 9]

        lm_loss_patch = self.loss_fn(logits_3x3.reshape(-1, V), labels_unfold.reshape(-1), ignore_index=IGNORE_LABEL_ID)
        lm_loss_patch = lm_loss_patch.view(B, SeqLen, 9)
        # 9マス分のロスを足し合わせる
        valid_patch_mask = (labels_unfold != IGNORE_LABEL_ID)
        lm_loss_patch = torch.where(valid_patch_mask, lm_loss_patch, 0.0)
        
        # 9マスのうち、有効なマスの数（パディング以外）をカウント
        valid_patch_counts = valid_patch_mask.sum(dim=-1).clamp_min(1) # [B, SeqLen]
        
        # 単純な sum ではなく、有効なマス数で割ってLossのスケールを元の実装に合わせる
        lm_loss = ((lm_loss_patch.sum(dim=-1) / valid_patch_counts) / loss_divisor).sum()
        metrics["lm_loss"] = lm_loss.detach()

        # ==========================================
        # 2. 重複マスの予測一貫性ロス (Consistency Loss)
        # ==========================================
        # Fold を使って、同じマスに対する異なるセルからの予測を足し合わせる
        logits_for_fold = logits_3x3.permute(0, 3, 2, 1).reshape(B, V * 9, SeqLen)
        summed_logits = F.fold(logits_for_fold, output_size=(H, W), kernel_size=3, padding=1) # [B, V, H, W]
        
        # 重複回数をカウントして平均を取る
        ones = torch.ones_like(logits_for_fold[:, :9, :])
        counts = F.fold(ones, output_size=(H, W), kernel_size=3, padding=1)
        mean_logits = summed_logits / counts # [B, V, H, W]
        
        # 再びUnfoldしてパッチの形に戻し、自身の予測と平均予測の誤差（MSE）を取る
        mean_unfolded = F.unfold(mean_logits, kernel_size=3, padding=1)
        mean_unfolded = mean_unfolded.view(B, V, 9, SeqLen).permute(0, 3, 2, 1) # [B, SeqLen, 9, V]
        
        valid_mask_9 = (labels_unfold != IGNORE_LABEL_ID).unsqueeze(-1).float()
        consistency_loss = F.mse_loss(logits_3x3, mean_unfolded.detach(), reduction='none')
        consistency_loss = (consistency_loss * valid_mask_9).sum() / valid_mask_9.sum().clamp_min(1)
        metrics["consistency_loss"] = consistency_loss.detach()

        # ==========================================
        
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

        # consistency_loss の重み（係数）は必要に応じて調整してください
        consistency_weight = 1.0
        total_loss = lm_loss + (consistency_weight * consistency_loss) + 0.5 * (q_halt_loss + q_continue_loss)
        
        if aux_loss is not None:
            total_loss = total_loss + aux_loss

        return (
            new_carry,
            total_loss,
            metrics,
            returned_outputs,
            new_carry.halted.all(),
        )