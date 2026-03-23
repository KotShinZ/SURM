from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from evaluate_trained_model import (
    IGNORE_LABEL_ID,
    _resolve_checkpoint_path,
    create_model_for_evaluation,
    create_test_dataloader,
    load_config_from_checkpoint_path,
    load_model_weights,
)


@dataclass
class TraceRecord:
    label: str
    preds: torch.Tensor
    logits: torch.Tensor
    batch_accuracy: float
    batch_exact_accuracy: float


def load_visualization_setup(
    checkpoint: str,
    *,
    split: str = "test",
    batch_size: int = 4096,
    loops: int = 1,
    h_cycles: Optional[int] = None,
    l_cycles: Optional[int] = None,
    data_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device_name)

    config = load_config_from_checkpoint_path(checkpoint)
    if data_path is not None:
        config.data_path = data_path
    config.global_batch_size = batch_size

    if config.arch.__pydantic_extra__ is None:
        config.arch.__pydantic_extra__ = {}
    config.arch.__pydantic_extra__["loops"] = loops
    if h_cycles is not None:
        config.arch.__pydantic_extra__["H_cycles"] = h_cycles
    if l_cycles is not None:
        config.arch.__pydantic_extra__["L_cycles"] = l_cycles

    dataloader = create_test_dataloader(config, split, config.global_batch_size)
    metadata = dataloader.dataset.metadata

    model = create_model_for_evaluation(config, metadata, device=torch_device)
    resolved_checkpoint = _resolve_checkpoint_path(checkpoint)
    if resolved_checkpoint is None:
        raise FileNotFoundError(f"Could not resolve checkpoint path from: {checkpoint}")
    resolved_step = load_model_weights(
        model,
        checkpoint_path=resolved_checkpoint,
        device=torch_device,
        strict=True,
    )
    model.eval()

    return {
        "config": config,
        "metadata": metadata,
        "dataloader": dataloader,
        "model": model,
        "device": torch_device,
        "step": resolved_step,
        "resolved_checkpoint": resolved_checkpoint,
    }


def get_batch(dataloader, device: torch.device, batch_index: int = 0) -> Dict[str, Any]:
    iterator = iter(dataloader)
    current = None
    for _ in range(batch_index + 1):
        current = next(iterator)
    assert current is not None
    set_name, batch, global_batch_size = current
    batch_gpu = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    return {
        "set_name": set_name,
        "batch_cpu": batch,
        "batch_gpu": batch_gpu,
        "global_batch_size": global_batch_size,
    }


def _compute_accuracy_stats(preds: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    valid_mask = labels != IGNORE_LABEL_ID
    valid_examples = valid_mask.any(dim=1)
    if not torch.any(valid_examples):
        return 0.0, 0.0

    preds = preds[valid_examples]
    labels = labels[valid_examples]
    valid_mask = valid_mask[valid_examples]

    token_counts = valid_mask.sum(dim=1).clamp_min(1)
    correct = valid_mask & (preds == labels)
    token_acc = (correct.to(torch.float32).sum(dim=1) / token_counts).mean().item()
    exact_acc = (correct.sum(dim=1) == token_counts).to(torch.float32).mean().item()
    return token_acc, exact_acc


@torch.inference_mode()
def trace_single_outer_loop(model, batch_gpu: Dict[str, torch.Tensor]) -> List[TraceRecord]:
    loss_head = model
    urm = loss_head.model
    inner = urm.inner

    with torch.device(str(batch_gpu["inputs"].device)):
        carry = loss_head.initial_carry(batch_gpu)

    carry_after_reset = inner.reset_carry(carry.halted, carry)
    current_data = {
        key: torch.where(
            carry.halted.view((-1,) + (1,) * (value.ndim - 1)),
            batch_gpu[key],
            value,
        )
        for key, value in carry.current_data.items()
    }

    seq_info = {"cos_sin": inner.rotary_emb()}
    input_embeddings = inner._input_embeddings(current_data["inputs"], current_data["puzzle_identifiers"])
    hidden_states = carry_after_reset.current_hidden

    records: List[TraceRecord] = []

    if inner.config.H_cycles > 1:
        for warmup_h in range(inner.config.H_cycles - 1):
            for cycle_idx in range(inner.config.L_cycles):
                hidden_states = hidden_states + input_embeddings
                for layer in inner.layers:
                    hidden_states = layer(hidden_states=hidden_states, **seq_info)
                logits = inner.lm_head(hidden_states)[:, inner.puzzle_emb_len:]
                preds = logits.argmax(dim=-1)
                batch_acc, batch_exact = _compute_accuracy_stats(preds, current_data["labels"])
                records.append(
                    TraceRecord(
                        label=f"H{warmup_h + 1}/L{cycle_idx + 1}",
                        preds=preds.detach().cpu(),
                        logits=logits.detach().cpu(),
                        batch_accuracy=batch_acc,
                        batch_exact_accuracy=batch_exact,
                    )
                )

    final_h_index = inner.config.H_cycles
    for cycle_idx in range(inner.config.L_cycles):
        hidden_states = hidden_states + input_embeddings
        for layer in inner.layers:
            hidden_states = layer(hidden_states=hidden_states, **seq_info)
        logits = inner.lm_head(hidden_states)[:, inner.puzzle_emb_len:]
        preds = logits.argmax(dim=-1)
        batch_acc, batch_exact = _compute_accuracy_stats(preds, current_data["labels"])
        records.append(
            TraceRecord(
                label=f"H{final_h_index}/L{cycle_idx + 1}",
                preds=preds.detach().cpu(),
                logits=logits.detach().cpu(),
                batch_accuracy=batch_acc,
                batch_exact_accuracy=batch_exact,
            )
        )

    return records


def token_grid_to_digits(tokens: torch.Tensor | np.ndarray) -> np.ndarray:
    array = np.asarray(tokens, dtype=np.int64).reshape(9, 9)
    digits = np.where(array >= 2, array - 1, 0)
    return digits


def build_trace_summary(batch_cpu: Dict[str, torch.Tensor], records: List[TraceRecord], sample_index: int) -> Dict[str, Any]:
    inputs = batch_cpu["inputs"][sample_index]
    labels = batch_cpu["labels"][sample_index]
    return {
        "input_digits": token_grid_to_digits(inputs),
        "label_digits": token_grid_to_digits(labels),
        "trace_digits": [token_grid_to_digits(record.preds[sample_index]) for record in records],
        "trace_labels": [record.label for record in records],
        "batch_accuracy": [record.batch_accuracy for record in records],
        "batch_exact_accuracy": [record.batch_exact_accuracy for record in records],
        "sample_index": sample_index,
    }


def _cell_backgrounds(pred_digits: np.ndarray, label_digits: np.ndarray, given_mask: np.ndarray) -> np.ndarray:
    colors = np.empty((9, 9), dtype=object)
    colors[:] = "#fffdf7"
    colors[given_mask] = "#dbeafe"

    predicted_mask = (~given_mask) & (pred_digits > 0)
    correct_mask = predicted_mask & (pred_digits == label_digits)
    incorrect_mask = predicted_mask & (pred_digits != label_digits)
    unresolved_mask = (~given_mask) & (pred_digits == 0)

    colors[correct_mask] = "#dcfce7"
    colors[incorrect_mask] = "#fee2e2"
    colors[unresolved_mask] = "#fef3c7"
    return colors


def _grid_table_html(
    digits: np.ndarray,
    *,
    title: str,
    given_mask: Optional[np.ndarray] = None,
    label_digits: Optional[np.ndarray] = None,
) -> str:
    if given_mask is None:
        colors = np.full((9, 9), "#ffffff", dtype=object)
    elif label_digits is None:
        colors = np.full((9, 9), "#fffdf7", dtype=object)
        colors[given_mask] = "#dbeafe"
    else:
        colors = _cell_backgrounds(digits, label_digits, given_mask)

    rows: List[str] = []
    for row in range(9):
        cells: List[str] = []
        for col in range(9):
            value = digits[row, col]
            display_value = "" if value <= 0 else str(int(value))
            if value > 0:
                if given_mask is not None and given_mask[row, col]:
                    text_color = "#1d4ed8"
                    font_weight = "700"
                elif label_digits is not None and value == label_digits[row, col]:
                    text_color = "#166534"
                    font_weight = "500"
                else:
                    text_color = "#b91c1c"
                    font_weight = "500"
            else:
                text_color = "#111827"
                font_weight = "400"

            border_top = "2.5px solid #111827" if row % 3 == 0 else "1px solid #d1d5db"
            border_left = "2.5px solid #111827" if col % 3 == 0 else "1px solid #d1d5db"
            border_bottom = "2.5px solid #111827" if row == 8 else ""
            border_right = "2.5px solid #111827" if col == 8 else ""
            style = [
                f"background:{colors[row, col]}",
                f"color:{text_color}",
                f"font-weight:{font_weight}",
                "width:34px",
                "height:34px",
                "text-align:center",
                "vertical-align:middle",
                "font-size:20px",
                "line-height:1",
                f"border-top:{border_top}",
                f"border-left:{border_left}",
            ]
            if border_bottom:
                style.append(f"border-bottom:{border_bottom}")
            if border_right:
                style.append(f"border-right:{border_right}")

            cells.append(f"<td style=\"{';'.join(style)}\">{escape(display_value)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        "<div style=\"display:flex;flex-direction:column;gap:8px;align-items:center;\">"
        f"<div style=\"font-size:15px;font-weight:700;text-align:center;white-space:pre-line;\">{escape(title)}</div>"
        "<table style=\"border-collapse:collapse;border-spacing:0;\">"
        + "".join(rows)
        + "</table></div>"
    )


def render_trace_summary_html(summary: Dict[str, Any], max_cols: int = 4) -> str:
    input_digits = summary["input_digits"]
    label_digits = summary["label_digits"]
    given_mask = input_digits > 0

    panels = [_grid_table_html(input_digits, title="Input", given_mask=given_mask)]
    for idx, (label, grid) in enumerate(zip(summary["trace_labels"], summary["trace_digits"])):
        batch_acc = summary["batch_accuracy"][idx]
        batch_exact = summary["batch_exact_accuracy"][idx]
        panels.append(
            _grid_table_html(
                grid,
                title=f"{label}\nacc={batch_acc:.3f}, exact={batch_exact:.3f}",
                given_mask=given_mask,
                label_digits=label_digits,
            )
        )
    panels.append(_grid_table_html(label_digits, title="Target"))

    return (
        "<div style=\"display:flex;flex-direction:column;gap:16px;\">"
        f"<div style=\"font-size:22px;font-weight:700;\">Sudoku solving trace for sample {summary['sample_index']}</div>"
        f"<div style=\"display:grid;grid-template-columns:repeat({max_cols}, minmax(220px, 1fr));gap:18px;align-items:start;\">"
        + "".join(panels)
        + "</div></div>"
    )


def display_trace_summary(summary: Dict[str, Any], max_cols: int = 4) -> str:
    html = render_trace_summary_html(summary, max_cols=max_cols)
    try:
        from IPython.display import HTML, display

        display(HTML(html))
    except Exception:
        pass
    return html
