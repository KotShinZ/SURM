from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import torch

from evaluators.arc import ARC
from pretrain import (
    PretrainConfig,
    TrainState,
    _get_loop_config,
    _load_model_state,
    _resolve_checkpoint_path,
    create_dataloader,
    create_evaluators,
    evaluate,
    init_train_state,
    load_config_from_checkpoint_path,
)


def load_config_from_checkpoint(checkpoint: str) -> PretrainConfig:
    config = load_config_from_checkpoint_path(checkpoint)
    if config is None:
        raise FileNotFoundError(f"Could not load a saved config for checkpoint: {checkpoint}")
    return config


def _checkpoint_step(checkpoint_path: Path) -> int:
    match = re.match(r"step_(\d+)(?:\.pt)?$", checkpoint_path.name)
    if match is not None:
        return int(match.group(1))
    return 0


def _set_arch_extra(config: PretrainConfig, key: str, value) -> None:
    extra = config.arch.__pydantic_extra__
    if extra is None:
        config.arch.__pydantic_extra__ = {}
        extra = config.arch.__pydantic_extra__
    extra[key] = value


def apply_eval_overrides(
    config: PretrainConfig,
    *,
    batch_size: int,
    loops: Optional[int],
    hidden_diff_threshold: Optional[float],
) -> None:
    config.global_batch_size = batch_size
    config.grad_accum_steps = 1
    config.load_checkpoint = None
    config.load_checkpoint_file = None
    config.load_optimizer_state = False

    if loops is not None:
        _set_arch_extra(config, "loops", loops)

    if hidden_diff_threshold is not None:
        _set_arch_extra(config, "norm_diff_min", hidden_diff_threshold)
        _set_arch_extra(config, "norm_diff_max", hidden_diff_threshold)


def _sample_count(batch: Dict[str, torch.Tensor]) -> int:
    if "puzzle_identifiers" in batch:
        return int(batch["puzzle_identifiers"].shape[0])
    return int(batch["inputs"].shape[0])


def _truncate_batch(batch: Dict[str, torch.Tensor], sample_count: int) -> Dict[str, torch.Tensor]:
    current_count = _sample_count(batch)
    if sample_count >= current_count:
        return batch

    if "seq_lengths" not in batch:
        return {
            key: value[:sample_count] if value.shape[:1] == (current_count,) else value
            for key, value in batch.items()
        }

    seq_lengths = batch["seq_lengths"][:sample_count]
    token_count = int(seq_lengths.sum().item())
    truncated: Dict[str, torch.Tensor] = {}
    original_token_count = int(batch["seq_lengths"].sum().item())

    for key, value in batch.items():
        if key == "seq_offsets":
            offsets = torch.zeros(sample_count + 1, dtype=value.dtype, device=value.device)
            offsets[1:] = torch.cumsum(seq_lengths.to(value.dtype), dim=0)
            truncated[key] = offsets
        elif value.shape[:1] == (current_count,):
            truncated[key] = value[:sample_count]
        elif value.shape[:1] == (original_token_count,):
            truncated[key] = value[:token_count]
        else:
            truncated[key] = value

    return truncated


class MaxProblemsDataLoader:
    def __init__(self, dataloader, max_problems: Optional[int]):
        self.dataloader = dataloader
        self.max_problems = max_problems

    def __len__(self) -> int:
        if self.max_problems is None:
            return len(self.dataloader)

        batch_size = getattr(self.dataloader.dataset.config, "global_batch_size", None)
        if batch_size is None or batch_size <= 0:
            return len(self.dataloader)
        return min(len(self.dataloader), math.ceil(self.max_problems / batch_size))

    def __iter__(self) -> Iterator[Tuple[str, Dict[str, torch.Tensor], int]]:
        processed = 0
        for set_name, batch, global_batch_size in self.dataloader:
            if self.max_problems is not None:
                remaining = self.max_problems - processed
                if remaining <= 0:
                    break
                batch_count = _sample_count(batch)
                if batch_count > remaining:
                    batch = _truncate_batch(batch, remaining)
                    batch_count = remaining
                global_batch_size = min(global_batch_size, batch_count)
            else:
                batch_count = _sample_count(batch)

            processed += batch_count
            yield set_name, batch, global_batch_size


def _load_weights_only(train_state: TrainState, config: PretrainConfig, checkpoint_path: Path) -> None:
    print(f"Loading checkpoint {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        step = checkpoint.get("step")
    else:
        state_dict = checkpoint
        step = None

    _load_model_state(train_state, config, state_dict, rank=0)
    train_state.step = int(step) if step is not None else _checkpoint_step(checkpoint_path)
    train_state.carry = None
    train_state.accum_carries = None
    train_state.accum_metrics = None


def _metrics_output_path(
    checkpoint_path: Path,
    *,
    max_problems: Optional[int],
    loops: Optional[int],
    hidden_diff_threshold: Optional[float],
) -> Path:
    checkpoint_dir = checkpoint_path.parent
    step = _checkpoint_step(checkpoint_path)
    max_label = "all" if max_problems is None else str(max_problems)
    loop_label = "config" if loops is None else str(loops)
    threshold_label = (
        "config"
        if hidden_diff_threshold is None
        else str(hidden_diff_threshold).replace(".", "_")
    )
    return checkpoint_dir / (
        f"step_{step}_evaluation_test_max_problems_{max_label}"
        f"_loops_{loop_label}_hidden_diff_threshold_{threshold_label}.json"
    )


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def maybe_create_arc_evaluator(config: PretrainConfig, eval_metadata):
    required_files = (
        Path(config.data_path) / "identifiers.json",
        Path(config.data_path) / "test_puzzles.json",
    )
    if not all(path.is_file() for path in required_files):
        return None
    return ARC(data_path=config.data_path, eval_metadata=eval_metadata)


class ARCThirtyByThirtyAdapter:
    required_outputs = ARC.required_outputs

    def __init__(self, evaluator: ARC, *, drop_labels: bool = False):
        self.evaluator = evaluator
        self.drop_labels = drop_labels

    def begin_eval(self):
        self.evaluator.begin_eval()

    def result(self, *args, **kwargs):
        return self.evaluator.result(*args, **kwargs)

    @staticmethod
    def _canvas_from_positions(tokens: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        canvas = torch.zeros((30 * 30,), dtype=tokens.dtype, device=tokens.device)
        if tokens.numel() == 0:
            return canvas

        rows = position_ids[:, -2].to(torch.long)
        cols = position_ids[:, -1].to(torch.long)
        valid = (rows >= 0) & (rows < 30) & (cols >= 0) & (cols < 30)
        canvas[rows[valid] * 30 + cols[valid]] = tokens[valid]
        return canvas

    @staticmethod
    def _target_output_mask(
        sample_inputs: torch.Tensor,
        sample_source_inputs: torch.Tensor,
        sample_positions: torch.Tensor,
        sample_labels: torch.Tensor,
    ) -> torch.Tensor:
        output_mask = sample_positions[:, 1] == 1
        pair_ids = torch.unique(sample_positions[output_mask, 0], sorted=True)
        label_len = int(sample_labels.numel())

        candidates = []
        changed_candidates = []
        matching_candidates = []
        for pair_id in pair_ids.tolist():
            mask = output_mask & (sample_positions[:, 0] == int(pair_id))
            if int(mask.sum().item()) != label_len:
                continue

            candidates.append(mask)
            if bool((sample_inputs[mask] != sample_source_inputs[mask]).any().item()):
                changed_candidates.append(mask)
            if label_len == 0 or bool((sample_source_inputs[mask] == sample_labels).all().item()):
                matching_candidates.append(mask)

        if len(changed_candidates) == 1:
            return changed_candidates[0]

        changed_matching = [
            mask
            for mask in changed_candidates
            if label_len == 0 or bool((sample_source_inputs[mask] == sample_labels).all().item())
        ]
        if len(changed_matching) == 1:
            return changed_matching[0]

        if matching_candidates:
            return matching_candidates[-1]
        if candidates:
            return candidates[-1]

        raise ValueError(
            "Could not align answer-only ARC labels with a target output slot: "
            f"label_len={label_len}, output_slots={int(pair_ids.numel())}."
        )

    def _to_fixed_30x30(
        self,
        batch: Dict[str, torch.Tensor],
        preds: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        labels = batch.get("labels")
        if labels is None:
            return batch, preds
        pred_tokens = preds["preds"]
        if (
            "label_seq_offsets" not in batch
            or labels.ndim != 1
            or pred_tokens.ndim != 1
            or labels.shape[0] == batch["inputs"].shape[0]
        ):
            return batch, preds

        seq_offsets = batch["seq_offsets"]
        label_offsets = batch["label_seq_offsets"]
        identifiers = batch.get("arc_identifiers", batch["puzzle_identifiers"])
        source_inputs = batch.get("source_inputs", batch["inputs"])

        input_canvases = []
        label_canvases = []
        pred_canvases = []

        for sample_idx in range(int(identifiers.shape[0])):
            start = int(seq_offsets[sample_idx].item())
            end = int(seq_offsets[sample_idx + 1].item())
            label_start = int(label_offsets[sample_idx].item())
            label_end = int(label_offsets[sample_idx + 1].item())

            sample_inputs = batch["inputs"][start:end]
            sample_source_inputs = source_inputs[start:end]
            sample_positions = batch["position_ids"][start:end]
            sample_labels = labels[label_start:label_end]
            sample_preds = pred_tokens[label_start:label_end]

            target_output_mask = self._target_output_mask(
                sample_inputs,
                sample_source_inputs,
                sample_positions,
                sample_labels,
            )
            target_pair_id = sample_positions[target_output_mask][0, 0]
            target_input_mask = (
                (sample_positions[:, 0] == target_pair_id)
                & (sample_positions[:, 1] == 0)
            )
            target_output_positions = sample_positions[target_output_mask]

            if sample_labels.numel() != target_output_positions.shape[0]:
                raise ValueError(
                    "Answer-only ARC label length does not match target output positions: "
                    f"sample={sample_idx}, labels={sample_labels.numel()}, "
                    f"positions={target_output_positions.shape[0]}."
                )

            input_canvases.append(
                self._canvas_from_positions(
                    sample_source_inputs[target_input_mask],
                    sample_positions[target_input_mask],
                )
            )
            label_canvases.append(self._canvas_from_positions(sample_labels, target_output_positions))
            pred_canvases.append(self._canvas_from_positions(sample_preds, target_output_positions))

        fixed_batch = {
            "inputs": torch.stack(input_canvases, dim=0),
            "source_inputs": torch.stack(input_canvases, dim=0),
            "labels": torch.stack(label_canvases, dim=0),
            "puzzle_identifiers": identifiers,
        }
        fixed_preds = {
            "preds": torch.stack(pred_canvases, dim=0),
            "q_halt_logits": preds["q_halt_logits"],
        }
        return fixed_batch, fixed_preds

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        fixed_batch, fixed_preds = self._to_fixed_30x30(batch, preds)
        if self.drop_labels:
            fixed_batch = dict(fixed_batch)
            fixed_batch["labels"] = None
        self.evaluator.update_batch(fixed_batch, fixed_preds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved URM checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file or directory.")
    parser.add_argument("--batch_size", type=int, default=4096, help="Evaluation global batch size.")
    parser.add_argument("--hidden_diff_threshold", type=float, default=None, help="Override norm_diff_min/max.")
    parser.add_argument("--loops", type=int, default=None, help="Override model loop count.")
    parser.add_argument("--max_problems", type=int, default=None, help="Maximum number of test problems to evaluate.")
    parser.add_argument("--output", type=str, default=None, help="Optional metrics JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.max_problems is not None and args.max_problems <= 0:
        raise ValueError("--max_problems must be positive when provided.")

    resolved_checkpoint = _resolve_checkpoint_path(args.checkpoint)
    if resolved_checkpoint is None:
        raise FileNotFoundError(f"Could not resolve checkpoint path from: {args.checkpoint}")
    checkpoint_path = Path(resolved_checkpoint)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because pretrain.py creates and loads the model on CUDA.")

    config = load_config_from_checkpoint(args.checkpoint)
    apply_eval_overrides(
        config,
        batch_size=args.batch_size,
        loops=args.loops,
        hidden_diff_threshold=args.hidden_diff_threshold,
    )
    config.checkpoint_path = str(checkpoint_path.parent)

    torch.random.manual_seed(config.seed)

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1,
    )
    eval_loader, eval_metadata = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1,
    )
    eval_loader = MaxProblemsDataLoader(eval_loader, args.max_problems)
    padding_eval_without_labels = bool(config.padding and eval_metadata.variable_seq_lengths)

    train_state = init_train_state(config, train_metadata, rank=0, world_size=1)
    _load_weights_only(train_state, config, checkpoint_path)
    train_state.model.eval()

    loop_config = _get_loop_config(train_state.model)
    if loop_config is not None:
        if args.loops is not None:
            loop_config.loops = args.loops
        if args.hidden_diff_threshold is not None:
            loop_config.norm_diff_min = args.hidden_diff_threshold
            loop_config.norm_diff_max = args.hidden_diff_threshold

    evaluators = create_evaluators(config, eval_metadata)
    if not any(isinstance(evaluator, ARC) for evaluator in evaluators):
        arc_evaluator = maybe_create_arc_evaluator(config, eval_metadata)
        if arc_evaluator is not None:
            print("Detected ARC-AGI dataset; enabling ARC augment consensus evaluator.")
            evaluators.append(arc_evaluator)
    evaluators = [
        ARCThirtyByThirtyAdapter(evaluator, drop_labels=padding_eval_without_labels)
        if isinstance(evaluator, ARC)
        else evaluator
        for evaluator in evaluators
    ]

    metrics = evaluate(
        config,
        train_state,
        eval_loader,
        eval_metadata,
        evaluators,
        rank=0,
        world_size=1,
        cpu_group=None,
    )

    output_path = (
        Path(args.output)
        if args.output is not None
        else _metrics_output_path(
            checkpoint_path,
            max_problems=args.max_problems,
            loops=args.loops,
            hidden_diff_threshold=args.hidden_diff_threshold,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": train_state.step,
        "batch_size": args.batch_size,
        "max_problems": args.max_problems,
        "loops": getattr(loop_config, "loops", args.loops),
        "hidden_diff_threshold": args.hidden_diff_threshold,
        "metrics": _jsonable(metrics),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved metrics to {output_path}")

    # Keep worker shutdown deterministic before process exit.
    del train_loader
    del eval_loader


if __name__ == "__main__":
    main()
