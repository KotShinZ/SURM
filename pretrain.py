from typing import Optional, Any, Sequence, List, Tuple, Dict, Literal
from dataclasses import dataclass, replace
import os
import math
import json
import sys
import yaml
import shutil
import re
import copy
import time
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig, OmegaConf
#from adam_atan2 import AdamATan2
from adam_atan2_pytorch import AdamAtan2
from models.muon import Muon
from puzzle_dataset import (
    ARCOutputMaskConfig,
    MaskedInputConfig,
    PuzzleDataset,
    PuzzleDatasetConfig,
    PuzzleDatasetSeparate,
    PuzzleDatasetMetadata,
)
from puzzle_full_dataset import PuzzleFullDataset
from data.online_aug import OnlineAugConfig
from utils import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from logger import global_logger


class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


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
    # Config
    arch: ArchConfig
    # Data
    data_path: str
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    target_q_update_every: int

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Gradient accumulation
    grad_accum_steps: int = 1

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    load_checkpoint: Optional[str] = None
    load_checkpoint_file: Optional[str] = None
    load_strict: bool = True
    load_optimizer_state: bool = True
    torch_compile: bool = True

    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_first: bool = False
    eval_save_outputs: List[str] = []

    loop_deltas: List[str] = []

    ema: bool = False
    ema_rate: float = 0.999

    use_muon: bool = False

    data_fraction: float = 1.0  # Fraction of training data to use per epoch (1.0 = all, 0.5 = half)
    examples_per_puzzle: Optional[int] = 1

    # Online augmentation (applied per batch during training only)
    online_aug: Optional[OnlineAugConfig] = None

    # Replace model inputs with a randomly masked version of the labels.
    masked_input: Optional[MaskedInputConfig] = None

    # ARC full-context training: pick one solved output pair per sample and mask it on the fly.
    arc_output_mask: Optional[ARCOutputMaskConfig] = None

    # Build full-context ARC samples on the fly from one-pair examples.
    mask_full_training: bool = False
    label_separate: bool = False
    SeparateMode: str = "D"
    separate_mode: Optional[str] = None
    label_separate_noise_token_min: int = 2
    label_separate_noise_token_max: Optional[int] = None
    label_separate_C_noise_scale: float = 1.0
    full_min_pairs: int = 3
    full_max_pairs: int = 8
    full_answer_initial_mode: Literal["black", "noised_label"] = "black"
    full_answer_initial_black_token_id: int = 2
    full_answer_initial_gamma_min: float = 1.0
    full_answer_initial_gamma_max: float = 1.0
    full_answer_initial_noise_token_min: int = 2
    full_answer_initial_noise_token_max: int = 11

    # Benchmark a fixed number of optimizer steps and exit without wandb/eval/checkpointing.
    benchmark_steps: int = 0
    benchmark_warmup_steps: int = 1

    # Training path that probes halting without gradients, then replays only
    # halted samples with gradients so backward kernels run on a smaller batch.
    halted_replay_training: bool = False



@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int
    accum_step: int = 0
    accum_carries: Optional[List[Any]] = None
    accum_metrics: Optional[Dict[str, torch.Tensor]] = None
    train_time_h: float = 0.0
    last_step_time_h: Optional[float] = None


class ShuffledTestPuzzleDataset(PuzzleDataset):
    def _iter_test(self):
        rng = np.random.Generator(np.random.Philox(seed=self.config.seed + 10_000 + self.config.rank))

        for set_name, dataset in self._data.items():  # type: ignore
            total_examples = (
                dataset["seq_offsets"].size - 1
                if self.metadata.variable_seq_lengths
                else len(dataset["inputs"])
            )
            shuffled_indices = rng.permutation(total_examples)

            start_index = 0
            while start_index < total_examples:
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                global_indices = shuffled_indices[start_index:end_index]

                local_start = self.config.rank * self.local_batch_size
                local_end = min((self.config.rank + 1) * self.local_batch_size, global_indices.size)
                local_indices = global_indices[local_start:local_end]
                puzzle_indices = np.searchsorted(dataset["puzzle_indices"], local_indices, side="right") - 1

                batch_fields = self._select_examples(dataset, local_indices)
                batch_fields["puzzle_identifiers"] = dataset["puzzle_identifiers"][puzzle_indices]
                batch = self._collate_batch(batch_fields, rng, make_masked_inputs=False)

                yield set_name, batch, end_index - start_index

                start_index += self.config.global_batch_size


class ShuffledTestPuzzleSeparateDataset(PuzzleDatasetSeparate):
    def _iter_test(self):
        rng = np.random.Generator(np.random.Philox(seed=self.config.seed + 10_000 + self.config.rank))

        for set_name, dataset in self._data.items():  # type: ignore
            total_examples = (
                dataset["seq_offsets"].size - 1
                if self.metadata.variable_seq_lengths
                else len(dataset["inputs"])
            )
            shuffled_indices = rng.permutation(total_examples)

            start_index = 0
            while start_index < total_examples:
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                global_indices = shuffled_indices[start_index:end_index]

                local_start = self.config.rank * self.local_batch_size
                local_end = min((self.config.rank + 1) * self.local_batch_size, global_indices.size)
                local_indices = global_indices[local_start:local_end]
                puzzle_indices = np.searchsorted(dataset["puzzle_indices"], local_indices, side="right") - 1

                batch_fields = self._select_examples(dataset, local_indices)
                batch_fields["puzzle_identifiers"] = dataset["puzzle_identifiers"][puzzle_indices]
                batch = self._collate_batch(batch_fields, rng, make_masked_inputs=False)

                yield set_name, batch, end_index - start_index

                start_index += self.config.global_batch_size


class ShuffledTestPuzzleFullDataset(PuzzleFullDataset):
    def _iter_test(self):
        rng = np.random.Generator(np.random.Philox(seed=self.config.seed + 10_000 + self.config.rank))

        for set_name, dataset in self._data.items():  # type: ignore
            total_examples = dataset["seq_offsets"].size - 1
            shuffled_indices = rng.permutation(total_examples)

            start_index = 0
            while start_index < total_examples:
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                global_indices = shuffled_indices[start_index:end_index]

                local_start = self.config.rank * self.local_batch_size
                local_end = min((self.config.rank + 1) * self.local_batch_size, global_indices.size)
                if local_start >= local_end:
                    break

                local_indices = global_indices[local_start:local_end]
                puzzle_indices = np.searchsorted(dataset["puzzle_indices"], local_indices, side="right") - 1
                samples = [self._build_test_sample(dataset, int(example_index), rng=rng) for example_index in local_indices]
                batch = self._collate_built_samples(
                    samples,
                    dataset["puzzle_identifiers"][puzzle_indices],
                )

                yield set_name, batch, end_index - start_index
                start_index += self.config.global_batch_size


def _label_separate_enabled(config: PretrainConfig) -> bool:
    return bool(config.label_separate or getattr(config.arch, "label_separate", False))


def _config_or_arch_extra(config: PretrainConfig, key: str):
    return getattr(config.arch, key, getattr(config, key))


def _separate_mode(config: PretrainConfig) -> str:
    mode = getattr(
        config.arch,
        "SeparateMode",
        getattr(config.arch, "separate_mode", config.separate_mode or config.SeparateMode),
    )
    return str(mode).upper()


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    is_test = kwargs.get("test_set_mode", False)
    label_separate = _label_separate_enabled(config)
    data_fraction = config.data_fraction if not is_test else 1.0
    # Apply online augmentation only during training
    online_aug = config.online_aug if not is_test else None
    # Keep dynamic ARC masking strictly on the training path.
    arc_output_mask = config.arc_output_mask if not is_test else None
    if config.mask_full_training and label_separate:
        raise ValueError("label_separate cannot be combined with mask_full_training.")
    if label_separate and online_aug is not None and online_aug.enabled:
        raise ValueError("label_separate cannot be combined with online_aug.")
    if config.mask_full_training:
        dataset_cls = ShuffledTestPuzzleFullDataset if is_test else PuzzleFullDataset
    elif label_separate:
        dataset_cls = ShuffledTestPuzzleSeparateDataset if is_test else PuzzleDatasetSeparate
    else:
        dataset_cls = ShuffledTestPuzzleDataset if is_test else PuzzleDataset
    dataset = dataset_cls(
        PuzzleDatasetConfig(
            seed=config.seed, dataset_path=config.data_path, rank=rank, num_replicas=world_size,
            data_fraction=data_fraction,
            grad_accum_steps=max(1, config.grad_accum_steps) if not is_test else 1,
            examples_per_puzzle=config.examples_per_puzzle,
            online_aug=online_aug,
            masked_input=config.masked_input,
            arc_output_mask=arc_output_mask,
            full_min_pairs=config.full_min_pairs,
            full_max_pairs=config.full_max_pairs,
            full_answer_initial_mode=config.full_answer_initial_mode,
            full_answer_initial_black_token_id=config.full_answer_initial_black_token_id,
            full_answer_initial_gamma_min=config.full_answer_initial_gamma_min,
            full_answer_initial_gamma_max=config.full_answer_initial_gamma_max,
            full_answer_initial_noise_token_min=config.full_answer_initial_noise_token_min,
            full_answer_initial_noise_token_max=config.full_answer_initial_noise_token_max,
            answer_only_labels=bool(getattr(config.arch, "answer_only", False)),
            label_separate=label_separate,
            SeparateMode=_separate_mode(config),
            label_separate_noise_token_min=int(_config_or_arch_extra(config, "label_separate_noise_token_min")),
            label_separate_noise_token_max=_config_or_arch_extra(config, "label_separate_noise_token_max"),
            **kwargs,
        ),
        split=split,
    )
    print(f"Dataset {split} has {dataset.metadata.total_groups} groups.")
    if is_test and label_separate:
        print("Evaluation split uses PuzzleDatasetSeparate with fully noised answer tokens.")
    elif is_test and not config.mask_full_training:
        print(f"Shuffling evaluation problems with seed {config.seed}.")
    elif is_test:
        print(
            "Evaluation split uses shuffled PuzzleFullDataset "
            f"with {config.full_min_pairs}-{config.full_max_pairs} pairs per sample "
            f"and seed {config.seed}."
        )
    elif config.mask_full_training:
        print(
            "Training split uses PuzzleFullDataset "
            f"with {config.full_min_pairs}-{config.full_max_pairs} pairs per sample."
        )
    elif label_separate:
        print("Training split uses PuzzleDatasetSeparate with appended noised answer tokens.")
    elif dataset.metadata.train_target_mode == "random_output_pair":
        min_context_pairs = dataset.metadata.min_context_pairs
        if min_context_pairs is None and config.arc_output_mask is not None:
            min_context_pairs = config.arc_output_mask.min_context_pairs
        print(
            "Training split uses dynamic ARC output masking"
            + (
                f" with min_context_pairs={min_context_pairs}."
                if min_context_pairs is not None
                else "."
            )
        )
    dataloader = DataLoader(
        dataset, batch_size=None, num_workers=1, prefetch_factor=8, pin_memory=True, persistent_workers=True
    )
    print(f"Created dataloader for split '{split}'.")
    return dataloader, dataset.metadata


def _apply_position_id_shape_to_model_cfg(model_cfg: dict, position_id_shape: Optional[Sequence[int]]) -> dict:
    if position_id_shape is None:
        return model_cfg

    if len(position_id_shape) == 4:
        model_cfg["grid_depth"] = model_cfg.get("grid_depth", 0) or position_id_shape[0]
        model_cfg["grid_io"] = model_cfg.get("grid_io", 0) or position_id_shape[1]
        model_cfg["grid_height"] = model_cfg.get("grid_height", 0) or position_id_shape[2]
        model_cfg["grid_width"] = model_cfg.get("grid_width", 0) or position_id_shape[3]
    elif len(position_id_shape) == 3:
        model_cfg["grid_depth"] = model_cfg.get("grid_depth", 0) or position_id_shape[0]
        model_cfg["grid_height"] = model_cfg.get("grid_height", 0) or position_id_shape[1]
        model_cfg["grid_width"] = model_cfg.get("grid_width", 0) or position_id_shape[2]
    elif len(position_id_shape) == 2:
        model_cfg["grid_height"] = model_cfg.get("grid_height", 0) or position_id_shape[0]
        model_cfg["grid_width"] = model_cfg.get("grid_width", 0) or position_id_shape[1]

    return model_cfg


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    effective_local_batch_size = config.global_batch_size * max(1, config.grad_accum_steps) // world_size
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=effective_local_batch_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        variable_seq_lengths=train_metadata.variable_seq_lengths,
        grad_logging_enabled=not config.halted_replay_training,
        causal=False,  # Non-autoregressive
    )
    model_cfg["label_separate"] = _label_separate_enabled(config)
    model_cfg["SeparateMode"] = _separate_mode(config)
    model_cfg["label_separate_C_noise_scale"] = float(_config_or_arch_extra(config, "label_separate_C_noise_scale"))
    model_cfg = _apply_position_id_shape_to_model_cfg(model_cfg, train_metadata.position_id_shape)

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if config.torch_compile:
            if rank == 0:
                print(
                    "torch.compile enabled "
                    f"(dynamic={train_metadata.variable_seq_lengths})"
                )
            model = torch.compile(model, dynamic=train_metadata.variable_seq_lengths)  # type: ignore
        elif rank == 0:
            print("torch.compile disabled")

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    if config.use_muon:
        adam_params = [p for p in model.parameters() if p.ndim != 2]
        muon_params = [p for p in model.parameters() if p.ndim == 2]

        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=1e-12,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            ),
            Muon([
                {
                    "params": muon_params,
                    "use_muon": True,
                    "lr": 1e-4,
                },
                {
                    "params": adam_params,
                    "use_muon": False,
                    "lr": 1e-4,
                    "weight_decay": 0.1,
                    "adamw_betas": (0.9, 0.95),
                    "adamw_eps": 1e-8,
                },
            ]),
        ]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=1e-12,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            ),
            AdamAtan2(
                model.parameters(),
                lr=1e-12,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2),
            ),
        ]

    optimizer_lrs = [config.puzzle_emb_lr, config.lr]

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0,
    num_cycles: float = 0.5,
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (
        min_ratio
        + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    )


def init_train_state(
    config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int
):
    # Estimate total optimizer steps using the same eval-interval chunking as the
    # training loop so dropped partial batches are reflected in the count.
    effective_gbs = config.global_batch_size * max(1, config.grad_accum_steps)
    if config.mask_full_training:
        sampled_examples_per_group = 1.0
    else:
        sampled_examples_per_group = (
            train_metadata.mean_puzzle_examples
            if config.examples_per_puzzle is None
            else float(config.examples_per_puzzle)
        )
    if config.eval_interval is None or config.eval_interval <= 0:
        total_steps = int(
            config.epochs
            * train_metadata.total_groups
            * sampled_examples_per_group
            / effective_gbs
        )
    else:
        total_iters = config.epochs // config.eval_interval
        steps_per_iter = int(
            config.eval_interval
            * train_metadata.total_groups
            * sampled_examples_per_group
            / effective_gbs
        )
        total_steps = total_iters * steps_per_iter

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)

    train_state = TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
    )

    load_initial_state(train_state, config, rank)

    return train_state


def save_train_state(config: PretrainConfig, train_state: TrainState):
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    state = {
        "step": train_state.step,
        "train_time_h": train_state.train_time_h,
        "last_step_time_h": train_state.last_step_time_h,
        "model_state_dict": train_state.model.state_dict(),
        "optimizer_states": [optim.state_dict() for optim in train_state.optimizers],
    }

    state["rng_state"] = torch.random.get_rng_state()
    if torch.cuda.is_available():
        try:
            state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        except RuntimeError:
            state["cuda_rng_state"] = torch.cuda.get_rng_state()

    torch.save(state, os.path.join(config.checkpoint_path, f"step_{train_state.step}.pt"))


def _resolve_checkpoint_path(path: str) -> Optional[str]:
    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        pattern = re.compile(r"step_(\d+)(?:\.pt)?$")
        candidates: List[Tuple[int, str]] = []
        for file_name in os.listdir(path):
            match = pattern.match(file_name)
            if match:
                candidates.append((int(match.group(1)), os.path.join(path, file_name)))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[-1][1]

    return None


def load_config_from_checkpoint_path(path: str) -> Optional[PretrainConfig]:
    """Load a saved config from a checkpoint directory, if present."""

    resolved_path = _resolve_checkpoint_path(path)
    checkpoint_dir = Path(resolved_path if resolved_path is not None else path)
    if checkpoint_dir.is_file():
        checkpoint_dir = checkpoint_dir.parent

    def _load_candidate(candidate: Path) -> Optional[PretrainConfig]:
        if not candidate.exists():
            return None

        if candidate.suffix.lower() == ".json":
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)
                if isinstance(config_dict, dict):
                    return PretrainConfig(**config_dict)
            except Exception:
                pass
            return None

        # Prefer OmegaConf so we can parse Hydra-specific tags written during training.
        try:
            conf = OmegaConf.load(candidate)
            # Convert to a plain container so pydantic can consume it.
            as_dict = OmegaConf.to_container(conf, resolve=True)
            if isinstance(as_dict, dict):
                return PretrainConfig(**as_dict)
        except Exception:
            pass

        # Fallback to a plain YAML load if OmegaConf parsing fails for any reason.
        try:
            with open(candidate, "r") as f:
                config_dict = yaml.safe_load(f)
            if isinstance(config_dict, dict):
                return PretrainConfig(**config_dict)
        except Exception:
            pass

        return None

    for candidate in [
        checkpoint_dir / "config.yaml",
        checkpoint_dir / "config.json",
        checkpoint_dir / "all_config.yaml",
        checkpoint_dir / ".hydra" / "config.yaml",
    ]:
        loaded = _load_candidate(candidate)
        if loaded is not None:
            return loaded

    return None


def _resize_puzzle_embedding_if_needed(model: nn.Module, state_dict: dict):
    puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
    puzzle_emb = _get_puzzle_embedding_module(model)
    if puzzle_emb is None:
        return
    expected_shape: torch.Size = puzzle_emb.weights.shape  # type: ignore
    if puzzle_emb_name in state_dict:
        puzzle_emb = state_dict[puzzle_emb_name]
        if puzzle_emb.shape != expected_shape:
            print(
                f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}"
            )

            # Re-initialize using mean
            state_dict[puzzle_emb_name] = (
                torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
            )


def _prepare_rng_state(state: Any, device: str | None) -> Any:
    """Ensure RNG state tensors are on the correct device and uint8 dtype."""

    if state is None:
        return None

    if isinstance(state, (list, tuple)):
        return [_prepare_rng_state(s, device) for s in state]

    tensor_state = torch.as_tensor(state, device=device)
    if tensor_state.dtype != torch.uint8:
        tensor_state = tensor_state.to(torch.uint8)

    return tensor_state


def _resolve_requested_checkpoint_path(load_path: str, checkpoint_path: Optional[str]) -> str:
    if load_path == "latest":
        if checkpoint_path is None:
            raise ValueError("Cannot load latest checkpoint without a checkpoint_path configured.")
        load_path = checkpoint_path

    resolved_path = _resolve_checkpoint_path(load_path)
    if resolved_path is None:
        raise FileNotFoundError(f"Could not resolve checkpoint path from '{load_path}'")

    return resolved_path


def _load_checkpoint_payload(load_path: str, checkpoint_path: Optional[str], rank: int):
    resolved_path = _resolve_requested_checkpoint_path(load_path, checkpoint_path)
    if rank == 0:
        print(f"Loading checkpoint {resolved_path}")

    checkpoint = torch.load(resolved_path, map_location="cuda")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        optimizer_states = checkpoint.get("optimizer_states")
        step = checkpoint.get("step")
        train_time_h = checkpoint.get("train_time_h")
        last_step_time_h = checkpoint.get("last_step_time_h")
        if train_time_h is None and checkpoint.get("train_time_s") is not None:
            train_time_h = float(checkpoint["train_time_s"]) / 3600.0
        if last_step_time_h is None and checkpoint.get("last_step_time_s") is not None:
            last_step_time_h = float(checkpoint["last_step_time_s"]) / 3600.0
        rng_state = checkpoint.get("rng_state")
        cuda_rng_state = checkpoint.get("cuda_rng_state")
    else:
        # Backwards compatibility with checkpoints that only contain model weights
        state_dict = checkpoint
        optimizer_states = None
        step = None
        train_time_h = None
        last_step_time_h = None
        rng_state = None
        cuda_rng_state = None

    return state_dict, optimizer_states, step, train_time_h, last_step_time_h, rng_state, cuda_rng_state


def _load_model_state(train_state: TrainState, config: PretrainConfig, state_dict: dict, rank: int):
    _resize_puzzle_embedding_if_needed(train_state.model, state_dict)
    try:
        # Keep parameter objects stable so pre-created optimizers still point at
        # the live model parameters after a training resume.
        load_result = train_state.model.load_state_dict(state_dict, strict=config.load_strict)
    except RuntimeError:
        # Re-raise with clearer guidance if strict loading was requested.
        raise

    if not config.load_strict and rank == 0:
        missing, unexpected = load_result
        if missing:
            print(f"Warning: missing keys during checkpoint load: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys during checkpoint load: {unexpected}")


def load_checkpoint(train_state: TrainState, config: PretrainConfig, rank: int):
    load_path = config.load_checkpoint
    if load_path is None:
        return

    state_dict, optimizer_states, step, train_time_h, last_step_time_h, rng_state, cuda_rng_state = _load_checkpoint_payload(
        load_path, checkpoint_path=config.checkpoint_path, rank=rank
    )
    _load_model_state(train_state, config, state_dict, rank)

    if optimizer_states is not None:
        if not config.load_optimizer_state:
            if rank == 0:
                print("Skipping optimizer state load because load_optimizer_state=False")
        elif len(optimizer_states) != len(train_state.optimizers):
            raise ValueError(
                "Checkpoint optimizer count does not match current configuration: "
                f"{len(optimizer_states)} vs {len(train_state.optimizers)}"
            )
        else:
            for optimizer, optimizer_state in zip(train_state.optimizers, optimizer_states):
                optimizer.load_state_dict(optimizer_state)

    if step is not None:
        train_state.step = int(step)
    if train_time_h is not None:
        train_state.train_time_h = float(train_time_h)
    if last_step_time_h is not None:
        train_state.last_step_time_h = float(last_step_time_h)

    # Reset carry since we do not serialize it
    train_state.carry = None
    train_state.accum_carries = None
    train_state.accum_metrics = None

    if rng_state is not None:
        normalized_rng_state = _prepare_rng_state(rng_state, device="cpu")
        # Older checkpoints should always store a single tensor here.
        if isinstance(normalized_rng_state, list):
            normalized_rng_state = normalized_rng_state[0]
        torch.random.set_rng_state(normalized_rng_state)

    if cuda_rng_state is not None and torch.cuda.is_available():
        normalized_cuda_state = _prepare_rng_state(cuda_rng_state, device="cpu")
        try:
            if isinstance(normalized_cuda_state, list):
                if len(normalized_cuda_state) != torch.cuda.device_count():
                    primary_state = normalized_cuda_state[0]
                    normalized_cuda_state = [
                        primary_state for _ in range(torch.cuda.device_count())
                    ]
                torch.cuda.set_rng_state_all(normalized_cuda_state)
            else:
                torch.cuda.set_rng_state(normalized_cuda_state)
        except RuntimeError:
            fallback_state = (
                normalized_cuda_state[0]
                if isinstance(normalized_cuda_state, list)
                else normalized_cuda_state
            )
            torch.cuda.set_rng_state(fallback_state)


def load_checkpoint_file(train_state: TrainState, config: PretrainConfig, rank: int):
    load_path = config.load_checkpoint_file
    if load_path is None:
        return

    state_dict, _, _, _, _, _, _ = _load_checkpoint_payload(
        load_path, checkpoint_path=config.checkpoint_path, rank=rank
    )
    _load_model_state(train_state, config, state_dict, rank)

    # Weight-only initialization intentionally keeps optimizer, step, and RNG state fresh.
    train_state.carry = None
    train_state.accum_carries = None
    train_state.accum_metrics = None
    if rank == 0:
        print("Loaded model weights only; optimizer state, step, and RNG state were not restored.")


def load_initial_state(train_state: TrainState, config: PretrainConfig, rank: int):
    if config.load_checkpoint is not None and config.load_checkpoint_file is not None:
        raise ValueError("load_checkpoint and load_checkpoint_file are mutually exclusive.")

    if config.load_checkpoint is not None:
        load_checkpoint(train_state, config, rank)
    elif config.load_checkpoint_file is not None:
        load_checkpoint_file(train_state, config, rank)


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio,
    )


def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    # Initialize evaluators
    evaluators = []
    print("creating evaluators...")
    print(config.evaluators)
    for cfg in config.evaluators:
        print(f"Creating evaluator: {cfg.name}")
        cls = load_model_class(cfg.name, "evaluators.")(
            data_path=config.data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
        )  # type: ignore
        evaluators.append(cls)

    return evaluators


def _get_puzzle_embedding_module(module: nn.Module):
    for candidate in (module, getattr(module, "_orig_mod", None)):
        if candidate is None:
            continue
        wrapped_model = getattr(candidate, "model", None)
        if wrapped_model is not None and hasattr(wrapped_model, "puzzle_emb"):
            return wrapped_model.puzzle_emb
        if hasattr(candidate, "puzzle_emb"):
            return candidate.puzzle_emb
    return None


def _get_metric_value(metrics: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return float(value)

        if "/" in key:
            nested = metrics
            for part in key.split("/"):
                if not isinstance(nested, dict) or part not in nested:
                    nested = None
                    break
                nested = nested[part]
            if nested is not None:
                return float(nested)

    return None


def _add_time_axis_metrics(
    train_state: TrainState,
    metrics: Dict[str, Any],
    metric_specs: Sequence[Tuple[str, Sequence[str]]],
) -> None:
    metrics["time/total_train_time_h"] = train_state.train_time_h

    for output_key, source_keys in metric_specs:
        value = _get_metric_value(metrics, source_keys)
        if value is not None:
            metrics[f"time/{output_key}"] = value


def _add_train_timing_metrics(train_state: TrainState, metrics: Dict[str, Any], prefix: str = "train") -> None:
    metrics[f"{prefix}/total_train_time_h"] = train_state.train_time_h
    metrics[f"{prefix}/avg_step_time_h"] = train_state.train_time_h / max(train_state.step, 1)
    metrics[f"{prefix}/total_train_time_s"] = train_state.train_time_h * 3600.0
    metrics[f"{prefix}/avg_step_time_s"] = train_state.train_time_h * 3600.0 / max(train_state.step, 1)
    if train_state.last_step_time_h is not None:
        metrics[f"{prefix}/step_time_h"] = train_state.last_step_time_h
        metrics[f"{prefix}/step_time_s"] = train_state.last_step_time_h * 3600.0

    _add_time_axis_metrics(
        train_state,
        metrics,
        (
            ("train_lm_loss", ("train/lm_loss",)),
            ("train_accuracy", ("train/accuracy",)),
            ("train_exact_accuracy", ("train/exact_accuracy",)),
        ),
    )


def _add_eval_timing_metrics(
    train_state: TrainState,
    metrics: Dict[str, Any],
    eval_time_h: Optional[float] = None,
) -> None:
    metrics["train/total_train_time_h"] = train_state.train_time_h
    metrics["train/avg_step_time_h"] = train_state.train_time_h / max(train_state.step, 1)
    metrics["train/total_train_time_s"] = train_state.train_time_h * 3600.0
    metrics["train/avg_step_time_s"] = train_state.train_time_h * 3600.0 / max(train_state.step, 1)
    if train_state.last_step_time_h is not None:
        metrics["train/step_time_h"] = train_state.last_step_time_h
        metrics["train/step_time_s"] = train_state.last_step_time_h * 3600.0
    if eval_time_h is not None:
        metrics["all/eval_time_h"] = eval_time_h
        metrics["all/eval_time_s"] = eval_time_h * 3600.0

    _add_time_axis_metrics(
        train_state,
        metrics,
        (
            (
                "all_accuracy",
                ("all_accuracy", "all/accuracy", "eval/all_accuracy", "eval/all/accuracy"),
            ),
            (
                "all_lm_loss",
                ("all_lm_loss", "all/lm_loss", "eval/all_lm_loss", "eval/all/lm_loss"),
            ),
            (
                "all_exact_accuracy",
                ("all_exact_accuracy", "all/exact_accuracy", "eval/all_exact_accuracy", "eval/all/exact_accuracy"),
            ),
        ),
    )


def _batch_num_examples(batch: Dict[str, torch.Tensor]) -> int:
    if "puzzle_identifiers" in batch:
        return int(batch["puzzle_identifiers"].shape[0])
    return int(batch["inputs"].shape[0])


def _offsets_from_lengths(lengths: torch.Tensor, dtype: torch.dtype = torch.int32) -> torch.Tensor:
    return F.pad(torch.cumsum(lengths.to(dtype), dim=0), (1, 0))


def _cat_indexed_ranges(
    values: torch.Tensor,
    offsets: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    chunks = []
    offsets_cpu = offsets.detach().cpu()
    for index in indices.detach().cpu().tolist():
        start = int(offsets_cpu[index].item())
        end = int(offsets_cpu[index + 1].item())
        chunks.append(values[start:end])

    if chunks:
        return torch.cat(chunks, dim=0)
    return values.new_empty((0,) + tuple(values.shape[1:]))


def _select_batch_examples(
    batch: Dict[str, torch.Tensor],
    sample_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    indices = torch.nonzero(sample_mask, as_tuple=False).flatten().to(device=sample_mask.device)
    batch_size = _batch_num_examples(batch)

    if "seq_offsets" not in batch:
        selected: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if value.ndim > 0 and value.shape[0] == batch_size:
                selected[key] = value[indices]
            else:
                selected[key] = value
        return selected

    seq_offsets = batch["seq_offsets"].to(device=batch["inputs"].device, dtype=torch.long)
    seq_lengths = batch.get("seq_lengths")
    if seq_lengths is None:
        seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    selected_seq_lengths = seq_lengths[indices].to(torch.int32)

    label_offsets = batch.get("label_seq_offsets", batch["seq_offsets"]).to(
        device=batch["inputs"].device,
        dtype=torch.long,
    )
    label_lengths = batch.get("label_seq_lengths")
    if label_lengths is None:
        label_lengths = label_offsets[1:] - label_offsets[:-1]
    selected_label_lengths = label_lengths[indices].to(torch.int32)

    token_total = int(seq_offsets[-1].item()) if seq_offsets.numel() else 0
    label_total = int(label_offsets[-1].item()) if label_offsets.numel() else 0

    selected = {}
    for key, value in batch.items():
        if key in {"seq_offsets", "label_seq_offsets"}:
            continue
        if key == "seq_lengths":
            selected[key] = selected_seq_lengths.to(dtype=value.dtype)
        elif key == "label_seq_lengths":
            selected[key] = selected_label_lengths.to(dtype=value.dtype)
        elif key == "labels" and value.ndim > 0 and value.shape[0] == label_total:
            selected[key] = _cat_indexed_ranges(value, label_offsets, indices)
        elif value.ndim > 0 and value.shape[0] == token_total:
            selected[key] = _cat_indexed_ranges(value, seq_offsets, indices)
        elif value.ndim > 0 and value.shape[0] == batch_size:
            selected[key] = value[indices]
        else:
            selected[key] = value

    selected["seq_lengths"] = selected_seq_lengths
    selected["seq_offsets"] = _offsets_from_lengths(selected_seq_lengths, dtype=batch["seq_offsets"].dtype).to(
        device=batch["seq_offsets"].device
    )
    if "label_seq_offsets" in batch or "label_seq_lengths" in batch:
        selected["label_seq_lengths"] = selected_label_lengths
        selected["label_seq_offsets"] = _offsets_from_lengths(
            selected_label_lengths,
            dtype=batch.get("label_seq_offsets", batch["seq_offsets"]).dtype,
        ).to(device=batch.get("label_seq_offsets", batch["seq_offsets"]).device)

    return selected


def _select_packed_hidden(
    hidden: torch.Tensor,
    current_data: Optional[Dict[str, torch.Tensor]],
    sample_mask: torch.Tensor,
) -> torch.Tensor:
    if hidden.shape[0] == 0 or current_data is None or "seq_lengths" not in current_data:
        return hidden.new_empty((0,) + tuple(hidden.shape[1:]))

    indices = torch.nonzero(sample_mask, as_tuple=False).flatten().to(device=sample_mask.device)
    lengths = current_data["seq_lengths"].to(device=hidden.device, dtype=torch.long)
    batch_size = int(lengths.shape[0])
    if batch_size == 0:
        return hidden.new_empty((0,) + tuple(hidden.shape[1:]))

    prefix_total = int(hidden.shape[0] - int(lengths.sum().item()))
    prefix_len = max(prefix_total // batch_size, 0)
    hidden_lengths = lengths + prefix_len
    hidden_offsets = _offsets_from_lengths(hidden_lengths, dtype=torch.long).to(device=hidden.device)
    return _cat_indexed_ranges(hidden, hidden_offsets, indices)


def _select_carry_examples(carry: Any, sample_mask: torch.Tensor) -> Any:
    indices = torch.nonzero(sample_mask, as_tuple=False).flatten().to(device=sample_mask.device)
    current_data = getattr(carry, "current_data", None)
    selected_current_data = (
        _select_batch_examples(current_data, sample_mask)
        if isinstance(current_data, dict)
        else current_data
    )

    current_hidden = getattr(carry, "current_hidden", None)
    if current_hidden is not None:
        if isinstance(current_data, dict) and "seq_offsets" in current_data:
            selected_hidden = _select_packed_hidden(current_hidden, current_data, sample_mask)
        elif current_hidden.ndim > 0 and current_hidden.shape[0] == sample_mask.shape[0]:
            selected_hidden = current_hidden[indices]
        else:
            selected_hidden = current_hidden
    else:
        selected_hidden = None

    updates = {"current_data": selected_current_data}
    if selected_hidden is not None:
        updates["current_hidden"] = selected_hidden

    steps = getattr(carry, "steps", None)
    if steps is not None:
        updates["steps"] = steps[indices]

    halted = getattr(carry, "halted", None)
    if halted is not None:
        updates["halted"] = halted[indices]

    return replace(carry, **updates)


def _merge_replay_loss_metrics(
    probe_metrics: Dict[str, torch.Tensor],
    replay_metrics: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    metrics = dict(probe_metrics)
    loss_keys = [key for key in metrics if key.endswith("loss")]
    if replay_metrics is None:
        for key in loss_keys:
            metrics[key] = torch.zeros_like(metrics[key])
        return metrics

    for key, value in replay_metrics.items():
        if key.endswith("loss"):
            metrics[key] = value.detach()
    for key in loss_keys:
        if key not in replay_metrics:
            metrics[key] = torch.zeros_like(metrics[key])
    return metrics


def _sanitize_sparse_embedding_replay_range(
    puzzle_emb: Any,
    start: int,
    reserved_size: int,
    active_size: int,
) -> None:
    if puzzle_emb is None or not hasattr(puzzle_emb, "local_ids"):
        return
    if active_size >= reserved_size:
        return

    local_ids = puzzle_emb.local_ids
    end = start + reserved_size
    active_end = start + active_size
    if active_size > 0:
        fill_value = local_ids[start].detach().clone()
    else:
        fill_value = torch.zeros((), dtype=local_ids.dtype, device=local_ids.device)
    local_ids[active_end:end] = fill_value


def _train_batch_halted_replay(
    config: PretrainConfig,
    train_state: TrainState,
    batch: Dict[str, torch.Tensor],
    carry: Any,
    global_batch_size: int,
    accum_index: int,
    local_batch_size: int,
    puzzle_emb: Any,
):
    compute_target_q = train_state.step % config.target_q_update_every == 0

    with torch.no_grad():
        probe_carry, _probe_loss, probe_metrics, _, _ = train_state.model(
            carry=carry,
            batch=batch,
            return_keys=[],
            compute_target_q=compute_target_q,
        )

    halted = probe_carry.halted
    if halted is None:
        raise RuntimeError("halted_replay_training requires model carries to expose a halted tensor.")

    halted_count = int(halted.sum().item())
    replay_metrics = None
    if halted_count > 0:
        replay_batch = _select_batch_examples(batch, halted)
        replay_carry = _select_carry_examples(carry, halted)
        _replay_carry, replay_loss, replay_metrics, _, _ = train_state.model(
            carry=replay_carry,
            batch=replay_batch,
            return_keys=[],
            compute_target_q=compute_target_q,
        )

        loss_scale = 1.0 / (global_batch_size * max(1, config.grad_accum_steps))
        (loss_scale * replay_loss).backward()

    _sanitize_sparse_embedding_replay_range(
        puzzle_emb,
        start=accum_index * local_batch_size,
        reserved_size=local_batch_size,
        active_size=halted_count,
    )
    return probe_carry, _merge_replay_loss_metrics(probe_metrics, replay_metrics)


def train_batch(
    config: PretrainConfig,
    train_state: TrainState,
    batch: Any,
    global_batch_size: int,
    rank: int,
    world_size: int,
):
    accum_steps = max(1, getattr(config, "grad_accum_steps", 1))
    if train_state.step >= train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    accum_index = train_state.accum_step % accum_steps
    local_batch_size = int(batch["puzzle_identifiers"].shape[0] if "puzzle_identifiers" in batch else batch["inputs"].shape[0])
    puzzle_emb = _get_puzzle_embedding_module(train_state.model)
    if puzzle_emb is not None and hasattr(puzzle_emb, "set_local_range"):
        puzzle_emb.set_local_range(accum_index * local_batch_size, local_batch_size)

    if accum_steps == 1:
        # Preserve the original single-carry behavior for non-accumulated training.
        if train_state.carry is None:
            with torch.device("cuda"):
                train_state.carry = train_state.model.initial_carry(batch)  # type: ignore
        carry = train_state.carry
    else:
        # Keep one independent recurrent carry per accumulation shard. This makes
        # N microbatches behave like N slices of a larger concurrent batch instead
        # of repeatedly advancing the same smaller set of recurrent slots.
        if train_state.accum_carries is None or len(train_state.accum_carries) != accum_steps:
            train_state.accum_carries = [None] * accum_steps
            train_state.carry = train_state.accum_carries
            train_state.accum_metrics = None

        if train_state.accum_carries[accum_index] is None:
            with torch.device("cuda"):
                train_state.accum_carries[accum_index] = train_state.model.initial_carry(batch)  # type: ignore
        carry = train_state.accum_carries[accum_index]

    # Forward
    if config.halted_replay_training:
        carry, metrics = _train_batch_halted_replay(
            config=config,
            train_state=train_state,
            batch=batch,
            carry=carry,
            global_batch_size=global_batch_size,
            accum_index=accum_index,
            local_batch_size=local_batch_size,
            puzzle_emb=puzzle_emb,
        )
    else:
        compute_target_q = train_state.step % config.target_q_update_every == 0
        carry, loss, metrics, _, _ = train_state.model(
            carry=carry, batch=batch, return_keys=[], compute_target_q=compute_target_q
        )
        loss_scale = 1.0 / (global_batch_size * accum_steps)
        (loss_scale * loss).backward()

    if accum_steps == 1:
        train_state.carry = carry
    else:
        train_state.accum_carries[accum_index] = carry

    halted = getattr(carry, "halted", None)
    if halted is not None:
        metrics["halted_ratio"] = halted.to(torch.float32).sum()

    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())
        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k].detach() for k in metric_keys])
        if train_state.accum_metrics is None:
            train_state.accum_metrics = {k: v.clone() for k, v in zip(metric_keys, metric_values)}
        else:
            for k, v in zip(metric_keys, metric_values):
                if k in train_state.accum_metrics:
                    train_state.accum_metrics[k] = train_state.accum_metrics[k] + v
                else:
                    train_state.accum_metrics[k] = v.clone()

    train_state.accum_step += 1

    should_step = train_state.accum_step % accum_steps == 0
    if not should_step:
        return

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if not param.requires_grad:
                continue

            grad = param.grad
            created_zero_grad = grad is None
            if grad is None:
                param.grad = torch.zeros_like(param)
                grad = param.grad
            dist.all_reduce(grad)
            if created_zero_grad and not torch.any(grad != 0):
                param.grad = None

    # Apply optimizer
    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group["lr"] = lr_this_step

        optim.step()
        optim.zero_grad()

    train_state.step += 1
    train_state.accum_step = 0

    # Reduce metrics
    if train_state.accum_metrics is not None and len(train_state.accum_metrics):
        metric_keys = list(sorted(train_state.accum_metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([train_state.accum_metrics[k] for k in metric_keys])
        train_state.accum_metrics = None
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

            # Postprocess
            count = max(reduced_metrics.get("count", 0), 1)  # Avoid NaNs

            def _normalize_metric(key: str, value: float) -> float:
                if key.startswith("profile/"):
                    return value / (world_size * accum_steps)
                if key.endswith("loss"):
                    return value / (global_batch_size * accum_steps)
                if key == "halted_ratio":
                    return value / (global_batch_size * accum_steps)
                return value / count

            reduced_metrics = {f"train/{k}": _normalize_metric(k, v) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            
            if global_logger.is_log and train_state.step % 2000 == 0:
                logger_dict = global_logger.get_log_dict(train_state.step)
                logger_dict = {f"train/{k}": v for k, v in logger_dict.items()}
                reduced_metrics.update(logger_dict)
            
            return reduced_metrics


def benchmark_training_steps(
    config: PretrainConfig,
    train_state: TrainState,
    train_loader: torch.utils.data.DataLoader,
    rank: int,
    world_size: int,
):
    warmup_steps = max(0, config.benchmark_warmup_steps)
    measure_steps = max(1, config.benchmark_steps)
    target_completed = warmup_steps + measure_steps
    step_times_ms: List[float] = []
    step_wall_times_s: List[float] = []
    completed_optimizer_steps = 0
    active_start_event: Optional[torch.cuda.Event] = None
    active_start_wall: Optional[float] = None

    if rank == 0:
        print(
            "Starting benchmark: "
            f"warmup_steps={warmup_steps}, measure_steps={measure_steps}, "
            f"grad_accum_steps={config.grad_accum_steps}"
        )

    train_state.model.train()
    torch.cuda.synchronize()

    while completed_optimizer_steps < target_completed:
        progressed = False
        for _set_name, batch, global_batch_size in train_loader:
            progressed = True
            if train_state.accum_step == 0:
                active_start_event = torch.cuda.Event(enable_timing=True)
                active_start_event.record()
                active_start_wall = time.perf_counter()

            before_step = train_state.step
            train_batch(
                config,
                train_state,
                batch,
                global_batch_size,
                rank=rank,
                world_size=world_size,
            )

            if train_state.step > before_step:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                torch.cuda.synchronize()

                completed_optimizer_steps += 1
                if completed_optimizer_steps > warmup_steps and active_start_event is not None:
                    step_times_ms.append(active_start_event.elapsed_time(end_event))
                    if active_start_wall is not None:
                        step_wall_times_s.append(time.perf_counter() - active_start_wall)

                    if rank == 0:
                        print(
                            f"Benchmark step {len(step_times_ms)}/{measure_steps}: "
                            f"{step_times_ms[-1] / 1000.0:.4f}s"
                        )

                active_start_event = None
                active_start_wall = None

                if completed_optimizer_steps >= target_completed:
                    break

        if not progressed:
            raise RuntimeError("Benchmark dataloader produced no batches.")

    if rank != 0:
        return

    mean_ms = statistics.fmean(step_times_ms)
    median_ms = statistics.median(step_times_ms)
    result = {
        "optimizer_steps": len(step_times_ms),
        "warmup_optimizer_steps": warmup_steps,
        "grad_accum_steps": config.grad_accum_steps,
        "mean_step_s": mean_ms / 1000.0,
        "median_step_s": median_ms / 1000.0,
        "mean_microbatch_s": mean_ms / 1000.0 / max(1, config.grad_accum_steps),
        "median_microbatch_s": median_ms / 1000.0 / max(1, config.grad_accum_steps),
        "mean_wall_step_s": statistics.fmean(step_wall_times_s) if step_wall_times_s else None,
    }
    print("BENCHMARK_RESULTS " + json.dumps(result, sort_keys=True))


def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
    early_eval: bool = False,
):
    reduced_metrics = None
    original_is_log = global_logger.is_log
    global_logger.is_log = False
    try:
        with torch.inference_mode():
            return_keys = set(config.eval_save_outputs)
            for evaluator in evaluators:
                evaluator.begin_eval()
                return_keys.update(evaluator.required_outputs)

            # Run evaluation
            set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

            save_preds = {}

            metric_keys = []
            metric_values = None

            carry = None
            processed_batches = 0
            
            print("Starting evaluation... len(eval_loader) =", len(eval_loader))
            for set_name, batch, global_batch_size in eval_loader:
                if early_eval and processed_batches > 50:
                    break

                processed_batches += 1
                if rank == 0:
                    print(f"Processing batch {processed_batches}: {set_name}")
                
                # To device
                batch = {k: v.cuda() for k, v in batch.items()}
                with torch.device("cuda"):
                    carry = train_state.model.initial_carry(batch)  # type: ignore

                # Forward
                inference_steps = 0
                while True:
                    carry, loss, metrics, preds, all_finish = train_state.model(
                        carry=carry, batch=batch, return_keys=return_keys
                    )
                    inference_steps += 1

                    if all_finish:
                        break

                if rank == 0:
                    print(f"  Completed inference in {inference_steps} steps")

                for collection in (batch, preds):
                    for k, v in collection.items():
                        if k in config.eval_save_outputs:
                            save_preds.setdefault(k, [])
                            save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

                for evaluator in evaluators:
                    evaluator.update_batch(batch, preds)

                del carry, loss, preds, batch, all_finish

                # Aggregate metrics
                set_id = set_ids[set_name]

                if metric_values is None:
                    metric_keys = list(
                        sorted(metrics.keys())
                    )  # Sort keys to guarantee all processes use the same order.
                    metric_values = torch.zeros(
                        (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                    )

                metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

                del metrics

            # concatenate save preds
            save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

            # Save preds
            if config.checkpoint_path is not None and len(save_preds):
                # Each rank save predictions independently
                os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
                torch.save(
                    save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
                )

            del save_preds

            # Reduce to rank 0
            if metric_values is not None:
                if world_size > 1:
                    dist.reduce(metric_values, dst=0)

                if rank == 0:
                    reduced_metrics = metric_values.cpu().numpy()
                    reduced_metrics = {
                        set_name: {
                            metric_name: reduced_metrics[set_id, metric_id]
                            for metric_id, metric_name in enumerate(metric_keys)
                        }
                        for set_id, set_name in enumerate(set_ids)
                    }

                    # Postprocess
                    for set_name, m in reduced_metrics.items():
                        count = m.pop("count")
                        reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

            # Run evaluators
            if rank == 0:
                print(f"\nRunning {len(evaluators)} evaluator(s)...")
                
            for i, evaluator in enumerate(evaluators):
                if rank == 0:
                    print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                    
                # Path for saving
                evaluator_save_path = None
                if config.checkpoint_path is not None:
                    evaluator_save_path = os.path.join(
                        config.checkpoint_path,
                        f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                    )
                    os.makedirs(evaluator_save_path, exist_ok=True)

                # Run and log
                metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
                if rank == 0 and metrics is not None:
                    if reduced_metrics is None:
                        reduced_metrics = {}

                    reduced_metrics.update(metrics)
                    print(f"  Completed {evaluator.__class__.__name__}")
                    
            if rank == 0:
                print("All evaluators completed!")
    finally:
        global_logger.is_log = original_is_log

    return reduced_metrics


def save_code_and_config(config: PretrainConfig, save_dir: str):
    import os, json
    import yaml

    os.makedirs(save_dir, exist_ok=True)

    cfg_path = os.path.join(save_dir, "config.yaml")
    json_path = os.path.join(save_dir, "config.json")

    config_dict = json.loads(config.model_dump_json())

    try:
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, sort_keys=False, allow_unicode=True)

    except Exception as e:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)

        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(
                "# Failed to write config as YAML, wrote config.json instead.\n"
                f"# Error: {type(e).__name__}: {e}\n"
            )


def _get_loop_config(model: nn.Module):
    candidates = [model, getattr(model, "_orig_mod", None)]
    seen = set()

    while candidates:
        candidate = candidates.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))

        model_config = getattr(candidate, "config", None)
        if model_config is not None and hasattr(model_config, "loops"):
            return model_config

        candidates.append(getattr(candidate, "model", None))
        candidates.append(getattr(candidate, "inner", None))

    return None


def _prefix_metrics(metrics: Any, prefix: str):
    if metrics is None:
        return {}

    prefixed = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                prefixed[f"{prefix}/{key}/{sub_key}"] = sub_value
        else:
            prefixed[f"{prefix}/{key}"] = value

    return prefixed


def _build_pretrain_config(hydra_config: DictConfig, rank: int) -> PretrainConfig:
    config_dict = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(config_dict, dict):
        raise ValueError("Expected Hydra config to resolve to a dictionary.")

    resume_from_checkpoint_dir = config_dict.pop("resume_from_checkpoint_dir", None)
    if resume_from_checkpoint_dir is not None:
        if rank == 0:
            print(f"Loading config from checkpoint directory: {resume_from_checkpoint_dir}")

        resume_config = load_config_from_checkpoint_path(str(resume_from_checkpoint_dir))
        if resume_config is None:
            raise FileNotFoundError(
                f"Could not load a saved config from checkpoint directory '{resume_from_checkpoint_dir}'"
            )

        merged_config = resume_config.model_dump()
        # Do not carry over a past checkpoint-loading request when resuming from saved config.
        merged_config["load_checkpoint"] = None
        merged_config["load_checkpoint_file"] = None
        # Only allow runtime-oriented overrides here. The saved training config remains the source
        # of truth for architecture and dataset settings so resume stays faithful to the checkpoint.
        for key in (
            "project_name",
            "run_name",
            "checkpoint_path",
            "load_checkpoint",
            "load_checkpoint_file",
            "load_strict",
            "load_optimizer_state",
            "seed",
        ):
            if key in config_dict:
                merged_config[key] = config_dict[key]

        config_dict = merged_config

    config = PretrainConfig(**config_dict)
    if config.project_name is None:
        config.project_name = "arcagi"

    return config


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = _build_pretrain_config(hydra_config, rank=rank)
        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


def _rewrite_cli_checkpoint_file_flag(argv: List[str]) -> List[str]:
    """Translate --load_checkpoint_file into a Hydra config override."""

    rewritten: List[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--load_checkpoint_file", "--load-checkpoint-file"):
            if i + 1 >= len(argv):
                raise ValueError(f"Expected a path after {arg}")
            rewritten.append(f"+load_checkpoint_file={argv[i + 1]}")
            i += 2
            continue

        matched_flag = None
        for flag in ("--load_checkpoint_file=", "--load-checkpoint-file="):
            if arg.startswith(flag):
                matched_flag = flag
                value = arg[len(flag):]
                if value == "":
                    raise ValueError(f"Expected a path after {flag[:-1]}")
                rewritten.append(f"+load_checkpoint_file={value}")
                break

        if matched_flag is None:
            rewritten.append(arg)

        i += 1

    return rewritten


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    if RANK == 0:
        print("Config:")
        print(config.model_dump_json(indent=2))
    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    # Train loader
    train_epochs_per_iter = config.eval_interval
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    
    # Eval loader
    if config.benchmark_steps > 0:
        eval_loader = eval_metadata = None
        evaluators = []
    else:
        try:
            eval_loader, eval_metadata = create_dataloader(
                config,
                "test",
                test_set_mode=True,
                epochs_per_iter=1,
                global_batch_size=config.global_batch_size,
                rank=RANK,
                world_size=WORLD_SIZE,
            )
            print("len(eval_loader) =", len(eval_loader))
            print("eval_problem_counts =", len(eval_loader) * config.global_batch_size)
            print("eval_metadata =", eval_metadata)
            # Evaluators
            evaluators = create_evaluators(config, eval_metadata)
        except FileNotFoundError as e:
            print(f"eval metadata FileNotFoundError")
            print(e)
            eval_loader = eval_metadata = None
            evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    if config.benchmark_steps > 0:
        benchmark_training_steps(config, train_state, train_loader, rank=RANK, world_size=WORLD_SIZE)
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    ema_helper = None
    if config.ema:
        if RANK == 0:
            print("Setup EMA")
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        if train_state.step > 0:
            progress_bar.update(train_state.step)

        wandb_mode = os.getenv("WANDB_MODE")
        if wandb_mode:
            print(f"W&B mode: {wandb_mode}")
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump(),
            mode=wandb_mode,
            settings=wandb.Settings(_disable_stats=True),
        )
        wandb.define_metric("time/total_train_time_h")
        wandb.define_metric("time/*", step_metric="time/total_train_time_h")
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config, config.checkpoint_path)
        
    print(train_state.model)
    # print parameter count
    total_params = sum(p.numel() for p in train_state.model.parameters())
    print(f"Total parameters: {total_params}")
    for name, param in train_state.model.named_parameters():
        print(f"Parameter: {name}, Shape: {param.shape}, Size: {param.numel()}")
    print(eval_loader, eval_metadata, evaluators)

    # Training Loop
    for _iter_id in range(total_iters):
        if RANK == 0:
            count = 0
            # for set_name, batch, global_batch_size in train_loader:
            #     count += 1
            print(f"_iter_id: {_iter_id}")
            print(f"train_epochs_per_iter: {train_epochs_per_iter}")
            print(f"total_iters: {total_iters}")
            #print(f"train_loader len: {count}")
            print(f"Epoch {_iter_id * train_epochs_per_iter}")

        ############ Initial Evaluation
        if (
            _iter_id == 0
            and config.eval_first
            and train_state.step == 0
            and eval_loader is not None
            and eval_metadata is not None
        ):
            if RANK == 0:
                print("Running initial evaluation at step 0")

            if config.ema and ema_helper is not None:
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state

            train_state_eval.model.eval()
            loop_config = _get_loop_config(train_state_eval.model)
            if loop_config is not None:
                original_loops = loop_config.loops
                if len(config.loop_deltas) == 0:
                    config.loop_deltas = [0]
                else:
                    config.loop_deltas = [0]

            for delta in config.loop_deltas:
                if loop_config is not None:
                    loop_config.loops = original_loops + delta

                torch.cuda.synchronize()
                eval_start_s = time.perf_counter()
                metrics = evaluate(
                    config,
                    train_state_eval,
                    eval_loader,
                    eval_metadata,
                    evaluators,
                    rank=RANK,
                    world_size=WORLD_SIZE,
                    cpu_group=CPU_PROCESS_GROUP,
                    early_eval=True,
                )
                torch.cuda.synchronize()
                eval_time_h = (time.perf_counter() - eval_start_s) / 3600.0
                if RANK == 0 and metrics is not None:
                    _add_eval_timing_metrics(train_state, metrics, eval_time_h=eval_time_h)
                    wandb.log(metrics, step=train_state.step)

            if loop_config is not None:
                loop_config.loops = original_loops

            if config.ema and ema_helper is not None and train_state_eval is not train_state:
                del train_state_eval

        ############ Train Iter
        train_state.model.train()
        active_train_step_start_s = None

        for set_name, batch, global_batch_size in train_loader:
            if train_state.accum_step == 0:
                torch.cuda.synchronize()
                active_train_step_start_s = time.perf_counter()

            before_step = train_state.step
            metrics = train_batch(
                config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE
            )
            stepped = train_state.step > before_step

            # EMA update
            if stepped and config.ema and ema_helper is not None:
                ema_helper.update(train_state.model)

            if stepped:
                torch.cuda.synchronize()
                if active_train_step_start_s is not None:
                    step_time_h = (time.perf_counter() - active_train_step_start_s) / 3600.0
                    train_state.last_step_time_h = step_time_h
                    train_state.train_time_h += step_time_h
                active_train_step_start_s = None

            if RANK == 0 and stepped:
                if train_state.step % 10 == 0 or train_state.step >= train_state.total_steps:
                    if metrics is None:
                        metrics = {}
                    _add_train_timing_metrics(train_state, metrics)
                    wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)

        ############ Evaluation
        if eval_loader is not None and eval_metadata is not None:
            # 选择用于评估的 train_state（EMA 或原始）
            if config.ema and ema_helper is not None:
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state

            train_state_eval.model.eval()
            loop_config = _get_loop_config(train_state_eval.model)
            if loop_config is not None:
                original_loops = loop_config.loops
                if len(config.loop_deltas) == 0:
                    config.loop_deltas = [0]
                else:
                    config.loop_deltas = [0]
            for delta in config.loop_deltas:
                if loop_config is not None:
                    loop_config.loops = original_loops + delta

                torch.cuda.synchronize()
                eval_start_s = time.perf_counter()
                metrics = evaluate(
                    config,
                    train_state_eval,
                    eval_loader,
                    eval_metadata,
                    evaluators,
                    rank=RANK,
                    world_size=WORLD_SIZE,
                    cpu_group=CPU_PROCESS_GROUP,
                    early_eval=True,
                )
                torch.cuda.synchronize()
                eval_time_h = (time.perf_counter() - eval_start_s) / 3600.0
                if RANK == 0 and metrics is not None:
                    _add_eval_timing_metrics(train_state, metrics, eval_time_h=eval_time_h)
                    wandb.log(metrics, step=train_state.step)

            if loop_config is not None:
                loop_config.loops = original_loops

            # 用完临时的 eval state 后可以丢掉，节省显存/内存
            if config.ema and ema_helper is not None and train_state_eval is not train_state:
                del train_state_eval

        if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
            if config.ema and ema_helper is not None:
                # 临时拷贝一个带 EMA 权重的 state 来保存
                ts_to_save = copy.deepcopy(train_state)
                ts_to_save.model = ema_helper.ema_copy(ts_to_save.model)
                save_train_state(config, ts_to_save)
                del ts_to_save
            else:
                save_train_state(config, train_state)


    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    sys.argv = _rewrite_cli_checkpoint_file_flag(sys.argv)
    launch()
