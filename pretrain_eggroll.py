from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import hydra
import pydantic
import torch
from omegaconf import DictConfig, OmegaConf

from pretrain import (
    ArchConfig,
    PretrainConfig,
    TrainState,
    create_dataloader,
    create_model,
    load_checkpoint,
    save_code_and_config,
)
from models.losses import IGNORE_LABEL_ID


_EGGROLL_TRAINER_PATH = Path(__file__).resolve().parent / "eggroll-trainer"
if str(_EGGROLL_TRAINER_PATH) not in sys.path:
    sys.path.insert(0, str(_EGGROLL_TRAINER_PATH))

from eggroll_trainer import EGGROLLTrainer  # noqa: E402


class EggrollPretrainConfig(PretrainConfig):
    model_config = pydantic.ConfigDict(extra="allow")

    # EGGROLL hyperparameters.  The default learning rate follows `lr` when left unset.
    eggroll_population_size: int = 32
    eggroll_learning_rate: Optional[float] = None
    eggroll_sigma: float = 0.01
    eggroll_rank: int = 1
    eggroll_noise_reuse: int = 2
    eggroll_group_size: int = 0
    eggroll_freeze_nonlora: bool = False

    # Training loop controls.
    eggroll_generations: Optional[int] = None
    eggroll_batches_per_generation: int = 1
    eggroll_objective_steps: int = 1
    eggroll_eval_every: int = 1
    eggroll_checkpoint_every: int = 0
    eggroll_assert_loss_drop: bool = False
    eggroll_fitness_metric: str = "loss"

    # Make this script runnable as a Sudoku trainer without a long CLI override list.
    eggroll_sudoku_defaults: bool = True


class EggrollBatchFitness:
    def __init__(
        self,
        batches: Sequence[Tuple[str, Dict[str, torch.Tensor], int]],
        *,
        objective_steps: int,
        fitness_metric: str,
    ):
        self.objective_steps = max(1, objective_steps)
        self.fitness_metric = fitness_metric
        self.set_batches(batches)

    def set_batches(self, batches: Sequence[Tuple[str, Dict[str, torch.Tensor], int]]) -> None:
        if len(batches) == 0:
            raise ValueError("EGGROLL fitness needs at least one batch.")
        self.batches = list(batches)

    def evaluate(self, model: torch.nn.Module) -> Dict[str, float]:
        was_training = model.training
        model.eval()
        inner_model = getattr(model, "model", None)
        model_config = getattr(inner_model, "config", None)
        forward_dtype = getattr(torch, getattr(model_config, "forward_dtype", "bfloat16"))

        total_loss = 0.0
        total_examples = 0
        total_tokens = 0
        total_correct_tokens = 0
        total_exact = 0
        total_sequences = 0
        with torch.inference_mode():
            for _set_name, batch, global_batch_size in self.batches:
                with torch.device(batch["inputs"].device):
                    carry = model.initial_carry(batch)  # type: ignore[attr-defined]
                loss = None
                preds = None
                all_finish = torch.tensor(False, device=batch["inputs"].device)

                for _ in range(self.objective_steps):
                    carry.current_hidden = carry.current_hidden.to(forward_dtype)
                    carry, loss, _metrics, returned, all_finish = model(
                        carry=carry,
                        batch=batch,
                        return_keys={"preds"},
                        compute_target_q=False,
                    )
                    preds = returned.get("preds")
                    if bool(all_finish):
                        break

                if loss is None:
                    raise RuntimeError("Model did not produce a loss.")
                if preds is None:
                    raise RuntimeError("Model did not return predictions.")
                total_loss += float(loss.detach().to(torch.float32).item())
                total_examples += int(global_batch_size)

                labels = batch["labels"]
                valid = labels != IGNORE_LABEL_ID
                correct = valid & (preds == labels)
                valid_counts = valid.sum(dim=-1)
                total_tokens += int(valid_counts.sum().item())
                total_correct_tokens += int(correct.sum().item())
                total_sequences += int((valid_counts > 0).sum().item())
                total_exact += int(((correct.sum(dim=-1) == valid_counts) & (valid_counts > 0)).sum().item())

        if was_training:
            model.train()

        return {
            "loss": total_loss / max(total_examples, 1),
            "accuracy": total_correct_tokens / max(total_tokens, 1),
            "exact_accuracy": total_exact / max(total_sequences, 1),
        }

    def loss(self, model: torch.nn.Module) -> float:
        return self.evaluate(model)["loss"]

    def __call__(self, model: torch.nn.Module) -> float:
        metrics = self.evaluate(model)
        if self.fitness_metric == "loss":
            return -metrics["loss"]
        if self.fitness_metric == "accuracy":
            return metrics["accuracy"]
        if self.fitness_metric == "exact_accuracy":
            return metrics["exact_accuracy"]
        if self.fitness_metric == "accuracy_minus_loss":
            return metrics["accuracy"] - 0.1 * metrics["loss"]
        raise ValueError(f"Unsupported eggroll_fitness_metric: {self.fitness_metric}")


class CyclingBatchProvider:
    def __init__(self, dataloader: torch.utils.data.DataLoader, device: torch.device):
        self.dataloader = dataloader
        self.device = device
        self._iterator = iter(dataloader)

    def next_batches(self, count: int) -> List[Tuple[str, Dict[str, torch.Tensor], int]]:
        batches: List[Tuple[str, Dict[str, torch.Tensor], int]] = []
        while len(batches) < count:
            try:
                set_name, batch, global_batch_size = next(self._iterator)
            except StopIteration:
                self._iterator = iter(self.dataloader)
                set_name, batch, global_batch_size = next(self._iterator)

            batch = {
                key: value.to(self.device, non_blocking=True)
                for key, value in batch.items()
            }
            batches.append((set_name, batch, int(global_batch_size)))

        return batches


_URM_BASE_DEFAULTS: Dict[str, Any] = {
    "loops": 16,
    "H_cycles": 4,
    "L_cycles": 3,
    "num_layers": 8,
    "grid_height": 0,
    "grid_width": 0,
    "use_act": False,
    "patch_io_enabled": False,
    "patch_height": 2,
    "patch_width": 2,
    "patch_pre_embedding_size": 3,
}


_SUDOKU_DEFAULTS: Dict[str, Any] = {
    "loops": 32,
    "H_cycles": 1,
    "L_cycles": 6,
    "num_layers": 4,
    "grid_height": 9,
    "grid_width": 9,
    "use_act": True,
    "halt_norm_in_use_act": True,
    "norm_diff_max": 0.1,
    "norm_diff_min": 0.001,
    "patch_io_enabled": True,
    "patch_height": 2,
    "patch_width": 2,
    "patch_pre_embedding_size": 3,
}


def _copy_arch_with_extra(arch: ArchConfig, extra: Dict[str, Any]) -> ArchConfig:
    return ArchConfig(name=arch.name, loss=arch.loss, **extra)


def apply_sudoku_defaults(config: EggrollPretrainConfig) -> EggrollPretrainConfig:
    if not config.eggroll_sudoku_defaults:
        return config

    updates: Dict[str, Any] = {}
    should_apply_task_defaults = config.data_path == "data/arc-aug-1000"
    if config.data_path == "data/arc-aug-1000":
        updates["data_path"] = "data/sudoku-extreme-1k-aug-1000"
    if config.evaluators and (should_apply_task_defaults or "sudoku" in config.data_path.lower()):
        updates["evaluators"] = []
    if config.run_name is None:
        updates["run_name"] = "URM-sudoku-eggroll"
    if config.checkpoint_path is None:
        updates["checkpoint_path"] = "checkpoints/URM-sudoku-eggroll"

    if should_apply_task_defaults:
        arch_extra = dict(config.arch.__pydantic_extra__ or {})
        for key, value in _SUDOKU_DEFAULTS.items():
            current = arch_extra.get(key)
            if key not in arch_extra or current == _URM_BASE_DEFAULTS.get(key):
                arch_extra[key] = value
        updates["arch"] = _copy_arch_with_extra(config.arch, arch_extra)

    return config.model_copy(update=updates)


def load_eggroll_config(hydra_config: DictConfig) -> EggrollPretrainConfig:
    config_dict = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(config_dict, dict):
        raise TypeError("Hydra config did not resolve to a dictionary.")
    return apply_sudoku_defaults(EggrollPretrainConfig(**config_dict))


def _make_train_state_for_loading(
    model: torch.nn.Module,
    total_steps: int,
) -> TrainState:
    return TrainState(
        model=model,
        optimizers=[],
        optimizer_lrs=[],
        carry=None,
        step=0,
        total_steps=total_steps,
    )


def save_eggroll_state(
    config: EggrollPretrainConfig,
    model: torch.nn.Module,
    trainer: EGGROLLTrainer,
    generation: int,
    metrics: Dict[str, Any],
    file_name: Optional[str] = None,
) -> None:
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    state = {
        "step": generation,
        "model_state_dict": model.state_dict(),
        "optimizer_states": [trainer.state_dict()],
        "eggroll_metrics": metrics,
    }
    if file_name is None:
        file_name = f"step_{generation}.pt"
    torch.save(state, os.path.join(config.checkpoint_path, file_name))


def create_eggroll_trainer(
    config: EggrollPretrainConfig,
    model: torch.nn.Module,
    fitness: EggrollBatchFitness,
    device: torch.device,
) -> EGGROLLTrainer:
    if config.eggroll_noise_reuse > 0 and config.eggroll_population_size % 2 != 0:
        raise ValueError("eggroll_population_size must be even when eggroll_noise_reuse > 0.")
    if config.eggroll_group_size > 0 and config.eggroll_population_size % config.eggroll_group_size != 0:
        raise ValueError("eggroll_population_size must be divisible by eggroll_group_size.")

    return EGGROLLTrainer(
        model.parameters(),
        model=model,
        fitness_fn=fitness,
        population_size=config.eggroll_population_size,
        learning_rate=config.eggroll_learning_rate or config.lr,
        sigma=config.eggroll_sigma,
        rank=config.eggroll_rank,
        noise_reuse=config.eggroll_noise_reuse,
        group_size=config.eggroll_group_size,
        freeze_nonlora=config.eggroll_freeze_nonlora,
        device=device,
        seed=config.seed,
    )


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig) -> None:
    if "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise NotImplementedError(
            "pretrain_eggroll.py currently supports single-process EGGROLL training only."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "pretrain_eggroll.py requires CUDA because the current URM implementation allocates CUDA RNGs."
        )

    os.environ.setdefault("DISABLE_COMPILE", "1")
    config = load_eggroll_config(hydra_config)
    torch.random.manual_seed(config.seed)

    if config.checkpoint_path is not None:
        os.makedirs(config.checkpoint_path, exist_ok=True)
        save_code_and_config(config, config.checkpoint_path)

    print("Config:")
    print(config.model_dump_json(indent=2))

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=max(1, config.eval_interval or 1),
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1,
    )

    model, _unused_optimizers, _unused_lrs = create_model(config, train_metadata, rank=0, world_size=1)
    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameter tensors: {sum(1 for _ in model.parameters())}")
    print(f"Total parameters visible to EGGROLL: {total_params}")

    generations = config.eggroll_generations
    if generations is None:
        generations = max(1, config.epochs)

    train_state = _make_train_state_for_loading(model, generations)
    if config.load_checkpoint is not None:
        load_config = copy.copy(config)
        load_config.load_optimizer_state = False
        load_checkpoint(train_state, load_config, rank=0)

    device = torch.device("cuda")
    batch_provider = CyclingBatchProvider(train_loader, device=device)
    probe_batches = batch_provider.next_batches(max(1, config.eggroll_batches_per_generation))
    probe_fitness = EggrollBatchFitness(
        probe_batches,
        objective_steps=config.eggroll_objective_steps,
        fitness_metric=config.eggroll_fitness_metric,
    )
    fitness = EggrollBatchFitness(
        batch_provider.next_batches(max(1, config.eggroll_batches_per_generation)),
        objective_steps=config.eggroll_objective_steps,
        fitness_metric=config.eggroll_fitness_metric,
    )
    trainer = create_eggroll_trainer(config, model, fitness, device=device)

    initial_probe = probe_fitness.evaluate(model)
    initial_loss = initial_probe["loss"]
    print(
        "Initial probe metrics: "
        f"loss={initial_probe['loss']:.6f}, "
        f"accuracy={initial_probe['accuracy']:.6f}, "
        f"exact_accuracy={initial_probe['exact_accuracy']:.6f}"
    )

    final_loss = initial_loss
    best_probe_metrics = dict(initial_probe)
    best_probe_generation = 0
    best_probe_state_dict = copy.deepcopy(model.state_dict())
    last_metrics: Dict[str, Any] = {
        "initial_loss": initial_loss,
        "initial_accuracy": initial_probe["accuracy"],
        "initial_exact_accuracy": initial_probe["exact_accuracy"],
    }
    for generation in range(1, generations + 1):
        fitness.set_batches(batch_provider.next_batches(max(1, config.eggroll_batches_per_generation)))
        should_eval = generation == generations or generation % max(1, config.eggroll_eval_every) == 0
        before_loss = fitness.loss(model) if should_eval else None

        with torch.no_grad():
            trainer_metrics = trainer.step()

        if should_eval:
            batch_metrics = fitness.evaluate(model)
            probe_metrics = probe_fitness.evaluate(model)
            final_loss = probe_metrics["loss"]
            batch_delta = batch_metrics["loss"] - before_loss if before_loss is not None else float("nan")
            probe_delta = final_loss - initial_loss
            if (
                probe_metrics["accuracy"] > best_probe_metrics["accuracy"]
                or (
                    probe_metrics["accuracy"] == best_probe_metrics["accuracy"]
                    and probe_metrics["loss"] < best_probe_metrics["loss"]
                )
            ):
                best_probe_generation = generation
                best_probe_metrics = dict(probe_metrics)
                best_probe_state_dict = copy.deepcopy(model.state_dict())

            print(
                "Generation "
                f"{generation}/{generations}: "
                f"batch_loss={batch_metrics['loss']:.6f}, "
                f"probe_loss={probe_metrics['loss']:.6f}, "
                f"probe_acc={probe_metrics['accuracy']:.6f}, "
                f"probe_exact={probe_metrics['exact_accuracy']:.6f}, "
                f"batch_delta={batch_delta:+.6f}, "
                f"probe_delta={probe_delta:+.6f}, "
                f"mean_fitness={trainer_metrics['mean_fitness']:.6f}, "
                f"best_fitness={trainer_metrics['best_fitness']:.6f}, "
                f"std_fitness={trainer_metrics['std_fitness']:.6f}"
            )
            print(
                "Best probe so far: "
                f"generation={best_probe_generation}, "
                f"loss={best_probe_metrics['loss']:.6f}, "
                f"accuracy={best_probe_metrics['accuracy']:.6f}, "
                f"exact_accuracy={best_probe_metrics['exact_accuracy']:.6f}"
            )

        last_metrics = {
            **trainer_metrics,
            "initial_loss": initial_loss,
            "initial_accuracy": initial_probe["accuracy"],
            "initial_exact_accuracy": initial_probe["exact_accuracy"],
            "final_loss": final_loss,
            "best_probe_generation": best_probe_generation,
            "best_probe_loss": best_probe_metrics["loss"],
            "best_probe_accuracy": best_probe_metrics["accuracy"],
            "best_probe_exact_accuracy": best_probe_metrics["exact_accuracy"],
        }
        if should_eval:
            last_metrics.update(
                {
                    "final_accuracy": probe_metrics["accuracy"],
                    "final_exact_accuracy": probe_metrics["exact_accuracy"],
                }
            )

        if (
            config.eggroll_checkpoint_every > 0
            and generation % config.eggroll_checkpoint_every == 0
        ):
            save_eggroll_state(config, model, trainer, generation, last_metrics)

    if config.checkpoint_path is not None:
        metrics_path = Path(config.checkpoint_path) / "eggroll_metrics.json"
        metrics_path.write_text(json.dumps(last_metrics, indent=2), encoding="utf-8")
        save_eggroll_state(config, model, trainer, generations, last_metrics)
        current_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_probe_state_dict)
        save_eggroll_state(
            config,
            model,
            trainer,
            best_probe_generation,
            last_metrics,
            file_name="best_probe.pt",
        )
        model.load_state_dict(current_state)

    final_accuracy = last_metrics.get("final_accuracy", initial_probe["accuracy"])
    final_exact_accuracy = last_metrics.get("final_exact_accuracy", initial_probe["exact_accuracy"])
    print(
        "Final probe metrics: "
        f"loss={final_loss:.6f}, "
        f"accuracy={final_accuracy:.6f}, "
        f"exact_accuracy={final_exact_accuracy:.6f}"
    )
    print(
        "Best probe metrics: "
        f"generation={best_probe_generation}, "
        f"loss={best_probe_metrics['loss']:.6f}, "
        f"accuracy={best_probe_metrics['accuracy']:.6f}, "
        f"exact_accuracy={best_probe_metrics['exact_accuracy']:.6f}"
    )
    print(f"Loss change: {final_loss - initial_loss:+.6f}")

    if config.eggroll_assert_loss_drop and final_loss > initial_loss:
        raise AssertionError(
            f"Expected loss to decrease, but initial={initial_loss:.6f}, final={final_loss:.6f}"
        )


if __name__ == "__main__":
    launch()
