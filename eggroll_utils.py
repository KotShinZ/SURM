from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional

import torch
from torch import nn


LOW_RANK = "low_rank"
GAUSSIAN = "gaussian"
PerturbationKind = Literal["low_rank", "gaussian"]


@dataclass
class EvolvableParameter:
    name: str
    parameter: nn.Parameter
    kind: PerturbationKind


def _is_embedding_parameter(name: str) -> bool:
    lowered = name.lower()
    return (
        "puzzle_emb" in lowered
        or "embed_tokens" in lowered
        or "embedding_weight" in lowered
        or ".embedding." in lowered
    )


def collect_evolvable_parameters(
    model: nn.Module,
    *,
    include_small_tensors: bool = False,
    small_tensor_limit: int = 4096,
) -> List[EvolvableParameter]:
    evolvable: List[EvolvableParameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if _is_embedding_parameter(name):
            continue

        if parameter.ndim == 2:
            evolvable.append(EvolvableParameter(name=name, parameter=parameter, kind=LOW_RANK))
            continue

        if include_small_tensors:
            include_gaussian = parameter.numel() <= small_tensor_limit or "dwconv" in name.lower()
            if include_gaussian:
                evolvable.append(EvolvableParameter(name=name, parameter=parameter, kind=GAUSSIAN))

    evolvable.sort(key=lambda item: item.name)
    return evolvable


def normalize_fitness(
    raw_scores: Iterable[float | int | torch.Tensor],
    *,
    eps: float = 1e-5,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    scores = torch.as_tensor(list(raw_scores), dtype=torch.float32, device=device)
    if scores.numel() == 0:
        raise ValueError("normalize_fitness requires at least one score.")

    mean = scores.mean()
    var = scores.var(unbiased=False)
    return (scores - mean) / torch.sqrt(var + eps)


def extract_parameter_state(specs: Iterable[EvolvableParameter]) -> Dict[str, torch.Tensor]:
    return {
        spec.name: spec.parameter.detach().cpu().clone()
        for spec in specs
    }


def load_parameter_state_(specs: Iterable[EvolvableParameter], state: Dict[str, torch.Tensor]) -> None:
    missing = [spec.name for spec in specs if spec.name not in state]
    if missing:
        raise KeyError(f"Missing parameter(s) in saved Eggroll state: {missing}")

    with torch.no_grad():
        for spec in specs:
            source = state[spec.name].to(device=spec.parameter.device, dtype=spec.parameter.dtype)
            spec.parameter.copy_(source)


class PopulationPerturbations:
    def __init__(
        self,
        specs: List[EvolvableParameter],
        *,
        population: int,
        rank: int,
        sigma: float,
        seed: int,
    ) -> None:
        if population <= 0 or population % 2 != 0:
            raise ValueError(f"population must be a positive even integer, got {population}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if not specs:
            raise ValueError("PopulationPerturbations requires at least one evolvable parameter.")

        self.specs = list(specs)
        self.population = int(population)
        self.num_pairs = self.population // 2
        self.rank = int(rank)
        self.sigma = float(sigma)
        self.seed = int(seed)
        self.device = self.specs[0].parameter.device

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)

        self._low_rank_factors: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._gaussian_noises: Dict[str, torch.Tensor] = {}

        for spec in self.specs:
            param = spec.parameter.detach()
            if spec.kind == LOW_RANK:
                out_dim, in_dim = param.shape
                left = torch.randn(
                    (self.num_pairs, out_dim, self.rank),
                    generator=generator,
                    device=param.device,
                    dtype=torch.float32,
                )
                right = torch.randn(
                    (self.num_pairs, in_dim, self.rank),
                    generator=generator,
                    device=param.device,
                    dtype=torch.float32,
                )
                left.mul_(self.sigma / math.sqrt(self.rank))
                self._low_rank_factors[spec.name] = (left, right)
            elif spec.kind == GAUSSIAN:
                noise = torch.randn(
                    (self.num_pairs, *param.shape),
                    generator=generator,
                    device=param.device,
                    dtype=torch.float32,
                )
                noise.mul_(self.sigma)
                self._gaussian_noises[spec.name] = noise
            else:
                raise ValueError(f"Unsupported perturbation kind: {spec.kind}")

    def _positive_delta(self, spec: EvolvableParameter, pair_index: int) -> torch.Tensor:
        if spec.kind == LOW_RANK:
            left, right = self._low_rank_factors[spec.name]
            return left[pair_index] @ right[pair_index].transpose(0, 1)
        if spec.kind == GAUSSIAN:
            return self._gaussian_noises[spec.name][pair_index]
        raise ValueError(f"Unsupported perturbation kind: {spec.kind}")

    def apply_member_(self, member_index: int) -> Dict[str, torch.Tensor]:
        if member_index < 0 or member_index >= self.population:
            raise IndexError(f"member_index {member_index} is out of range for population {self.population}")

        pair_index = member_index // 2
        sign = 1.0 if member_index % 2 == 0 else -1.0
        active_deltas: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for spec in self.specs:
                positive_delta = self._positive_delta(spec, pair_index)
                delta = positive_delta if sign > 0 else -positive_delta
                typed_delta = delta.to(dtype=spec.parameter.dtype)
                spec.parameter.add_(typed_delta)
                active_deltas[spec.name] = typed_delta

        return active_deltas

    def revert_member_(self, active_deltas: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for spec in self.specs:
                spec.parameter.sub_(active_deltas[spec.name])

    def compute_update_tensors(self, normalized_fitness: torch.Tensor) -> Dict[str, torch.Tensor]:
        if normalized_fitness.shape != (self.population,):
            raise ValueError(
                f"normalized_fitness must have shape ({self.population},), got {tuple(normalized_fitness.shape)}"
            )

        pair_coefficients = (
            normalized_fitness[0::2] - normalized_fitness[1::2]
        ) * (math.sqrt(self.population) / self.population)

        updates: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for spec in self.specs:
                accumulated = torch.zeros_like(spec.parameter, dtype=torch.float32)
                for pair_index in range(self.num_pairs):
                    coeff = float(pair_coefficients[pair_index].item())
                    if coeff == 0.0:
                        continue
                    accumulated.add_(self._positive_delta(spec, pair_index), alpha=coeff)
                updates[spec.name] = accumulated

        return updates


def apply_updates_(
    specs: Iterable[EvolvableParameter],
    updates: Dict[str, torch.Tensor],
    *,
    lr: float,
) -> None:
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")

    with torch.no_grad():
        for spec in specs:
            update = updates[spec.name].to(device=spec.parameter.device, dtype=spec.parameter.dtype)
            spec.parameter.add_(update, alpha=lr)
