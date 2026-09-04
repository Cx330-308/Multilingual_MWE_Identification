from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LanguageSamplingPlan:
    alpha: float
    chunk_counts: dict[str, int]
    language_weights: dict[str, float]
    expected_probabilities: dict[str, float]
    chunk_weights: tuple[float, ...]

    @property
    def uses_weighted_sampling(self) -> bool:
        return self.alpha < 1.0

    def summary(self) -> dict[str, object]:
        return {
            "strategy": "language_power" if self.uses_weighted_sampling else "natural",
            "alpha": self.alpha,
            "replacement": self.uses_weighted_sampling,
            "samples_per_epoch": len(self.chunk_weights),
            "source_chunk_counts": self.chunk_counts,
            "per_chunk_language_weights": self.language_weights,
            "expected_language_probabilities": self.expected_probabilities,
        }


def build_language_sampling_plan(
    chunk_languages: Sequence[str],
    alpha: float,
) -> LanguageSamplingPlan:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Language sampling alpha must be between 0.0 and 1.0.")
    if not chunk_languages:
        raise ValueError("Cannot build a language sampling plan for an empty dataset.")

    counts = Counter(chunk_languages)
    language_weights = {
        language: count ** (alpha - 1.0)
        for language, count in sorted(counts.items())
    }
    language_masses = {
        language: count**alpha
        for language, count in sorted(counts.items())
    }
    total_mass = sum(language_masses.values())
    expected_probabilities = {
        language: mass / total_mass
        for language, mass in language_masses.items()
    }
    chunk_weights = tuple(language_weights[language] for language in chunk_languages)

    return LanguageSamplingPlan(
        alpha=alpha,
        chunk_counts=dict(sorted(counts.items())),
        language_weights=language_weights,
        expected_probabilities=expected_probabilities,
        chunk_weights=chunk_weights,
    )


def create_weighted_sampler(
    plan: LanguageSamplingPlan,
    dataset_length: int,
    seed: int,
) -> object:
    if not plan.uses_weighted_sampling:
        raise ValueError("Natural sampling should use the Trainer's default sampler.")
    if dataset_length != len(plan.chunk_weights):
        raise ValueError(
            "Language sampling weights do not match the training dataset: "
            f"{len(plan.chunk_weights)} weights for {dataset_length} chunks."
        )

    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(plan.chunk_weights, dtype=torch.double),
        num_samples=len(plan.chunk_weights),
        replacement=True,
        generator=generator,
    )
