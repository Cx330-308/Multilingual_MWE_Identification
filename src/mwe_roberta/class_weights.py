from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ClassWeightPlan:
    labels: tuple[str, ...]
    alpha: float
    min_weight: float
    max_weight: float
    count_basis: str
    reference_count: float
    label_counts: tuple[float, ...]
    weights: tuple[float, ...]

    @property
    def uses_weighted_loss(self) -> bool:
        return self.alpha > 0.0

    def summary(self) -> dict[str, object]:
        absent_labels = [
            label
            for label, count in zip(self.labels, self.label_counts)
            if count == 0.0
        ]
        return {
            "strategy": "class_power" if self.uses_weighted_loss else "standard_cross_entropy",
            "alpha": self.alpha,
            "clip": {
                "min": self.min_weight,
                "max": self.max_weight,
            },
            "count_basis": self.count_basis,
            "reference_count": self.reference_count,
            "effective_label_counts": {
                label: count
                for label, count in zip(self.labels, self.label_counts)
            },
            "label_weights": {
                label: weight
                for label, weight in zip(self.labels, self.weights)
            },
            "absent_labels": absent_labels,
        }


def build_class_weight_plan(
    chunks: Sequence[Mapping[str, Sequence[int]]],
    labels: Sequence[str],
    alpha: float,
    min_weight: float,
    max_weight: float,
    chunk_sampling_weights: Sequence[float] | None = None,
) -> ClassWeightPlan:
    if not labels:
        raise ValueError("Cannot build class weights without labels.")
    if len(set(labels)) != len(labels):
        raise ValueError("Class-weight labels must be unique.")
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("Class-weight alpha must be finite and between 0.0 and 1.0.")
    if not math.isfinite(min_weight) or min_weight <= 0.0:
        raise ValueError("Minimum class weight must be finite and greater than zero.")
    if not math.isfinite(max_weight) or max_weight < min_weight:
        raise ValueError("Maximum class weight must be finite and at least the minimum.")

    if chunk_sampling_weights is None:
        chunk_multipliers = [1.0] * len(chunks)
        count_basis = "natural_chunks"
    else:
        if len(chunk_sampling_weights) != len(chunks):
            raise ValueError(
                "Chunk sampling weights do not match the training dataset: "
                f"{len(chunk_sampling_weights)} weights for {len(chunks)} chunks."
            )
        if any(not math.isfinite(weight) or weight <= 0.0 for weight in chunk_sampling_weights):
            raise ValueError("Chunk sampling weights must be finite and greater than zero.")
        total_sampling_weight = sum(chunk_sampling_weights)
        scale = len(chunks) / total_sampling_weight
        chunk_multipliers = [weight * scale for weight in chunk_sampling_weights]
        count_basis = "expected_sampled_epoch"

    label_counts = [0.0] * len(labels)
    for chunk, multiplier in zip(chunks, chunk_multipliers):
        if "labels" not in chunk:
            raise ValueError("Every training chunk must contain aligned labels.")
        for label_id in chunk["labels"]:
            if label_id == -100:
                continue
            if not isinstance(label_id, int) or not 0 <= label_id < len(labels):
                raise ValueError(f"Invalid aligned label id: {label_id!r}.")
            label_counts[label_id] += multiplier

    observed_counts = [count for count in label_counts if count > 0.0]
    if not observed_counts:
        raise ValueError("No trainable labels were found in the training chunks.")
    reference_count = sum(observed_counts) / len(observed_counts)

    if alpha == 0.0:
        weights = [1.0] * len(labels)
    else:
        weights = [
            max_weight
            if count == 0.0
            else min(
                max_weight,
                max(min_weight, (reference_count / count) ** alpha),
            )
            for count in label_counts
        ]

    return ClassWeightPlan(
        labels=tuple(labels),
        alpha=alpha,
        min_weight=min_weight,
        max_weight=max_weight,
        count_basis=count_basis,
        reference_count=reference_count,
        label_counts=tuple(label_counts),
        weights=tuple(weights),
    )


def weighted_token_cross_entropy(
    logits: Any,
    targets: Any,
    class_weights: Any,
) -> Any:
    import torch.nn.functional as functional

    weights = class_weights.to(device=logits.device, dtype=logits.dtype)
    return functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        weight=weights,
        ignore_index=-100,
    )
