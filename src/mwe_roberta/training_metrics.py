from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from .cupt import split_path, write_cupt
from .data import SentenceExample
from .hf_dataset import ChunkMetadata, TokenClassificationDataset, token_metrics
from .labels import bio_to_mwe_columns
from .official import (
    OfficialMweScore,
    average_evaluations,
    evaluator_paths,
    run_language_evaluation,
)


def decode_chunk_predictions(
    logits: Any,
    examples: Sequence[SentenceExample],
    chunk_metadata: Sequence[ChunkMetadata],
    id2label: dict[int, str],
) -> list[list[str]]:
    import numpy as np

    logits_array = np.asarray(logits)
    if logits_array.ndim != 3:
        raise ValueError(f"Expected [chunks, tokens, labels] logits, got {logits_array.shape}.")
    if logits_array.shape[0] != len(chunk_metadata):
        raise ValueError(
            f"Received {logits_array.shape[0]} prediction chunks for "
            f"{len(chunk_metadata)} chunk metadata entries."
        )

    per_example: list[dict[int, list[Any]]] = [defaultdict(list) for _ in examples]
    for chunk_index, metadata in enumerate(chunk_metadata):
        seen_word_ids: set[int] = set()
        for token_index, word_id in enumerate(metadata.word_ids):
            if word_id is None or word_id in seen_word_ids:
                continue
            if token_index >= logits_array.shape[1]:
                raise ValueError(
                    f"Chunk {chunk_index} metadata has more tokens than its prediction tensor."
                )
            seen_word_ids.add(word_id)
            per_example[metadata.example_index][word_id].append(
                logits_array[chunk_index, token_index]
            )

    decoded: list[list[str]] = []
    for example_index, example in enumerate(examples):
        labels: list[str] = []
        for word_index in range(len(example.words)):
            candidates = per_example[example_index].get(word_index)
            if not candidates:
                labels.append("O")
                continue
            averaged_logits = np.mean(np.stack(candidates), axis=0)
            labels.append(id2label[int(np.argmax(averaged_logits))])
        decoded.append(labels)
    return decoded


class OfficialMetricsComputer:
    def __init__(
        self,
        *,
        data_dir: Path,
        output_dir: Path,
        languages: Sequence[str],
        examples: Sequence[SentenceExample],
        dataset: TokenClassificationDataset,
        id2label: dict[int, str],
    ) -> None:
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.languages = list(languages)
        self.examples = list(examples)
        self.dataset = dataset
        self.id2label = id2label
        self.eval_script, self.average_script = evaluator_paths(data_dir)
        self.evaluation_count = 0
        self.history: list[dict[str, Any]] = []

        if any(example.sentence is None for example in self.examples):
            raise ValueError("Official metrics require original CUPT sentences in dev examples.")
        if len(dataset.chunk_metadata) != len(dataset):
            raise ValueError(
                "Official metrics require track_metadata=True for the development dataset."
            )

    def __call__(self, eval_prediction: Any) -> dict[str, float]:
        metrics = token_metrics(eval_prediction)
        logits = (
            eval_prediction.predictions
            if hasattr(eval_prediction, "predictions")
            else eval_prediction[0]
        )
        if isinstance(logits, tuple):
            logits = logits[0]

        decoded = decode_chunk_predictions(
            logits,
            self.examples,
            self.dataset.chunk_metadata,
            self.id2label,
        )
        self.evaluation_count += 1
        evaluation_dir = self.output_dir / f"eval-{self.evaluation_count:03d}"
        evaluation_files = self._write_and_score(decoded, evaluation_dir)
        score = average_evaluations(
            average_script=self.average_script,
            evaluation_files=evaluation_files,
            output_path=evaluation_dir / "average.txt",
        )
        self._record(score, evaluation_dir)

        print(
            "Official dev macro: "
            f"P={score.precision:.4f} R={score.recall:.4f} F={score.f1:.4f} "
            f"@({score.evaluated_languages}/{score.requested_languages})"
        )
        metrics.update(
            {
                "official_mwe_macro_precision": score.precision,
                "official_mwe_macro_recall": score.recall,
                "official_mwe_macro_f1": score.f1,
            }
        )
        return metrics

    def _write_and_score(
        self,
        decoded: Sequence[Sequence[str]],
        evaluation_dir: Path,
    ) -> list[Path]:
        gold_by_language: dict[str, list[Any]] = {language: [] for language in self.languages}
        predictions_by_language: dict[str, list[Any]] = {
            language: [] for language in self.languages
        }

        if len(decoded) != len(self.examples):
            raise ValueError(
                f"Decoded {len(decoded)} sentences for {len(self.examples)} dev examples."
            )
        for example, labels in zip(self.examples, decoded):
            sentence = example.sentence
            if sentence is None:
                raise ValueError("Dev example is missing its original CUPT sentence.")
            gold_by_language[example.language].append(sentence)
            predicted_mwes = bio_to_mwe_columns(labels)
            predictions_by_language[example.language].append(
                sentence.with_predicted_mwes(predicted_mwes)
            )

        evaluation_files: list[Path] = []
        for language in self.languages:
            if not gold_by_language[language]:
                raise ValueError(f"No development examples are available for {language}.")
            gold_path = evaluation_dir / "gold" / language / "dev.cupt"
            prediction_path = evaluation_dir / "predictions" / language / "dev.system.cupt"
            write_cupt(gold_by_language[language], gold_path)
            write_cupt(predictions_by_language[language], prediction_path)
            evaluation_files.append(
                run_language_evaluation(
                    eval_script=self.eval_script,
                    gold_path=gold_path,
                    prediction_path=prediction_path,
                    train_path=split_path(self.data_dir, language, "train"),
                    output_path=evaluation_dir / language / "eval.txt",
                )
            )
        return evaluation_files

    def _record(self, score: OfficialMweScore, evaluation_dir: Path) -> None:
        self.history.append(
            {
                "evaluation": self.evaluation_count,
                "output_dir": str(evaluation_dir.resolve()),
                "precision": score.precision,
                "recall": score.recall,
                "f1": score.f1,
                "evaluated_languages": score.evaluated_languages,
                "requested_languages": score.requested_languages,
            }
        )
