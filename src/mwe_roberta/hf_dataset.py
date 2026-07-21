from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class EncodedChunk:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int] | None = None


class TokenClassificationDataset:
    def __init__(
        self,
        examples: Sequence[Any],
        tokenizer: Any,
        label2id: dict[str, int],
        max_length: int,
        stride: int,
        label_all_tokens: bool = False,
    ) -> None:
        self.chunks: list[dict[str, list[int]]] = []
        for example in examples:
            encoded = tokenizer(
                example.words,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
            )
            for batch_index in range(len(encoded["input_ids"])):
                word_ids = encoded.word_ids(batch_index=batch_index)
                aligned_labels: list[int] = []
                previous_word_id = None
                for word_id in word_ids:
                    if word_id is None:
                        aligned_labels.append(-100)
                    elif word_id != previous_word_id:
                        aligned_labels.append(label2id[example.labels[word_id]])
                    else:
                        if label_all_tokens:
                            label = example.labels[word_id]
                            if label.startswith("B-"):
                                label = "I-" + label[2:]
                            aligned_labels.append(label2id[label])
                        else:
                            aligned_labels.append(-100)
                    previous_word_id = word_id

                self.chunks.append(
                    {
                        "input_ids": encoded["input_ids"][batch_index],
                        "attention_mask": encoded["attention_mask"][batch_index],
                        "labels": aligned_labels,
                    }
                )

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.chunks[idx]


def token_metrics(eval_prediction: Any, o_label_id: int = 0) -> dict[str, float]:
    import numpy as np

    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    pred = predictions[mask]
    gold = labels[mask]

    if gold.size == 0:
        return {"token_accuracy": 0.0, "mwe_precision": 0.0, "mwe_recall": 0.0, "mwe_f1": 0.0}

    accuracy = float((pred == gold).mean())
    pred_positive = pred != o_label_id
    gold_positive = gold != o_label_id
    correct_positive = (pred == gold) & gold_positive

    precision = float(correct_positive.sum() / pred_positive.sum()) if pred_positive.any() else 0.0
    recall = float(correct_positive.sum() / gold_positive.sum()) if gold_positive.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "token_accuracy": accuracy,
        "mwe_precision": precision,
        "mwe_recall": recall,
        "mwe_f1": f1,
    }
