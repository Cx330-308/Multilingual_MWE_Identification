from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .cupt import CuptSentence, read_cupt, split_path
from .labels import mwe_columns_to_bio


@dataclass
class SentenceExample:
    language: str
    source_path: str
    sentence_index: int
    words: list[str]
    labels: list[str] | None = None
    sentence: CuptSentence | None = None


def load_examples(
    data_dir: Path | str,
    languages: Sequence[str],
    split: str,
    limit_per_language: int | None = None,
    include_sentence: bool = False,
    require_gold: bool = True,
) -> list[SentenceExample]:
    examples: list[SentenceExample] = []
    for language in languages:
        path = split_path(data_dir, language, split)
        if not path.exists():
            continue
        for idx, sentence in enumerate(read_cupt(path, limit=limit_per_language)):
            labels = mwe_columns_to_bio(sentence) if require_gold else None
            examples.append(
                SentenceExample(
                    language=language,
                    source_path=str(path),
                    sentence_index=idx,
                    words=sentence.words,
                    labels=labels,
                    sentence=sentence if include_sentence else None,
                )
            )
    return examples
