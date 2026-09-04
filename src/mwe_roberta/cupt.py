from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


@dataclass
class CuptToken:
    columns: list[str]

    @property
    def token_id(self) -> str:
        return self.columns[0]

    @property
    def form(self) -> str:
        return self.columns[1]

    @property
    def mwe(self) -> str:
        return self.columns[10] if len(self.columns) > 10 else "*"

    @mwe.setter
    def mwe(self, value: str) -> None:
        while len(self.columns) < 11:
            self.columns.append("_")
        self.columns[10] = value

    @property
    def is_syntactic_token(self) -> bool:
        return self.token_id.isdigit()

    def serialize(self) -> str:
        return "\t".join(self.columns)


@dataclass
class CuptSentence:
    comments: list[str]
    tokens: list[CuptToken]

    @property
    def words(self) -> list[str]:
        return [token.form for token in self.tokens if token.is_syntactic_token]

    @property
    def mwe_columns(self) -> list[str]:
        return [token.mwe for token in self.tokens if token.is_syntactic_token]

    def with_predicted_mwes(self, predicted_mwes: Sequence[str]) -> "CuptSentence":
        if len(predicted_mwes) != len(self.words):
            raise ValueError(
                f"Prediction length {len(predicted_mwes)} does not match "
                f"sentence length {len(self.words)}."
            )
        copied = CuptSentence(
            comments=list(self.comments),
            tokens=[CuptToken(list(token.columns)) for token in self.tokens],
        )
        pred_iter = iter(predicted_mwes)
        for token in copied.tokens:
            if token.is_syntactic_token:
                token.mwe = next(pred_iter)
            else:
                token.mwe = "*"
        return copied

    def serialize(self) -> str:
        lines = []
        lines.extend(self.comments)
        lines.extend(token.serialize() for token in self.tokens)
        return "\n".join(lines)


def iter_cupt(path: Path | str) -> Iterator[CuptSentence]:
    comments: list[str] = []
    tokens: list[CuptToken] = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                if comments or tokens:
                    yield CuptSentence(comments=comments, tokens=tokens)
                    comments = []
                    tokens = []
                continue
            if line.startswith("#"):
                comments.append(line)
            else:
                columns = line.split("\t")
                if len(columns) < 11:
                    raise ValueError(f"Expected 11 CUPT columns in {path}: {line}")
                tokens.append(CuptToken(columns=columns))
    if comments or tokens:
        yield CuptSentence(comments=comments, tokens=tokens)


def read_cupt(path: Path | str, limit: int | None = None) -> list[CuptSentence]:
    sentences = []
    for sentence in iter_cupt(path):
        sentences.append(sentence)
        if limit is not None and len(sentences) >= limit:
            break
    return sentences


def write_cupt(sentences: Iterable[CuptSentence], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for sentence in sentences:
            handle.write(sentence.serialize())
            handle.write("\n\n")


def language_dirs(data_dir: Path | str) -> list[Path]:
    root = Path(data_dir)
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name != "tools" and not path.name.startswith(".")
    )


def resolve_languages(data_dir: Path | str, languages: str | Sequence[str]) -> list[str]:
    available = [path.name for path in language_dirs(data_dir)]
    if isinstance(languages, str):
        requested = available if languages.lower() == "all" else [
            item.strip().upper() for item in languages.split(",") if item.strip()
        ]
    else:
        requested = [item.upper() for item in languages]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"Unknown language directories: {', '.join(missing)}")
    return requested


def split_path(data_dir: Path | str, language: str, split: str) -> Path:
    suffix = "test.blind.cupt" if split == "test.blind" else f"{split}.cupt"
    return Path(data_dir) / language / suffix


def available_split_languages(data_dir: Path | str, languages: Sequence[str], split: str) -> list[str]:
    return [language for language in languages if split_path(data_dir, language, split).exists()]


def concatenate(paths: Sequence[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        for path in paths:
            last_line = "\n"
            with path.open(encoding="utf-8") as src:
                for line in src:
                    out.write(line)
                    last_line = line
                if not last_line.endswith("\n"):
                    out.write("\n")
