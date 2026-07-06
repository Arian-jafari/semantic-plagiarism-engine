"""
Dataset loading helpers for text files and labeled pair CSV files.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .preprocessing import normalize_text, remove_stopwords, tokenize

TEXT_EXTENSIONS = {".txt", ".md", ".text"}


@dataclass(frozen=True)
class PairRecord:
    """
    One labeled or unlabeled pair of texts.
    """

    text_a: str
    text_b: str
    label: int | None = None
    pair_id: str | None = None


def read_text_file(path: str | Path) -> str:
    """
    Read a text file as UTF-8, replacing malformed bytes.
    """

    return Path(path).read_text(encoding="utf-8", errors="replace")


def load_text_corpus(directory: str | Path) -> dict[str, str]:
    """
    Load supported text files from a directory recursively.
    """

    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus path is not a directory: {root}")

    documents: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS:
            doc_id = str(path.relative_to(root))
            documents[doc_id] = read_text_file(path)

    return documents


def token_pipeline(text: str, remove_stops: bool = True) -> list[str]:
    """
    Normalize and tokenize a document for token-based methods.
    """

    tokens = tokenize(normalize_text(text))
    if remove_stops:
        tokens = remove_stopwords(tokens)
    return tokens


def preprocessed_text(text: str, remove_stops: bool = True) -> str:
    """
    Return normalized, tokenized text as a single space-separated string.
    """

    return " ".join(token_pipeline(text, remove_stops=remove_stops))


def load_pair_csv(
    path: str | Path,
    text_col_a: str,
    text_col_b: str,
    label_col: str | None = None,
    limit: int | None = None,
) -> list[PairRecord]:
    """
    Load labeled text pairs from a CSV file.
    """

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided.")

    records: list[PairRecord] = []
    with Path(path).open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {text_col_a, text_col_b} - set(reader.fieldnames or [])
        if label_col:
            missing = missing | ({label_col} - set(reader.fieldnames or []))
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"Missing required CSV columns: {missing_columns}")

        for index, row in enumerate(reader):
            label = None
            if label_col:
                label = int(float(row[label_col]))

            records.append(
                PairRecord(
                    text_a=row[text_col_a] or "",
                    text_b=row[text_col_b] or "",
                    label=label,
                    pair_id=str(index),
                )
            )

            if limit is not None and len(records) >= limit:
                break

    return records
