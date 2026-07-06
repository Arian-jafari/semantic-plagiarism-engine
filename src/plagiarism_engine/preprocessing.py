"""
Text preprocessing utilities for the plagiarism engine.
"""

from __future__ import annotations

import re
import string

# A small English stopword list.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at",
    "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on",
    "that", "the", "to", "was", "were",
    "will", "with", "this", "these", "those",
    "or", "if", "into", "than", "then"
}


def normalize_text(text: str) -> str:
    """
    Normalize raw text.

    - lowercase
    - remove punctuation
    - remove extra whitespaces
    """

    if not text:
        return ""

    text = text.lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text: str) -> list[str]:
    """
    Split normalized text into tokens.
    """

    if not text:
        return []

    return text.split()


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Remove common stopwords.
    """

    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


def generate_shingles(
    tokens: list[str],
    k: int = 3
) -> set[str]:
    """
    Generate word shingles.

    Example
    -------
    tokens = ["this","is","a","test"]

    k = 2

    ->
    {
        "this is",
        "is a",
        "a test"
    }
    """

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if len(tokens) < k:
        return set()

    shingles = set()

    for i in range(len(tokens) - k + 1):
        shingle = " ".join(tokens[i:i + k])
        shingles.add(shingle)

    return shingles


def preprocess_document(
    text: str,
    k: int = 3
) -> set[str]:
    """
    Complete preprocessing pipeline.
    """

    normalized = normalize_text(text)

    if not normalized:
        return set()

    tokens = tokenize(normalized)

    tokens = remove_stopwords(tokens)

    return generate_shingles(tokens, k)