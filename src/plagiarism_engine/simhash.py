"""
TF-IDF weighted SimHash.

The SimHash algorithm is implemented directly for the project instead of using
an external near-duplicate library.
"""

from __future__ import annotations

import math
import hashlib
from collections import Counter
from typing import Iterable


def term_frequencies(tokens: Iterable[str]) -> Counter[str]:
    """
    Count token frequencies.
    """

    return Counter(token for token in tokens if token)


def inverse_document_frequencies(
    documents: Iterable[Iterable[str]],
) -> dict[str, float]:
    """
    Compute smoothed IDF weights for a corpus.
    """

    doc_sets = [set(token for token in document if token) for document in documents]
    total_docs = len(doc_sets)
    if total_docs == 0:
        return {}

    df: Counter[str] = Counter()
    for tokens in doc_sets:
        df.update(tokens)

    return {
        token: math.log((1 + total_docs) / (1 + count)) + 1.0
        for token, count in df.items()
    }


def _stable_hash64(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def simhash(
    tokens: Iterable[str],
    idf: dict[str, float] | None = None,
    bits: int = 64,
) -> int:
    """
    Compute a TF-IDF weighted SimHash integer.
    """

    if bits <= 0:
        raise ValueError("bits must be greater than zero.")
    if bits > 64:
        raise ValueError("This implementation supports up to 64 bits.")

    counts = term_frequencies(tokens)
    if not counts:
        return 0

    weights = [0.0] * bits
    total_terms = sum(counts.values())
    idf = idf or {}

    for token, count in counts.items():
        tf = count / total_terms
        weight = tf * idf.get(token, 1.0)
        token_hash = _stable_hash64(token)

        for bit_index in range(bits):
            bit_is_set = (token_hash >> bit_index) & 1
            if bit_is_set:
                weights[bit_index] += weight
            else:
                weights[bit_index] -= weight

    fingerprint = 0
    for bit_index, value in enumerate(weights):
        if value >= 0:
            fingerprint |= 1 << bit_index

    return fingerprint


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """
    Count differing bits between two integer fingerprints.
    """

    if hash_a < 0 or hash_b < 0:
        raise ValueError("SimHash fingerprints must be non-negative integers.")

    return (hash_a ^ hash_b).bit_count()


def simhash_similarity(hash_a: int, hash_b: int, bits: int = 64) -> float:
    """
    Convert Hamming distance to a similarity score in [0, 1].
    """

    if bits <= 0:
        raise ValueError("bits must be greater than zero.")

    distance = hamming_distance(hash_a, hash_b)
    return max(0.0, 1.0 - (distance / bits))
