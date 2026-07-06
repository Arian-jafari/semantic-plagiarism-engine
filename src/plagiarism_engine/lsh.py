"""
Locality-sensitive hashing for MinHash signatures.

Signatures are split into bands. Documents sharing at least one band bucket are
returned as candidate pairs for final exact or approximate comparison.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from itertools import combinations
from typing import Iterable, Mapping


Pair = tuple[str, str]


def _bucket_key(band_index: int, band: tuple[int, ...]) -> tuple[int, str]:
    raw = ",".join(str(value) for value in band)
    digest = hashlib.blake2b(raw.encode("utf-8"), digest_size=8).hexdigest()
    return band_index, digest


def lsh_candidates(
    signatures: Mapping[str, Iterable[int]],
    bands: int = 32,
) -> set[Pair]:
    """
    Return document-id pairs that collide in at least one LSH band.
    """

    if bands <= 0:
        raise ValueError("bands must be greater than zero.")

    normalized = {doc_id: tuple(signature) for doc_id, signature in signatures.items()}
    if not normalized:
        return set()

    lengths = {len(signature) for signature in normalized.values()}
    if len(lengths) != 1:
        raise ValueError("All signatures must have the same length.")

    signature_length = lengths.pop()
    if signature_length == 0:
        return set()
    if signature_length % bands != 0:
        raise ValueError("Signature length must be divisible by number of bands.")

    rows = signature_length // bands
    buckets: dict[tuple[int, str], list[str]] = defaultdict(list)

    for doc_id, signature in normalized.items():
        for band_index in range(bands):
            start = band_index * rows
            end = start + rows
            key = _bucket_key(band_index, signature[start:end])
            buckets[key].append(doc_id)

    candidates: set[Pair] = set()
    for doc_ids in buckets.values():
        if len(doc_ids) < 2:
            continue
        for doc_a, doc_b in combinations(sorted(doc_ids), 2):
            candidates.add((doc_a, doc_b))

    return candidates


def all_pairs(doc_ids: Iterable[str]) -> set[Pair]:
    """
    Return every unique pair for a collection of document ids.
    """

    return set(combinations(sorted(doc_ids), 2))
