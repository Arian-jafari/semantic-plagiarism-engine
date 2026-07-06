"""
MinHash signatures for estimating Jaccard similarity.

The implementation is intentionally self-contained for the project rules: no
ready-made MinHash package is used here.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable


MAX_HASH = (1 << 61) - 1
PRIME = (1 << 61) - 1


def stable_hash(value: str) -> int:
    """
    Return a deterministic integer hash for text.

    Python's built-in hash is salted per process, so it is not reproducible
    enough for saved outputs or tests.
    """

    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % PRIME


@dataclass(frozen=True)
class MinHasher:
    """
    Build fixed-length MinHash signatures.

    Parameters
    ----------
    num_hashes:
        Signature length. The guide suggests 128 or 256; tests can use smaller
        values for speed.
    seed:
        Seed for reproducible hash function coefficients.
    """

    num_hashes: int = 128
    seed: int = 42

    def __post_init__(self) -> None:
        if self.num_hashes <= 0:
            raise ValueError("num_hashes must be greater than zero.")

        rng = random.Random(self.seed)
        coefficients = []
        used = set()

        while len(coefficients) < self.num_hashes:
            a = rng.randint(1, PRIME - 1)
            b = rng.randint(0, PRIME - 1)
            pair = (a, b)
            if pair not in used:
                used.add(pair)
                coefficients.append(pair)

        object.__setattr__(self, "_coefficients", tuple(coefficients))

    def signature(self, shingles: Iterable[str]) -> tuple[int, ...]:
        """
        Compute a MinHash signature for a set or iterable of shingles.
        """

        unique_shingles = set(shingles)
        if not unique_shingles:
            return tuple([MAX_HASH] * self.num_hashes)

        shingle_hashes = [stable_hash(shingle) for shingle in unique_shingles]
        signature = []

        for a, b in self._coefficients:
            min_value = MAX_HASH
            for value in shingle_hashes:
                candidate = (a * value + b) % PRIME
                if candidate < min_value:
                    min_value = candidate
            signature.append(min_value)

        return tuple(signature)


def minhash_similarity(
    signature_a: Iterable[int],
    signature_b: Iterable[int],
) -> float:
    """
    Estimate Jaccard similarity from two MinHash signatures.
    """

    sig_a = tuple(signature_a)
    sig_b = tuple(signature_b)

    if len(sig_a) != len(sig_b):
        raise ValueError("Signatures must have the same length.")

    if not sig_a:
        return 1.0

    matches = sum(1 for left, right in zip(sig_a, sig_b) if left == right)
    return matches / len(sig_a)
