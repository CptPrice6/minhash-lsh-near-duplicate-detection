from collections import defaultdict
from itertools import combinations

import numpy as np


def lsh_candidates(
    signatures: np.ndarray,
    num_bands: int,
    rows_per_band: int,
) -> set[tuple[int, int]]:
    if signatures.ndim != 2:
        raise ValueError("signatures must be a two-dimensional matrix")
    if num_bands <= 0 or rows_per_band <= 0:
        raise ValueError("num_bands and rows_per_band must be greater than 0")

    num_documents, num_hashes = signatures.shape
    if num_bands * rows_per_band != num_hashes:
        raise ValueError("num_bands * rows_per_band must equal the signature length")

    candidates = set()

    for band_index in range(num_bands):
        start = band_index * rows_per_band
        end = start + rows_per_band
        buckets: dict[bytes, list[int]] = defaultdict(list)

        for document_index in range(num_documents):
            key = signatures[document_index, start:end].tobytes()
            buckets[key].append(document_index)

        for bucket in buckets.values():
            candidates.update(combinations(bucket, 2))

    return candidates


def theoretical_lsh_probability(
    similarity: float,
    num_bands: int,
    rows_per_band: int,
) -> float:
    if not 0.0 <= similarity <= 1.0:
        raise ValueError("similarity must be between 0 and 1")
    if num_bands <= 0 or rows_per_band <= 0:
        raise ValueError("num_bands and rows_per_band must be greater than 0")

    return 1.0 - (1.0 - similarity**rows_per_band) ** num_bands
