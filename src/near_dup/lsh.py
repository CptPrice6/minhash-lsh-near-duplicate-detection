from collections import defaultdict
from itertools import combinations
import hashlib
import numpy as np


def hash_band(band: np.ndarray) -> str:
    band_bytes = band.tobytes()
    return hashlib.md5(band_bytes).hexdigest()


def lsh_candidates(
    signatures: np.ndarray,
    num_bands: int,
    rows_per_band: int,
) -> set[tuple[int, int]]:
    num_documents, num_hashes = signatures.shape

    if num_bands * rows_per_band != num_hashes:
        raise ValueError(
            "num_bands * rows_per_band must equal the number of hash functions"
        )

    candidates = set()

    for band_index in range(num_bands):
        buckets = defaultdict(list)

        start = band_index * rows_per_band
        end = start + rows_per_band

        for doc_index in range(num_documents):
            band = signatures[doc_index, start:end]
            band_hash = hash_band(band)
            buckets[band_hash].append(doc_index)

        for bucket_documents in buckets.values():
            if len(bucket_documents) > 1:
                for i, j in combinations(bucket_documents, 2):
                    candidates.add((i, j))

    return candidates


def theoretical_lsh_probability(
    similarity: float, num_bands: int, rows_per_band: int
) -> float:
    return 1 - (1 - similarity**rows_per_band) ** num_bands
