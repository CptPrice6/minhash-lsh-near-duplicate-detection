import hashlib
import random

import mmh3
import numpy as np

LARGE_PRIME = 4_294_967_291
SUPPORTED_HASH_FAMILIES = {"linear", "murmur", "tabulation"}


def _stable_hash(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % LARGE_PRIME


def _hash_parameters(num_hashes: int, seed: int, hash_family: str):
    if num_hashes <= 0:
        raise ValueError("num_hashes must be greater than 0")
    if hash_family not in SUPPORTED_HASH_FAMILIES:
        raise ValueError(f"Unsupported hash family: {hash_family}")

    rng = random.Random(seed)

    if hash_family == "linear":
        return [
            (
                rng.randrange(1, LARGE_PRIME),
                rng.randrange(LARGE_PRIME),
            )
            for _ in range(num_hashes)
        ]

    if hash_family == "murmur":
        return [rng.randrange(2**32) for _ in range(num_hashes)]

    np_rng = np.random.default_rng(seed)
    return np_rng.integers(
        0,
        LARGE_PRIME,
        size=(num_hashes, 4, 256),
        dtype=np.uint64,
    )


def _linear_signature(shingles: set[str], parameters) -> np.ndarray:
    hashes = np.array([_stable_hash(value) for value in shingles], dtype=np.uint64)
    signature = np.empty(len(parameters), dtype=np.uint64)

    for index, (a, b) in enumerate(parameters):
        signature[index] = np.min((a * hashes + b) % LARGE_PRIME)

    return signature


def _murmur_signature(shingles: set[str], seeds: list[int]) -> np.ndarray:
    return np.array(
        [
            min(
                mmh3.hash(value.encode("utf-8"), seed, signed=False) % LARGE_PRIME
                for value in shingles
            )
            for seed in seeds
        ],
        dtype=np.uint64,
    )


def _tabulation_signature(shingles: set[str], tables: np.ndarray) -> np.ndarray:
    values = np.array([_stable_hash(value) for value in shingles], dtype=np.uint64)
    signature = np.empty(len(tables), dtype=np.uint64)

    for index, table in enumerate(tables):
        hashed = np.zeros(len(values), dtype=np.uint64)
        for byte_position in range(4):
            byte_values = ((values >> (8 * byte_position)) & 0xFF).astype(np.intp)
            hashed ^= table[byte_position, byte_values]
        signature[index] = np.min(hashed % LARGE_PRIME)

    return signature


def _signature(shingles: set[str], parameters, hash_family: str) -> np.ndarray:
    num_hashes = len(parameters)
    if not shingles:
        return np.full(num_hashes, LARGE_PRIME, dtype=np.uint64)

    if hash_family == "linear":
        return _linear_signature(shingles, parameters)
    if hash_family == "murmur":
        return _murmur_signature(shingles, parameters)
    return _tabulation_signature(shingles, parameters)


def compute_signature_matrix(
    shingle_sets: list[set[str]],
    num_hashes: int,
    seed: int = 42,
    hash_family: str = "linear",
) -> np.ndarray:
    if not shingle_sets:
        raise ValueError("shingle_sets must not be empty")

    parameters = _hash_parameters(num_hashes, seed, hash_family)
    return np.vstack(
        [_signature(shingles, parameters, hash_family) for shingles in shingle_sets]
    )


def estimate_jaccard_from_signatures(
    signature_a: np.ndarray,
    signature_b: np.ndarray,
) -> float:
    if signature_a.ndim != 1 or signature_b.ndim != 1:
        raise ValueError("Signatures must be one-dimensional")
    if len(signature_a) == 0 or len(signature_b) == 0:
        raise ValueError("Signatures must not be empty")
    if len(signature_a) != len(signature_b):
        raise ValueError("Signatures must have the same length")

    return float(np.mean(signature_a == signature_b))
