import hashlib
import random
import numpy as np

LARGE_PRIME = 4_294_967_291


def stable_hash(value: str) -> int:
    encoded = value.encode("utf-8")
    digest = hashlib.md5(encoded).hexdigest()
    return int(digest, 16) % LARGE_PRIME


def generate_hash_functions(num_hashes: int, seed: int = 42) -> list[tuple[int, int]]:
    random.seed(seed)

    hash_functions = []

    for _ in range(num_hashes):
        a = random.randint(1, LARGE_PRIME - 1)
        b = random.randint(0, LARGE_PRIME - 1)
        hash_functions.append((a, b))

    return hash_functions


def compute_minhash_signature(
    shingles: set[str],
    hash_functions: list[tuple[int, int]],
) -> np.ndarray:
    if len(shingles) == 0:
        return np.full(len(hash_functions), LARGE_PRIME, dtype=np.uint64)

    shingle_hashes = [stable_hash(shingle) for shingle in shingles]

    signature = []

    for a, b in hash_functions:
        min_hash_value = min((a * x + b) % LARGE_PRIME for x in shingle_hashes)
        signature.append(min_hash_value)

    return np.array(signature, dtype=np.uint64)


def compute_signature_matrix(
    shingle_sets: list[set[str]],
    num_hashes: int,
    seed: int = 42,
) -> np.ndarray:
    hash_functions = generate_hash_functions(num_hashes=num_hashes, seed=seed)

    signatures = [
        compute_minhash_signature(shingles, hash_functions) for shingles in shingle_sets
    ]

    return np.vstack(signatures)


def estimate_jaccard_from_signatures(
    signature_a: np.ndarray,
    signature_b: np.ndarray,
) -> float:
    if len(signature_a) != len(signature_b):
        raise ValueError("Signatures must have the same length.")

    return float(np.mean(signature_a == signature_b))
