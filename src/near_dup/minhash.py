import hashlib
import random
import numpy as np
import mmh3

LARGE_PRIME = 4_294_967_291

SUPPORTED_HASH_FAMILIES = {
    "linear",
    "murmur",
    "tabulation",
}


def stable_hash(value: str) -> int:
    encoded = value.encode("utf-8")
    digest = hashlib.md5(encoded).hexdigest()
    return int(digest, 16) % LARGE_PRIME


def generate_hash_functions(
    num_hashes: int, seed: int = 42, hash_family: str = "linear"
):
    if hash_family not in SUPPORTED_HASH_FAMILIES:
        raise ValueError(f"Unsupported hash family: {hash_family}")
    if num_hashes <= 0:
        raise ValueError("num_hashes must be greater than 0")

    rng = random.Random(seed)

    if hash_family == "linear":
        return [
            (rng.randint(1, LARGE_PRIME - 1), rng.randint(0, LARGE_PRIME - 1))
            for _ in range(num_hashes)
        ]

    if hash_family == "murmur":
        return [rng.randint(0, 2**32 - 1) for _ in range(num_hashes)]

    if hash_family == "tabulation":
        np_rng = np.random.default_rng(seed)
        return np_rng.integers(
            0, LARGE_PRIME, size=(num_hashes, 4, 256), dtype=np.int64
        )


def _murmur_hash(value: str, seed: int) -> int:
    return mmh3.hash(value.encode("utf-8"), seed, signed=False) % LARGE_PRIME


def _tabulation_hash(value: int, table: np.ndarray) -> int:
    h = 0
    for byte_pos in range(4):
        h ^= int(table[byte_pos, (value >> (8 * byte_pos)) & 0xFF])
    return h % LARGE_PRIME


def compute_minhash_signature(
    shingles: set[str],
    hash_functions,
    hash_family: str = "linear",
) -> np.ndarray:
    if hash_family not in SUPPORTED_HASH_FAMILIES:
        raise ValueError(f"Unsupported hash family: {hash_family}")

    num_hashes = (
        len(hash_functions) if hash_family != "tabulation" else hash_functions.shape[0]
    )

    if len(shingles) == 0:
        return np.full(num_hashes, LARGE_PRIME, dtype=np.uint64)

    signature = []

    if hash_family == "linear":
        shingle_hashes = [stable_hash(s) for s in shingles]
        for a, b in hash_functions:
            signature.append(min((a * x + b) % LARGE_PRIME for x in shingle_hashes))

    elif hash_family == "murmur":
        for seed in hash_functions:
            signature.append(min(_murmur_hash(s, seed) for s in shingles))

    elif hash_family == "tabulation":
        shingle_hashes = [stable_hash(s) for s in shingles]
        for table in hash_functions:
            signature.append(min(_tabulation_hash(x, table) for x in shingle_hashes))

    return np.array(signature, dtype=np.uint64)


def compute_signature_matrix(
    shingle_sets: list[set[str]],
    num_hashes: int,
    seed: int = 42,
    hash_family: str = "linear",
) -> np.ndarray:
    hash_functions = generate_hash_functions(num_hashes, seed, hash_family)
    signatures = [
        compute_minhash_signature(shingles, hash_functions, hash_family)
        for shingles in shingle_sets
    ]
    return np.vstack(signatures)


def estimate_jaccard_from_signatures(
    signature_a: np.ndarray,
    signature_b: np.ndarray,
) -> float:
    if len(signature_a) != len(signature_b):
        raise ValueError("Signatures must have the same length.")
    return float(np.mean(signature_a == signature_b))
