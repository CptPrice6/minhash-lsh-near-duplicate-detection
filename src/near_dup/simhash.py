import hashlib
import math
from collections import Counter
from collections.abc import Mapping

from near_dup.preprocessing import tokenize_words


def term_frequencies(text: str) -> Counter[str]:
    return Counter(tokenize_words(text))


def cosine_similarity(
    frequencies_a: Mapping[str, int],
    frequencies_b: Mapping[str, int],
) -> float:
    if not frequencies_a or not frequencies_b:
        return 0.0

    shared_terms = frequencies_a.keys() & frequencies_b.keys()

    dot_product = sum(
        frequencies_a[term] * frequencies_b[term] for term in shared_terms
    )

    norm_a = math.sqrt(sum(value * value for value in frequencies_a.values()))
    norm_b = math.sqrt(sum(value * value for value in frequencies_b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def stable_feature_hash(
    feature: str,
    num_bits: int,
    seed: int,
) -> int:
    if num_bits <= 0 or num_bits > 512 or num_bits % 8 != 0:
        raise ValueError("num_bits must be a positive multiple of 8 and at most 512")

    digest_size = num_bits // 8
    encoded = f"{seed}:{feature}".encode("utf-8")

    digest = hashlib.blake2b(
        encoded,
        digest_size=digest_size,
    ).digest()

    return int.from_bytes(digest, byteorder="big")


def compute_simhash(
    frequencies: Mapping[str, int],
    num_bits: int = 128,
    seed: int = 42,
) -> int:
    if num_bits <= 0 or num_bits > 512 or num_bits % 8 != 0:
        raise ValueError("num_bits must be a positive multiple of 8 and at most 512")

    if not frequencies:
        return 0

    bit_scores = [0.0] * num_bits

    for feature, weight in frequencies.items():
        feature_hash = stable_feature_hash(
            feature=feature,
            num_bits=num_bits,
            seed=seed,
        )

        for bit_index in range(num_bits):
            if feature_hash & (1 << bit_index):
                bit_scores[bit_index] += weight
            else:
                bit_scores[bit_index] -= weight

    fingerprint = 0

    for bit_index, score in enumerate(bit_scores):
        if score >= 0:
            fingerprint |= 1 << bit_index

    return fingerprint


def hamming_distance(
    fingerprint_a: int,
    fingerprint_b: int,
) -> int:
    """
    Count differing bits between two SimHash fingerprints.
    """
    return (fingerprint_a ^ fingerprint_b).bit_count()


def estimate_cosine_from_simhash(
    fingerprint_a: int,
    fingerprint_b: int,
    num_bits: int,
) -> float:
    if num_bits <= 0:
        raise ValueError("num_bits must be greater than 0")

    distance = hamming_distance(
        fingerprint_a,
        fingerprint_b,
    )

    estimated_angle = math.pi * distance / num_bits
    estimate = math.cos(estimated_angle)

    return max(-1.0, min(1.0, estimate))
