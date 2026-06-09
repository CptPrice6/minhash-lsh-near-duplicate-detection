import hashlib
import math
from collections import Counter
from collections.abc import Mapping

import numpy as np
import pandas as pd

from near_dup.preprocessing import tokenize_words


def _validate_num_bits(num_bits: int) -> None:
    if num_bits <= 0 or num_bits > 512 or num_bits % 8 != 0:
        raise ValueError("num_bits must be a positive multiple of 8 and at most 512")


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
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _stable_feature_hash(feature: str, num_bits: int, seed: int) -> int:
    _validate_num_bits(num_bits)
    digest = hashlib.blake2b(
        f"{seed}:{feature}".encode("utf-8"),
        digest_size=64,
    ).digest()
    return int.from_bytes(digest, "big") & ((1 << num_bits) - 1)


def compute_simhash(
    frequencies: Mapping[str, int],
    num_bits: int = 128,
    seed: int = 42,
) -> int:
    _validate_num_bits(num_bits)
    if any(weight < 0 for weight in frequencies.values()):
        raise ValueError("Feature weights must not be negative")
    if not frequencies or not any(frequencies.values()):
        return 0

    scores = [0] * num_bits

    for feature, weight in frequencies.items():
        if weight == 0:
            continue
        feature_hash = _stable_feature_hash(feature, num_bits, seed)
        for bit in range(num_bits):
            scores[bit] += weight if feature_hash & (1 << bit) else -weight

    fingerprint = 0
    for bit, score in enumerate(scores):
        if score >= 0:
            fingerprint |= 1 << bit

    return fingerprint


def hamming_distance(fingerprint_a: int, fingerprint_b: int) -> int:
    if fingerprint_a < 0 or fingerprint_b < 0:
        raise ValueError("Fingerprints must not be negative")
    return (fingerprint_a ^ fingerprint_b).bit_count()


def estimate_cosine_from_simhash(
    fingerprint_a: int,
    fingerprint_b: int,
    num_bits: int,
) -> float:
    _validate_num_bits(num_bits)
    if fingerprint_a >= 1 << num_bits or fingerprint_b >= 1 << num_bits:
        raise ValueError("Fingerprint exceeds the configured bit length")

    angle = math.pi * hamming_distance(fingerprint_a, fingerprint_b) / num_bits
    return math.cos(angle)


def calculate_simhash_summary(
    results_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
) -> pd.DataFrame:
    required = {"pair_type", "num_bits", "exact_cosine", "estimated_cosine"}
    missing = required - set(results_df.columns)
    if missing:
        raise ValueError(f"Results are missing columns: {sorted(missing)}")
    if results_df.empty:
        raise ValueError("Results must not be empty")

    rows = []
    groups = {
        "all": results_df,
        "similar": results_df[results_df["pair_type"] == "similar"],
        "dissimilar": results_df[results_df["pair_type"] == "dissimilar"],
    }

    for pair_type, group in groups.items():
        for num_bits, subset in group.groupby("num_bits"):
            errors = subset["estimated_cosine"] - subset["exact_cosine"]
            correlation = np.nan
            if (
                len(subset) > 1
                and subset["exact_cosine"].nunique() > 1
                and subset["estimated_cosine"].nunique() > 1
            ):
                correlation = subset["exact_cosine"].corr(subset["estimated_cosine"])

            rows.append(
                {
                    "pair_type": pair_type,
                    "num_bits": int(num_bits),
                    "number_of_pairs": len(subset),
                    "mean_absolute_error": errors.abs().mean(),
                    "root_mean_squared_error": np.sqrt(np.mean(errors**2)),
                    "maximum_absolute_error": errors.abs().max(),
                    "correlation": correlation,
                }
            )

    return pd.DataFrame(rows).merge(
        runtime_df,
        on="num_bits",
        how="left",
        validate="many_to_one",
    )
