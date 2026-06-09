import pytest
import pandas as pd

from near_dup.simhash import (
    compute_simhash,
    cosine_similarity,
    estimate_cosine_from_simhash,
    hamming_distance,
    term_frequencies,
    calculate_simhash_summary,
)


def test_term_frequencies():
    frequencies = term_frequencies("Apple banana apple")
    assert frequencies["apple"] == 2
    assert frequencies["banana"] == 1


def test_cosine_similarity():
    vector = {"apple": 2, "banana": 1}
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)
    assert cosine_similarity({"apple": 1}, {"banana": 1}) == 0.0
    assert cosine_similarity({}, vector) == 0.0


def test_simhash_is_deterministic():
    vector = {"apple": 2, "banana": 1}
    assert compute_simhash(vector, 64, 42) == compute_simhash(vector, 64, 42)


def test_identical_fingerprints():
    fingerprint = compute_simhash({"apple": 2}, 64, 42)
    assert hamming_distance(fingerprint, fingerprint) == 0
    assert estimate_cosine_from_simhash(fingerprint, fingerprint, 64) == pytest.approx(
        1.0
    )


@pytest.mark.parametrize("num_bits", [0, -8, 7, 520])
def test_invalid_bit_size(num_bits):
    with pytest.raises(ValueError):
        compute_simhash({"apple": 1}, num_bits, 42)


def test_negative_weights_are_rejected():
    with pytest.raises(ValueError):
        compute_simhash({"apple": -1}, 64, 42)


def test_simhash_summary_groups_by_bit_size():

    results = pd.DataFrame(
        {
            "pair_type": ["similar", "similar", "dissimilar", "dissimilar"],
            "num_bits": [64, 128, 64, 128],
            "exact_cosine": [0.8, 0.8, 0.0, 0.0],
            "estimated_cosine": [0.7, 0.78, 0.1, 0.05],
        }
    )
    runtimes = pd.DataFrame(
        {
            "num_bits": [64, 128],
            "fingerprint_time_seconds": [0.1, 0.2],
        }
    )

    summary = calculate_simhash_summary(results, runtimes)
    assert set(summary["num_bits"]) == {64, 128}
    assert set(summary["pair_type"]) == {"all", "similar", "dissimilar"}
