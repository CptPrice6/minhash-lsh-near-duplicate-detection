import pytest

from near_dup.simhash import (
    compute_simhash,
    cosine_similarity,
    estimate_cosine_from_simhash,
    hamming_distance,
    term_frequencies,
)


def test_term_frequencies():
    frequencies = term_frequencies("Apple banana apple.")

    assert frequencies["apple"] == 2
    assert frequencies["banana"] == 1


def test_identical_vectors_have_cosine_one():
    frequencies = {
        "apple": 2,
        "banana": 1,
    }

    similarity = cosine_similarity(
        frequencies,
        frequencies,
    )

    assert similarity == pytest.approx(1.0)


def test_disjoint_vectors_have_cosine_zero():
    similarity = cosine_similarity(
        {"apple": 2},
        {"banana": 3},
    )

    assert similarity == 0.0


def test_empty_vector_has_cosine_zero():
    assert cosine_similarity({}, {"apple": 1}) == 0.0


def test_simhash_is_deterministic():
    frequencies = {
        "apple": 2,
        "banana": 1,
    }

    first = compute_simhash(
        frequencies=frequencies,
        num_bits=64,
        seed=42,
    )

    second = compute_simhash(
        frequencies=frequencies,
        num_bits=64,
        seed=42,
    )

    assert first == second


def test_identical_fingerprints_have_zero_distance():
    fingerprint = compute_simhash(
        frequencies={"apple": 2, "banana": 1},
        num_bits=64,
        seed=42,
    )

    assert hamming_distance(fingerprint, fingerprint) == 0


def test_identical_fingerprints_estimate_cosine_one():
    fingerprint = compute_simhash(
        frequencies={"apple": 2, "banana": 1},
        num_bits=64,
        seed=42,
    )

    estimate = estimate_cosine_from_simhash(
        fingerprint_a=fingerprint,
        fingerprint_b=fingerprint,
        num_bits=64,
    )

    assert estimate == pytest.approx(1.0)


@pytest.mark.parametrize(
    "invalid_num_bits",
    [0, -8, 7, 520],
)
def test_invalid_simhash_bit_size(invalid_num_bits):
    with pytest.raises(ValueError):
        compute_simhash(
            frequencies={"apple": 1},
            num_bits=invalid_num_bits,
            seed=42,
        )
