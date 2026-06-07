import numpy as np
import pytest

from near_dup.minhash import (
    LARGE_PRIME,
    compute_minhash_signature,
    compute_signature_matrix,
    estimate_jaccard_from_signatures,
    generate_hash_functions,
    stable_hash,
)


def test_stable_hash_is_deterministic():
    assert stable_hash("example shingle") == stable_hash("example shingle")


def test_stable_hash_is_within_expected_range():
    value = stable_hash("example shingle")

    assert 0 <= value < LARGE_PRIME


@pytest.mark.parametrize(
    "hash_family",
    ["linear", "murmur", "tabulation"],
)
def test_signature_matrix_shape(hash_family):
    shingle_sets = [
        {"a", "b", "c"},
        {"a", "b", "d"},
        {"x", "y", "z"},
    ]

    signatures = compute_signature_matrix(
        shingle_sets=shingle_sets,
        num_hashes=20,
        seed=42,
        hash_family=hash_family,
    )

    assert signatures.shape == (3, 20)
    assert signatures.dtype == np.uint64


@pytest.mark.parametrize(
    "hash_family",
    ["linear", "murmur", "tabulation"],
)
def test_identical_sets_have_identical_signatures(hash_family):
    shingle_sets = [
        {"a", "b", "c"},
        {"a", "b", "c"},
    ]

    signatures = compute_signature_matrix(
        shingle_sets=shingle_sets,
        num_hashes=20,
        seed=42,
        hash_family=hash_family,
    )

    assert np.array_equal(signatures[0], signatures[1])
    assert (
        estimate_jaccard_from_signatures(
            signatures[0],
            signatures[1],
        )
        == 1.0
    )


@pytest.mark.parametrize(
    "hash_family",
    ["linear", "murmur", "tabulation"],
)
def test_same_seed_produces_same_signatures(hash_family):
    shingle_sets = [
        {"one", "two", "three"},
        {"one", "two", "four"},
    ]

    first = compute_signature_matrix(
        shingle_sets=shingle_sets,
        num_hashes=20,
        seed=42,
        hash_family=hash_family,
    )

    second = compute_signature_matrix(
        shingle_sets=shingle_sets,
        num_hashes=20,
        seed=42,
        hash_family=hash_family,
    )

    assert np.array_equal(first, second)


def test_empty_shingle_set_signature():
    hash_functions = generate_hash_functions(
        num_hashes=10,
        seed=42,
        hash_family="linear",
    )

    signature = compute_minhash_signature(
        shingles=set(),
        hash_functions=hash_functions,
        hash_family="linear",
    )

    assert len(signature) == 10
    assert np.all(signature == LARGE_PRIME)


def test_invalid_number_of_hashes():
    with pytest.raises(ValueError):
        generate_hash_functions(
            num_hashes=0,
            seed=42,
            hash_family="linear",
        )


def test_unsupported_hash_family():
    with pytest.raises(ValueError):
        generate_hash_functions(
            num_hashes=10,
            seed=42,
            hash_family="unknown",
        )


def test_different_signature_lengths_raise_error():
    signature_a = np.array([1, 2, 3], dtype=np.uint64)
    signature_b = np.array([1, 2], dtype=np.uint64)

    with pytest.raises(ValueError):
        estimate_jaccard_from_signatures(
            signature_a,
            signature_b,
        )


def test_empty_signature_matrix_input():
    with pytest.raises(ValueError):
        compute_signature_matrix(
            shingle_sets=[],
            num_hashes=10,
        )


def test_empty_signatures_raise_error():
    with pytest.raises(ValueError):
        estimate_jaccard_from_signatures(
            np.array([], dtype=np.uint64),
            np.array([], dtype=np.uint64),
        )
