import numpy as np
import pytest

from near_dup.minhash import compute_signature_matrix, estimate_jaccard_from_signatures


@pytest.mark.parametrize("hash_family", ["linear", "murmur", "tabulation"])
def test_signature_matrix_shape(hash_family):
    signatures = compute_signature_matrix(
        [{"a", "b"}, {"a", "c"}, {"x", "y"}],
        num_hashes=20,
        seed=42,
        hash_family=hash_family,
    )
    assert signatures.shape == (3, 20)
    assert signatures.dtype == np.uint64


@pytest.mark.parametrize("hash_family", ["linear", "murmur", "tabulation"])
def test_identical_sets_have_identical_signatures(hash_family):
    signatures = compute_signature_matrix(
        [{"a", "b", "c"}, {"a", "b", "c"}],
        20,
        seed=42,
        hash_family=hash_family,
    )
    assert np.array_equal(signatures[0], signatures[1])


@pytest.mark.parametrize("hash_family", ["linear", "murmur", "tabulation"])
def test_same_seed_is_deterministic(hash_family):
    sets = [{"a", "b"}, {"a", "c"}]
    first = compute_signature_matrix(sets, 20, 42, hash_family)
    second = compute_signature_matrix(sets, 20, 42, hash_family)
    assert np.array_equal(first, second)


def test_empty_shingle_set_has_valid_signature():
    signature = compute_signature_matrix([set()], 10)[0]
    assert len(signature) == 10


def test_invalid_number_of_hashes():
    with pytest.raises(ValueError):
        compute_signature_matrix([{"a"}], 0)


def test_invalid_hash_family():
    with pytest.raises(ValueError):
        compute_signature_matrix([{"a"}], 10, hash_family="unknown")


def test_empty_input():
    with pytest.raises(ValueError):
        compute_signature_matrix([], 10)


def test_signature_estimate():
    first = np.array([1, 2, 3, 4], dtype=np.uint64)
    second = np.array([1, 9, 3, 8], dtype=np.uint64)
    assert estimate_jaccard_from_signatures(first, second) == 0.5


def test_signature_lengths_must_match():
    with pytest.raises(ValueError):
        estimate_jaccard_from_signatures(np.array([1]), np.array([1, 2]))


def test_signatures_must_not_be_empty():
    with pytest.raises(ValueError):
        estimate_jaccard_from_signatures(np.array([]), np.array([]))
