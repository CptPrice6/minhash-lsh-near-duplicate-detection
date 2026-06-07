import pytest

from near_dup.similarity import (
    compute_ground_truth,
    jaccard_similarity,
)


def test_identical_sets_have_similarity_one():
    shingles = {
        "one two",
        "two three",
    }

    assert jaccard_similarity(shingles, shingles) == 1.0


def test_disjoint_sets_have_similarity_zero():
    first = {"one two"}
    second = {"three four"}

    assert jaccard_similarity(first, second) == 0.0


def test_partial_overlap():
    first = {
        "one two",
        "two three",
    }

    second = {
        "two three",
        "three four",
    }

    similarity = jaccard_similarity(first, second)

    assert similarity == 1 / 3


def test_ground_truth_contains_pairs_above_threshold():
    shingle_sets = [
        {"a", "b", "c"},
        {"a", "b", "d"},
        {"x", "y", "z"},
    ]

    ground_truth = compute_ground_truth(
        shingle_sets=shingle_sets,
        threshold=0.4,
    )

    assert set(ground_truth.keys()) == {(0, 1)}
    assert ground_truth[(0, 1)] == 0.5


def test_ground_truth_uses_inclusive_threshold():
    shingle_sets = [
        {"a", "b"},
        {"a", "c"},
    ]

    ground_truth = compute_ground_truth(
        shingle_sets=shingle_sets,
        threshold=1 / 3,
    )

    assert (0, 1) in ground_truth


def test_both_empty_sets_have_zero_jaccard():
    assert jaccard_similarity(set(), set()) == 0.0


def test_invalid_ground_truth_threshold():
    with pytest.raises(ValueError):
        compute_ground_truth([], threshold=-0.1)
