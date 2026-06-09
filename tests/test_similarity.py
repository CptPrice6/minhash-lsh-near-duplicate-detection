import pytest

from near_dup.similarity import compute_ground_truth, jaccard_similarity


def test_identical_sets():
    assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0


def test_disjoint_sets():
    assert jaccard_similarity({"a"}, {"b"}) == 0.0


def test_partial_overlap():
    assert jaccard_similarity({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def test_two_empty_sets_are_zero():
    assert jaccard_similarity(set(), set()) == 0.0


def test_ground_truth_uses_inclusive_threshold():
    result = compute_ground_truth([{"a", "b"}, {"a", "c"}], 1 / 3)
    assert result == {(0, 1): pytest.approx(1 / 3)}


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_ground_truth_validates_threshold(threshold):
    with pytest.raises(ValueError):
        compute_ground_truth([], threshold)
