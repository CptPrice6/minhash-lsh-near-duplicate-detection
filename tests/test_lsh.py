import numpy as np
import pytest

from near_dup.lsh import lsh_candidates, theoretical_lsh_probability


def test_matching_band_creates_candidate():
    signatures = np.array(
        [[1, 2, 3, 4], [1, 2, 8, 9], [7, 7, 3, 4]],
        dtype=np.uint64,
    )
    assert lsh_candidates(signatures, 2, 2) == {(0, 1), (0, 2)}


def test_unique_signatures_create_no_candidates():
    signatures = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.uint64)
    assert lsh_candidates(signatures, 1, 2) == set()


def test_invalid_signature_shape():
    with pytest.raises(ValueError):
        lsh_candidates(np.array([1, 2]), 1, 2)


def test_invalid_band_dimensions():
    with pytest.raises(ValueError):
        lsh_candidates(np.ones((2, 4)), 3, 2)


@pytest.mark.parametrize(("bands", "rows"), [(0, 2), (2, 0), (-1, 2)])
def test_positive_lsh_parameters(bands, rows):
    with pytest.raises(ValueError):
        lsh_candidates(np.ones((2, 4)), bands, rows)


def test_theoretical_probability_boundaries():
    assert theoretical_lsh_probability(0.0, 50, 2) == 0.0
    assert theoretical_lsh_probability(1.0, 50, 2) == 1.0


def test_theoretical_probability_formula():
    expected = 1 - (1 - 0.5**5) ** 20
    assert theoretical_lsh_probability(0.5, 20, 5) == pytest.approx(expected)


@pytest.mark.parametrize("similarity", [-0.1, 1.1])
def test_theoretical_probability_validates_similarity(similarity):
    with pytest.raises(ValueError):
        theoretical_lsh_probability(similarity, 20, 5)
