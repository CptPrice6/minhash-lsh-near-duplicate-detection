import numpy as np
import pytest

from near_dup.lsh import (
    lsh_candidates,
    theoretical_lsh_probability,
)


def test_identical_band_creates_candidate_pair():
    signatures = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, 8, 9],
            [7, 7, 3, 4],
        ],
        dtype=np.uint64,
    )

    candidates = lsh_candidates(
        signatures=signatures,
        num_bands=2,
        rows_per_band=2,
    )

    assert (0, 1) in candidates
    assert (0, 2) in candidates
    assert (1, 2) not in candidates


def test_identical_signatures_are_candidates():
    signatures = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ],
        dtype=np.uint64,
    )

    candidates = lsh_candidates(
        signatures=signatures,
        num_bands=2,
        rows_per_band=2,
    )

    assert candidates == {(0, 1)}


def test_unique_signatures_produce_no_candidates():
    signatures = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ],
        dtype=np.uint64,
    )

    candidates = lsh_candidates(
        signatures=signatures,
        num_bands=2,
        rows_per_band=2,
    )

    assert candidates == set()


def test_theoretical_probability_boundaries():
    assert (
        theoretical_lsh_probability(
            similarity=0.0,
            num_bands=50,
            rows_per_band=2,
        )
        == 0.0
    )

    assert (
        theoretical_lsh_probability(
            similarity=1.0,
            num_bands=50,
            rows_per_band=2,
        )
        == 1.0
    )


def test_theoretical_probability_matches_formula():
    similarity = 0.5
    num_bands = 20
    rows_per_band = 5

    expected = 1 - (1 - similarity**rows_per_band) ** num_bands

    actual = theoretical_lsh_probability(
        similarity=similarity,
        num_bands=num_bands,
        rows_per_band=rows_per_band,
    )

    assert actual == pytest.approx(expected)


def test_invalid_lsh_dimensions_raise_error():
    signatures = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ],
        dtype=np.uint64,
    )

    with pytest.raises(ValueError):
        lsh_candidates(
            signatures=signatures,
            num_bands=3,
            rows_per_band=2,
        )


@pytest.mark.parametrize(
    ("num_bands", "rows_per_band"),
    [
        (0, 2),
        (-1, 2),
        (2, 0),
        (2, -1),
    ],
)
def test_nonpositive_lsh_parameters_raise_error(
    num_bands,
    rows_per_band,
):
    signatures = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ],
        dtype=np.uint64,
    )

    with pytest.raises(ValueError):
        lsh_candidates(
            signatures=signatures,
            num_bands=num_bands,
            rows_per_band=rows_per_band,
        )


def test_non_matrix_signatures_raise_error():
    signatures = np.array([1, 2, 3, 4], dtype=np.uint64)

    with pytest.raises(ValueError):
        lsh_candidates(
            signatures=signatures,
            num_bands=2,
            rows_per_band=2,
        )


@pytest.mark.parametrize("similarity", [-0.1, 1.1])
def test_invalid_theoretical_similarity(similarity):
    with pytest.raises(ValueError):
        theoretical_lsh_probability(
            similarity=similarity,
            num_bands=10,
            rows_per_band=2,
        )


@pytest.mark.parametrize(
    ("num_bands", "rows_per_band"),
    [
        (0, 2),
        (10, 0),
    ],
)
def test_invalid_theoretical_lsh_parameters(
    num_bands,
    rows_per_band,
):
    with pytest.raises(ValueError):
        theoretical_lsh_probability(
            similarity=0.5,
            num_bands=num_bands,
            rows_per_band=rows_per_band,
        )
