import pytest

from near_dup.sampling import sample_pairs_below_threshold


def test_sample_pairs_below_threshold():
    shingle_sets = [
        {"a", "b"},
        {"a", "c"},
        {"x", "y"},
        {"m", "n"},
    ]

    sampled = sample_pairs_below_threshold(
        shingle_sets=shingle_sets,
        excluded_pairs=set(),
        number_of_pairs=2,
        similarity_threshold=0.2,
        seed=42,
    )

    assert len(sampled) == 2
    assert len(set(sampled)) == 2
    assert all(similarity < 0.2 for similarity in sampled.values())


def test_sampling_excludes_requested_pairs():
    shingle_sets = [
        {"a"},
        {"b"},
        {"c"},
    ]

    sampled = sample_pairs_below_threshold(
        shingle_sets=shingle_sets,
        excluded_pairs={(0, 1)},
        number_of_pairs=2,
        similarity_threshold=0.5,
        seed=42,
    )

    assert (0, 1) not in sampled


def test_invalid_number_of_pairs():
    with pytest.raises(ValueError):
        sample_pairs_below_threshold(
            shingle_sets=[{"a"}, {"b"}],
            excluded_pairs=set(),
            number_of_pairs=0,
            similarity_threshold=0.5,
        )


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_sampling_threshold(threshold):
    with pytest.raises(ValueError):
        sample_pairs_below_threshold(
            shingle_sets=[{"a"}, {"b"}],
            excluded_pairs=set(),
            number_of_pairs=1,
            similarity_threshold=threshold,
        )


def test_sampling_requires_two_documents():
    with pytest.raises(ValueError):
        sample_pairs_below_threshold(
            shingle_sets=[{"a"}],
            excluded_pairs=set(),
            number_of_pairs=1,
            similarity_threshold=0.5,
        )


def test_impossible_sampling_raises_error():
    with pytest.raises(RuntimeError):
        sample_pairs_below_threshold(
            shingle_sets=[
                {"a", "b"},
                {"a", "b"},
            ],
            excluded_pairs=set(),
            number_of_pairs=1,
            similarity_threshold=0.5,
            seed=42,
        )
