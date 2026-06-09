import pytest

from near_dup.sampling import sample_pairs_below_threshold


def test_samples_requested_number():
    sampled = sample_pairs_below_threshold(
        [{"a", "b"}, {"a", "c"}, {"x"}, {"y"}],
        excluded_pairs=set(),
        number_of_pairs=2,
        similarity_threshold=0.2,
        seed=42,
    )
    assert len(sampled) == 2
    assert all(value < 0.2 for value in sampled.values())


def test_excludes_requested_pairs():
    sampled = sample_pairs_below_threshold(
        [{"a"}, {"b"}, {"c"}],
        excluded_pairs={(0, 1)},
        number_of_pairs=2,
        similarity_threshold=0.5,
        seed=42,
    )
    assert (0, 1) not in sampled


def test_validates_pair_count():
    with pytest.raises(ValueError):
        sample_pairs_below_threshold([{"a"}, {"b"}], set(), 0, 0.5)


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_validates_threshold(threshold):
    with pytest.raises(ValueError):
        sample_pairs_below_threshold([{"a"}, {"b"}], set(), 1, threshold)


def test_requires_two_documents():
    with pytest.raises(ValueError):
        sample_pairs_below_threshold([{"a"}], set(), 1, 0.5)


def test_impossible_sample_raises_error():
    with pytest.raises(RuntimeError):
        sample_pairs_below_threshold(
            [{"a", "b"}, {"a", "b"}],
            set(),
            1,
            0.5,
            seed=42,
        )
