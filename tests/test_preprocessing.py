import pytest
from sklearn.utils import Bunch

from near_dup.preprocessing import (
    create_shingle_sets,
    create_word_shingles,
    normalize_text,
    prepare_documents,
    select_document_indices,
    tokenize_words,
)


def test_normalize_text():
    assert normalize_text("  Hello,\nWORLD!  ") == "hello, world!"


def test_tokenize_words():
    assert tokenize_words("Hello, world! It's 2026.") == [
        "hello",
        "world",
        "it's",
        "2026",
    ]


def test_create_word_shingles():
    assert create_word_shingles("one two three four", 2) == {
        "one two",
        "two three",
        "three four",
    }


def test_shingles_are_unique():
    assert create_word_shingles("one two one two", 2) == {
        "one two",
        "two one",
    }


def test_short_document_has_no_k_shingles():
    assert create_word_shingles("one two", 3) == set()


def test_invalid_k():
    with pytest.raises(ValueError):
        create_word_shingles("one two", 0)


def test_create_shingle_sets():
    assert create_shingle_sets(["one two three", "four five six"], 2) == [
        {"one two", "two three"},
        {"four five", "five six"},
    ]


def sampling_dataset() -> Bunch:
    return Bunch(
        data=[f"document {index} alpha beta gamma delta epsilon" for index in range(20)]
    )


def test_sampling_is_deterministic():
    first = select_document_indices(sampling_dataset(), 2, 5, 2, 42)
    second = select_document_indices(sampling_dataset(), 2, 5, 2, 42)
    assert first == second


def test_different_seeds_change_sample():
    first = select_document_indices(sampling_dataset(), 2, 5, 2, 11)
    second = select_document_indices(sampling_dataset(), 2, 5, 2, 73)
    assert first != second


def test_sampling_uses_only_eligible_documents():
    dataset = Bunch(
        data=[
            "too short",
            "one two three four five",
            "six seven eight nine ten",
        ]
    )
    selected = select_document_indices(dataset, 3, 2, 3, 42)
    assert set(selected) == {1, 2}


@pytest.mark.parametrize(
    ("filter_k", "sample_size", "min_shingles"),
    [(0, 1, 1), (2, 0, 1), (2, 1, 0)],
)
def test_sampling_validates_positive_values(filter_k, sample_size, min_shingles):
    with pytest.raises(ValueError):
        select_document_indices(
            sampling_dataset(),
            filter_k,
            sample_size,
            min_shingles,
            42,
        )


def test_sampling_requires_enough_documents():
    with pytest.raises(RuntimeError):
        select_document_indices(Bunch(data=["one two three"]), 2, 2, 1, 42)


def test_prepare_documents_returns_matching_sets():
    documents, shingle_sets = prepare_documents(
        Bunch(data=["too short", "one two three four", "five six seven eight"]),
        k=2,
        sample_size=2,
        min_shingles=2,
        sample_seed=42,
    )
    assert len(documents) == len(shingle_sets) == 2
    assert all(len(shingles) >= 2 for shingles in shingle_sets)
