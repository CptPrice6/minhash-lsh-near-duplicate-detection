import pytest

from sklearn.utils import Bunch
from near_dup.preprocessing import get_filtered_documents, prepare_shingled_documents


from near_dup.preprocessing import (
    create_shingle_sets,
    create_word_shingles,
    normalize_text,
    tokenize_words,
)


def test_normalize_text():
    text = "  Hello,\n\nWORLD!  "

    assert normalize_text(text) == "hello, world!"


def test_tokenize_words():
    text = "Hello, world! It's 2026."

    assert tokenize_words(text) == [
        "hello",
        "world",
        "it's",
        "2026",
    ]


def test_create_word_shingles():
    text = "one two three four"

    shingles = create_word_shingles(text, k=2)

    assert shingles == {
        "one two",
        "two three",
        "three four",
    }


def test_duplicate_shingles_are_removed():
    text = "one two one two"

    shingles = create_word_shingles(text, k=2)

    assert shingles == {
        "one two",
        "two one",
    }


def test_document_shorter_than_k():
    shingles = create_word_shingles(
        "one two",
        k=5,
    )

    assert shingles == {"one two"}


def test_empty_document():
    assert create_word_shingles("", k=3) == set()


def test_invalid_k():
    with pytest.raises(ValueError):
        create_word_shingles("one two three", k=0)


def test_create_shingle_sets():
    documents = [
        "one two three",
        "four five six",
    ]

    shingle_sets = create_shingle_sets(
        documents=documents,
        k=2,
    )

    assert shingle_sets == [
        {"one two", "two three"},
        {"four five", "five six"},
    ]


def test_get_filtered_documents_filters_short_documents():
    dataset = Bunch(
        data=[
            "short",
            "one two three four five six",
            "seven eight nine ten eleven twelve",
        ]
    )

    documents = get_filtered_documents(
        dataset=dataset,
        k=3,
        sample_size=2,
        min_shingles=3,
    )

    assert documents == [
        "one two three four five six",
        "seven eight nine ten eleven twelve",
    ]


def test_get_filtered_documents_respects_sample_size():
    dataset = Bunch(
        data=[
            "one two three four five",
            "six seven eight nine ten",
            "eleven twelve thirteen fourteen fifteen",
        ]
    )

    documents = get_filtered_documents(
        dataset=dataset,
        k=2,
        sample_size=2,
        min_shingles=2,
    )

    assert len(documents) == 2


@pytest.mark.parametrize(
    ("k", "sample_size", "min_shingles"),
    [
        (0, 2, 1),
        (2, 0, 1),
        (2, 2, 0),
    ],
)
def test_prepare_shingled_documents_validates_parameters(
    k,
    sample_size,
    min_shingles,
):
    dataset = Bunch(data=["one two three"])

    with pytest.raises(ValueError):
        prepare_shingled_documents(
            dataset=dataset,
            k=k,
            sample_size=sample_size,
            min_shingles=min_shingles,
        )


def test_prepare_shingled_documents_returns_matching_results():
    dataset = Bunch(
        data=[
            "too short",
            "one two three four five",
            "six seven eight nine ten",
        ]
    )

    documents, shingle_sets = prepare_shingled_documents(
        dataset=dataset,
        k=2,
        sample_size=2,
        min_shingles=3,
    )

    assert len(documents) == 2
    assert len(shingle_sets) == 2
    assert all(len(shingles) >= 3 for shingles in shingle_sets)
