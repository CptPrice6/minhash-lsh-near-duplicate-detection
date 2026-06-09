import re

import numpy as np


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", normalize_text(text))


def create_word_shingles(text: str, k: int = 5) -> set[str]:
    if k <= 0:
        raise ValueError("k must be greater than 0")

    tokens = tokenize_words(text)
    if len(tokens) < k:
        return set()

    return {" ".join(tokens[start : start + k]) for start in range(len(tokens) - k + 1)}


def create_shingle_sets(documents: list[str], k: int) -> list[set[str]]:
    return [create_word_shingles(document, k) for document in documents]


def select_document_indices(
    dataset,
    filter_k: int,
    sample_size: int,
    min_shingles: int,
    sample_seed: int = 42,
) -> list[int]:
    if filter_k <= 0:
        raise ValueError("filter_k must be greater than 0")
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than 0")
    if min_shingles <= 0:
        raise ValueError("min_shingles must be greater than 0")

    eligible = [
        index
        for index, text in enumerate(dataset.data)
        if len(create_word_shingles(text, filter_k)) >= min_shingles
    ]

    if len(eligible) < sample_size:
        raise RuntimeError(
            f"Only {len(eligible)} eligible documents were found; "
            f"{sample_size} were requested"
        )

    rng = np.random.default_rng(sample_seed)
    selected = rng.choice(eligible, size=sample_size, replace=False)
    return sorted(int(index) for index in selected)


def prepare_documents(
    dataset,
    k: int,
    sample_size: int,
    min_shingles: int,
    sample_seed: int = 42,
) -> tuple[list[str], list[set[str]]]:
    indices = select_document_indices(
        dataset=dataset,
        filter_k=k,
        sample_size=sample_size,
        min_shingles=min_shingles,
        sample_seed=sample_seed,
    )
    documents = [dataset.data[index] for index in indices]
    return documents, create_shingle_sets(documents, k)
