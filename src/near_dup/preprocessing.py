import re


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_words(text: str) -> list[str]:
    text = normalize_text(text)
    tokens = re.findall(r"[a-zA-Z0-9']+", text)
    return tokens


def create_word_shingles(text: str, k: int = 5) -> set[str]:
    if k <= 0:
        raise ValueError("k must be greater than 0")

    tokens = tokenize_words(text)

    if len(tokens) == 0:
        return set()

    if len(tokens) < k:
        return {" ".join(tokens)}

    shingles = set()

    for i in range(len(tokens) - k + 1):
        shingle = " ".join(tokens[i : i + k])
        shingles.add(shingle)

    return shingles


def get_filtered_documents(dataset, k: int, sample_size: int, min_shingles: int):
    documents = []

    for text in dataset.data:
        if len(create_word_shingles(text, k=k)) >= min_shingles:
            documents.append(text)
        if len(documents) == sample_size:
            break

    return documents


def create_shingle_sets(documents: list[str], k: int) -> list[set[str]]:
    return [create_word_shingles(document, k=k) for document in documents]
