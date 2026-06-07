import re


def normalize_text(text: str) -> str:
    """Lowercase text and collapse consecutive whitespace."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_words(text: str) -> list[str]:
    """Normalize text and split it into alphanumeric word tokens."""
    text = normalize_text(text)
    tokens = re.findall(r"[a-zA-Z0-9']+", text)
    return tokens


def create_word_shingles(text: str, k: int = 5) -> set[str]:
    """Create the set of unique contiguous word k-shingles."""
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


def create_shingle_sets(documents: list[str], k: int) -> list[set[str]]:
    """Create one word-shingle set for each document."""
    return [create_word_shingles(document, k=k) for document in documents]


def get_filtered_documents(
    dataset,
    k: int,
    sample_size: int,
    min_shingles: int,
) -> list[str]:
    documents, _ = prepare_shingled_documents(
        dataset=dataset,
        k=k,
        sample_size=sample_size,
        min_shingles=min_shingles,
    )
    return documents


def prepare_shingled_documents(
    dataset,
    k: int,
    sample_size: int,
    min_shingles: int,
) -> tuple[list[str], list[set[str]]]:
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than 0")
    if min_shingles <= 0:
        raise ValueError("min_shingles must be greater than 0")

    documents = []
    shingle_sets = []

    for text in dataset.data:
        shingles = create_word_shingles(text, k=k)

        if len(shingles) >= min_shingles:
            documents.append(text)
            shingle_sets.append(shingles)

        if len(documents) == sample_size:
            break

    return documents, shingle_sets
