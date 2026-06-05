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
