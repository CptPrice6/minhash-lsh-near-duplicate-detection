from near_dup.data import load_20newsgroups
from near_dup.preprocessing import normalize_text, tokenize_words, create_word_shingles


def main():
    print("Loading 20 Newsgroups dataset...")

    dataset = load_20newsgroups()
    document = dataset.data[0]

    print("\nOriginal document preview:")
    print(document[:500])
    normalized = normalize_text(document)
    tokens = tokenize_words(document)
    shingles = create_word_shingles(document, k=5)

    print("\nPreprocessing result:")
    print(f"Number of characters before normalization: {len(document)}")
    print(f"Number of characters after normalization: {len(normalized)}")
    print(f"Number of word tokens: {len(tokens)}")

    print("\nShingling result:")
    print("Shingle type: word k-shingles")
    print("k = 5")
    print(f"Number of unique shingles: {len(shingles)}")

    print("\nFirst 20 shingles:")
    for shingle in sorted(shingles)[:20]:
        print(f"- {shingle}")


if __name__ == "__main__":
    main()
