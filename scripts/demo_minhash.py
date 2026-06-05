from near_dup.data import load_20newsgroups
from near_dup.preprocessing import create_word_shingles
from near_dup.similarity import jaccard_similarity
from near_dup.minhash import compute_signature_matrix, estimate_jaccard_from_signatures


def main():
    print("Loading 20 Newsgroups dataset...")

    dataset = load_20newsgroups()

    documents = dataset.data[:1000]
    categories = [dataset.target_names[target] for target in dataset.target[:1000]]

    k = 5
    num_hashes = 100

    print(f"Using {len(documents)} documents.")
    print(f"Creating word shingles with k = {k}...")

    shingle_sets = [create_word_shingles(document, k=k) for document in documents]

    print(f"Computing MinHash signatures with {num_hashes} hash functions...")
    signatures = compute_signature_matrix(
        shingle_sets=shingle_sets,
        num_hashes=num_hashes,
        seed=42,
    )

    print("\nSignature matrix created.")
    print(f"Shape: {signatures.shape}")

    pairs_to_check = [
        (0, 1),
        (123, 594),
        (358, 408),
    ]

    print("\nExact Jaccard vs MinHash estimated Jaccard:")

    for i, j in pairs_to_check:
        exact = jaccard_similarity(shingle_sets[i], shingle_sets[j])
        estimated = estimate_jaccard_from_signatures(signatures[i], signatures[j])

        print(
            f"Pair ({i}, {j}) | "
            f"categories = {categories[i]} / {categories[j]} | "
            f"exact = {exact:.4f} | "
            f"estimated = {estimated:.4f}"
        )


if __name__ == "__main__":
    main()
