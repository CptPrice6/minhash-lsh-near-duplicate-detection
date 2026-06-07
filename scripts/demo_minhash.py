from near_dup.data import load_20newsgroups
from near_dup.preprocessing import create_shingle_sets
from near_dup.similarity import jaccard_similarity
from near_dup.minhash import compute_signature_matrix, estimate_jaccard_from_signatures


def main():
    dataset = load_20newsgroups()

    documents = dataset.data[:1000]
    categories = [dataset.target_names[target] for target in dataset.target[:1000]]

    k = 5
    num_hashes = 100

    shingle_sets = create_shingle_sets(documents, k=k)

    pairs_to_check = [
        (0, 1),
        (123, 594),
        (358, 408),
    ]

    hash_families = [
        "linear",
        "murmur",
        "tabulation",
    ]

    print(f"Documents used: {len(documents)}")
    print(f"k: {k}")
    print(f"Number of hash functions: {num_hashes}")

    for hash_family in hash_families:
        print()
        print(f"Hash family: {hash_family}")

        signatures = compute_signature_matrix(
            shingle_sets=shingle_sets,
            num_hashes=num_hashes,
            seed=42,
            hash_family=hash_family,
        )

        print(f"Signature matrix shape: {signatures.shape}")
        print("Exact Jaccard vs MinHash estimated Jaccard:")

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
