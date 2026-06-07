from near_dup.data import load_20newsgroups
from near_dup.preprocessing import create_shingle_sets
from near_dup.similarity import compute_ground_truth


def main():
    print("Loading 20 Newsgroups dataset...")

    dataset = load_20newsgroups()

    documents = dataset.data[:1000]
    categories = [dataset.target_names[target] for target in dataset.target[:1000]]

    print(f"Using first {len(documents)} documents for brute-force test.")

    k = 5
    threshold = 0.2

    print(f"\nCreating word shingles with k = {k}...")
    shingle_sets = create_shingle_sets(documents, k=k)

    print("\nComputing brute-force ground truth...")
    ground_truth = compute_ground_truth(shingle_sets, threshold=threshold)

    print(f"Threshold: {threshold}")
    print(f"Number of near-duplicate pairs found: {len(ground_truth)}")

    print("\nFirst 10 ground truth pairs:")
    for (i, j), sim in list(ground_truth.items())[:10]:
        print(
            f"Pair ({i}, {j}) | similarity = {sim:.4f} | "
            f"categories = {categories[i]} / {categories[j]}"
        )


if __name__ == "__main__":
    main()
