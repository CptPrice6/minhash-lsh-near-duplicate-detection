from near_dup.data import load_20newsgroups
from near_dup.preprocessing import create_word_shingles
from near_dup.similarity import compute_ground_truth
from near_dup.minhash import compute_signature_matrix
from near_dup.lsh import lsh_candidates


def main():
    dataset = load_20newsgroups()

    k = 5
    sample_size = 1000
    min_shingles = 20
    num_hashes = 100
    num_bands = 50
    rows_per_band = 2
    threshold = 0.2

    documents = []

    for text in dataset.data:
        shingles = create_word_shingles(text, k=k)

        if len(shingles) >= min_shingles:
            documents.append(text)

        if len(documents) == sample_size:
            break

    shingle_sets = [create_word_shingles(document, k=k) for document in documents]
    ground_truth = compute_ground_truth(shingle_sets, threshold=threshold)

    signatures = compute_signature_matrix(
        shingle_sets=shingle_sets,
        num_hashes=num_hashes,
        seed=42,
    )

    candidates = lsh_candidates(
        signatures=signatures,
        num_bands=num_bands,
        rows_per_band=rows_per_band,
    )

    true_pairs = set(ground_truth.keys())
    true_positives = candidates & true_pairs
    false_positives = candidates - true_pairs
    false_negatives = true_pairs - candidates

    precision = len(true_positives) / len(candidates) if candidates else 0.0
    recall = len(true_positives) / len(true_pairs) if true_pairs else 0.0

    print(f"Documents used: {len(documents)}")
    print(f"k: {k}")
    print(f"Number of hash functions: {num_hashes}")
    print(f"Bands: {num_bands}")
    print(f"Rows per band: {rows_per_band}")
    print(f"Ground truth threshold: {threshold}")
    print(f"Ground truth pairs: {len(ground_truth)}")
    print(f"LSH candidate pairs: {len(candidates)}")
    print(f"True positives: {len(true_positives)}")
    print(f"False positives: {len(false_positives)}")
    print(f"False negatives: {len(false_negatives)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


if __name__ == "__main__":
    main()
