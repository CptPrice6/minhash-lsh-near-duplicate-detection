from near_dup.data import load_20newsgroups
from near_dup.evaluation import evaluate_candidates
from near_dup.preprocessing import (
    create_shingle_sets,
    get_filtered_documents,
)
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

    documents = get_filtered_documents(
        dataset=dataset,
        k=k,
        sample_size=sample_size,
        min_shingles=min_shingles,
    )

    shingle_sets = create_shingle_sets(documents, k=k)
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

    metrics = evaluate_candidates(candidates, ground_truth)

    print(f"Documents used: {len(documents)}")
    print(f"k: {k}")
    print(f"Number of hash functions: {num_hashes}")
    print(f"Bands: {num_bands}")
    print(f"Rows per band: {rows_per_band}")
    print(f"Ground truth threshold: {threshold}")
    print(f"Ground truth pairs: {len(ground_truth)}")
    print(f"LSH candidate pairs: {len(candidates)}")
    print(f"True positives: {metrics['true_positives']}")
    print(f"False positives: {metrics['false_positives']}")
    print(f"False negatives: {metrics['false_negatives']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
