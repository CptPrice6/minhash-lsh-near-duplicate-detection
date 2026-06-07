from pathlib import Path
from time import perf_counter

import pandas as pd

from near_dup.data import load_20newsgroups
from near_dup.preprocessing import (
    create_shingle_sets,
    get_filtered_documents,
)
from near_dup.similarity import compute_ground_truth
from near_dup.minhash import compute_signature_matrix
from near_dup.lsh import lsh_candidates
from near_dup.evaluation import evaluate_candidates


def main():
    dataset = load_20newsgroups()

    sample_size = 2000
    min_shingles = 20
    threshold = 0.1
    seed = 42

    k_values = [3, 5]
    hash_families = ["linear", "murmur", "tabulation"]

    lsh_configs = [
        {"num_hashes": 100, "num_bands": 50, "rows_per_band": 2},
        {"num_hashes": 100, "num_bands": 25, "rows_per_band": 4},
        {"num_hashes": 100, "num_bands": 10, "rows_per_band": 10},
        {"num_hashes": 100, "num_bands": 20, "rows_per_band": 5},
        {"num_hashes": 200, "num_bands": 50, "rows_per_band": 4},
        {"num_hashes": 200, "num_bands": 100, "rows_per_band": 2},
    ]

    results = []

    for k in k_values:
        print(f"\nPreparing documents for k={k}...")
        documents = get_filtered_documents(
            dataset=dataset,
            k=k,
            sample_size=sample_size,
            min_shingles=min_shingles,
        )

        shingle_sets = create_shingle_sets(documents, k=k)

        print("Computing brute-force ground truth...")
        ground_truth_start = perf_counter()
        ground_truth = compute_ground_truth(shingle_sets, threshold=threshold)
        ground_truth_time = perf_counter() - ground_truth_start

        total_possible_pairs = len(documents) * (len(documents) - 1) // 2

        for hash_family in hash_families:
            signature_cache = {}

            for config in lsh_configs:
                num_hashes = config["num_hashes"]
                num_bands = config["num_bands"]
                rows_per_band = config["rows_per_band"]

                print(
                    f"Running k={k}, hash_family={hash_family}, "
                    f"hashes={num_hashes}, bands={num_bands}, rows={rows_per_band}"
                )

                if num_hashes not in signature_cache:
                    signature_cache[num_hashes] = compute_signature_matrix(
                        shingle_sets=shingle_sets,
                        num_hashes=num_hashes,
                        seed=seed,
                        hash_family=hash_family,
                    )

                start = perf_counter()

                signatures = signature_cache[num_hashes]

                candidates = lsh_candidates(
                    signatures=signatures,
                    num_bands=num_bands,
                    rows_per_band=rows_per_band,
                )

                runtime = perf_counter() - start

                metrics = evaluate_candidates(candidates, ground_truth)

                candidate_reduction = (
                    1 - (len(candidates) / total_possible_pairs)
                    if total_possible_pairs > 0
                    else 0.0
                )

                row = {
                    "dataset": "20_newsgroups",
                    "documents": len(documents),
                    "k": k,
                    "min_shingles": min_shingles,
                    "threshold": threshold,
                    "hash_family": hash_family,
                    "seed": seed,
                    "num_hashes": num_hashes,
                    "num_bands": num_bands,
                    "rows_per_band": rows_per_band,
                    "ground_truth_pairs": len(ground_truth),
                    "lsh_candidate_pairs": len(candidates),
                    "total_possible_pairs": total_possible_pairs,
                    "candidate_reduction": candidate_reduction,
                    "true_positives": metrics["true_positives"],
                    "false_positives": metrics["false_positives"],
                    "false_negatives": metrics["false_negatives"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "ground_truth_time_seconds": ground_truth_time,
                    "lsh_time_seconds": runtime,
                }

                results.append(row)

    results_df = pd.DataFrame(results)

    float_columns = [
        "candidate_reduction",
        "precision",
        "recall",
        "f1_score",
        "ground_truth_time_seconds",
        "lsh_time_seconds",
    ]

    results_df[float_columns] = results_df[float_columns].round(4)

    output_path = Path("results") / "tables" / "experiment_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_path, index=False)

    print("\nExperiments completed.")
    print(f"Results saved to: {output_path}")
    print("\nResults preview:")
    print(results_df)


if __name__ == "__main__":
    main()
