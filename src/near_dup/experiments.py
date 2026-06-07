from time import perf_counter

import pandas as pd

from near_dup.evaluation import evaluate_candidates
from near_dup.lsh import lsh_candidates
from near_dup.minhash import compute_signature_matrix
from near_dup.preprocessing import create_shingle_sets, get_filtered_documents
from near_dup.similarity import compute_ground_truth


def run_experiment_grid(
    dataset,
    dataset_name: str,
    sample_size: int,
    min_shingles: int,
    threshold: float,
    seed: int,
    k_values: list[int],
    hash_families: list[str],
    lsh_configs: list[dict[str, int]],
) -> pd.DataFrame:
    results = []

    for k in k_values:
        print(f"\nPreparing {dataset_name} documents for k={k}...")

        documents = get_filtered_documents(
            dataset=dataset,
            k=k,
            sample_size=sample_size,
            min_shingles=min_shingles,
        )

        shingle_sets = create_shingle_sets(documents, k=k)

        print(f"Documents used: {len(documents)}")
        print("Computing brute-force ground truth...")

        ground_truth_start = perf_counter()
        ground_truth = compute_ground_truth(
            shingle_sets=shingle_sets,
            threshold=threshold,
        )
        ground_truth_time = perf_counter() - ground_truth_start

        total_possible_pairs = len(documents) * (len(documents) - 1) // 2

        for hash_family in hash_families:
            signature_cache: dict[int, object] = {}

            for config in lsh_configs:
                num_hashes = config["num_hashes"]
                num_bands = config["num_bands"]
                rows_per_band = config["rows_per_band"]

                print(
                    f"Running dataset={dataset_name}, k={k}, "
                    f"family={hash_family}, hashes={num_hashes}, "
                    f"bands={num_bands}, rows={rows_per_band}"
                )

                if num_hashes not in signature_cache:
                    signature_cache[num_hashes] = compute_signature_matrix(
                        shingle_sets=shingle_sets,
                        num_hashes=num_hashes,
                        seed=seed,
                        hash_family=hash_family,
                    )

                signatures = signature_cache[num_hashes]

                lsh_start = perf_counter()

                candidates = lsh_candidates(
                    signatures=signatures,
                    num_bands=num_bands,
                    rows_per_band=rows_per_band,
                )

                lsh_time = perf_counter() - lsh_start

                metrics = evaluate_candidates(
                    candidates=candidates,
                    ground_truth=ground_truth,
                )

                candidate_reduction = (
                    1.0 - len(candidates) / total_possible_pairs
                    if total_possible_pairs > 0
                    else 0.0
                )

                results.append(
                    {
                        "dataset": dataset_name,
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
                        "lsh_time_seconds": lsh_time,
                    }
                )

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

    return results_df
