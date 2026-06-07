from time import perf_counter

import numpy as np
import pandas as pd

from near_dup.evaluation import evaluate_candidates
from near_dup.lsh import lsh_candidates
from near_dup.minhash import (
    SUPPORTED_HASH_FAMILIES,
    compute_signature_matrix,
)
from near_dup.preprocessing import create_shingle_sets, prepare_shingled_documents
from near_dup.similarity import compute_ground_truth


def _validate_experiment_inputs(
    dataset_name: str,
    sample_size: int,
    min_shingles: int,
    threshold: float,
    k_values: list[int],
    hash_families: list[str],
    lsh_configs: list[dict[str, int]],
) -> None:
    """Validate an experiment grid before expensive computation starts."""
    if not dataset_name.strip():
        raise ValueError("dataset_name must not be empty")
    if sample_size <= 1:
        raise ValueError("sample_size must be greater than 1")
    if min_shingles <= 0:
        raise ValueError("min_shingles must be greater than 0")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    if not hash_families:
        raise ValueError("hash_families must not be empty")

    unsupported = set(hash_families) - SUPPORTED_HASH_FAMILIES
    if unsupported:
        raise ValueError(f"Unsupported hash families: {sorted(unsupported)}")

    if not lsh_configs:
        raise ValueError("lsh_configs must not be empty")

    required_keys = {
        "num_hashes",
        "num_bands",
        "rows_per_band",
    }

    for config in lsh_configs:
        missing_keys = required_keys - config.keys()

        if missing_keys:
            raise ValueError(
                f"LSH configuration is missing keys: {sorted(missing_keys)}"
            )

        num_hashes = config["num_hashes"]
        num_bands = config["num_bands"]
        rows_per_band = config["rows_per_band"]

        if min(num_hashes, num_bands, rows_per_band) <= 0:
            raise ValueError("LSH configuration values must be positive")

        if num_bands * rows_per_band != num_hashes:
            raise ValueError("num_bands * rows_per_band must equal num_hashes")


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
    _validate_experiment_inputs(
        dataset_name=dataset_name,
        sample_size=sample_size,
        min_shingles=min_shingles,
        threshold=threshold,
        k_values=k_values,
        hash_families=hash_families,
        lsh_configs=lsh_configs,
    )

    results = []

    filter_k = max(k_values)

    documents, _ = prepare_shingled_documents(
        dataset=dataset,
        k=filter_k,
        sample_size=sample_size,
        min_shingles=min_shingles,
    )

    if len(documents) < sample_size:
        raise RuntimeError(
            f"Only {len(documents)} valid documents were found; "
            f"{sample_size} were requested."
        )

    for k in k_values:
        print(f"\nPreparing {dataset_name} shingles for k={k}...")

        shingling_start = perf_counter()

        shingle_sets = create_shingle_sets(
            documents=documents,
            k=k,
        )

        shingling_time = perf_counter() - shingling_start

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
            signature_cache: dict[int, tuple[np.ndarray, float]] = {}

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
                    signature_start = perf_counter()

                    signatures = compute_signature_matrix(
                        shingle_sets=shingle_sets,
                        num_hashes=num_hashes,
                        seed=seed,
                        hash_family=hash_family,
                    )

                    signature_time = perf_counter() - signature_start

                    signature_cache[num_hashes] = (
                        signatures,
                        signature_time,
                    )

                signatures, signature_time = signature_cache[num_hashes]

                lsh_start = perf_counter()

                candidates = lsh_candidates(
                    signatures=signatures,
                    num_bands=num_bands,
                    rows_per_band=rows_per_band,
                )

                lsh_time = perf_counter() - lsh_start
                total_runtime = shingling_time + signature_time + lsh_time

                metrics = evaluate_candidates(
                    candidates=candidates,
                    ground_truth=ground_truth,
                )

                candidate_fraction = (
                    len(candidates) / total_possible_pairs
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
                        "candidate_fraction": candidate_fraction,
                        "candidate_reduction": 1.0 - candidate_fraction,
                        "true_positives": metrics["true_positives"],
                        "false_positives": metrics["false_positives"],
                        "false_negatives": metrics["false_negatives"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1_score": metrics["f1_score"],
                        "ground_truth_time_seconds": ground_truth_time,
                        "shingling_time_seconds": shingling_time,
                        "signature_time_seconds": signature_time,
                        "lsh_time_seconds": lsh_time,
                        "total_runtime_seconds": total_runtime,
                    }
                )

    results_df = pd.DataFrame(results)

    float_columns = [
        "candidate_fraction",
        "candidate_reduction",
        "precision",
        "recall",
        "f1_score",
        "ground_truth_time_seconds",
        "shingling_time_seconds",
        "signature_time_seconds",
        "lsh_time_seconds",
        "total_runtime_seconds",
    ]

    results_df[float_columns] = results_df[float_columns].round(6)

    return results_df
