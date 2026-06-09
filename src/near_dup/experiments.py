from time import perf_counter

import numpy as np
import pandas as pd

from near_dup.evaluation import evaluate_candidates
from near_dup.lsh import lsh_candidates
from near_dup.minhash import SUPPORTED_HASH_FAMILIES, compute_signature_matrix
from near_dup.preprocessing import create_shingle_sets, select_document_indices
from near_dup.similarity import compute_ground_truth


def _validate_grid(
    dataset_name: str,
    sample_size: int,
    min_shingles: int,
    threshold: float,
    k_values: list[int],
    hash_families: list[str],
    lsh_configs: list[dict[str, int]],
) -> None:
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
    if not hash_families or not set(hash_families) <= SUPPORTED_HASH_FAMILIES:
        raise ValueError("hash_families contains an unsupported value")
    if not lsh_configs:
        raise ValueError("lsh_configs must not be empty")

    required = {"num_hashes", "num_bands", "rows_per_band"}
    for config in lsh_configs:
        if set(config) != required:
            raise ValueError(f"Each LSH configuration must contain {sorted(required)}")
        if min(config.values()) <= 0:
            raise ValueError("LSH configuration values must be positive")
        if config["num_bands"] * config["rows_per_band"] != config["num_hashes"]:
            raise ValueError("num_bands * rows_per_band must equal num_hashes")


def run_experiment_grid(
    dataset,
    dataset_name: str,
    sample_size: int,
    min_shingles: int,
    threshold: float,
    sample_seed: int,
    hash_seed: int,
    k_values: list[int],
    hash_families: list[str],
    lsh_configs: list[dict[str, int]],
) -> pd.DataFrame:
    _validate_grid(
        dataset_name,
        sample_size,
        min_shingles,
        threshold,
        k_values,
        hash_families,
        lsh_configs,
    )

    selected_indices = select_document_indices(
        dataset=dataset,
        filter_k=max(k_values),
        sample_size=sample_size,
        min_shingles=min_shingles,
        sample_seed=sample_seed,
    )
    documents = [dataset.data[index] for index in selected_indices]
    total_possible_pairs = sample_size * (sample_size - 1) // 2
    results = []

    print(f"Selected {sample_size} documents with sample seed {sample_seed}.")

    for k in k_values:
        shingling_start = perf_counter()
        shingle_sets = create_shingle_sets(documents, k)
        shingling_time = perf_counter() - shingling_start

        ground_truth_start = perf_counter()
        ground_truth = compute_ground_truth(shingle_sets, threshold)
        ground_truth_time = perf_counter() - ground_truth_start

        print(
            f"\nk={k}: {len(ground_truth)} ground-truth pairs "
            f"in {ground_truth_time:.2f}s"
        )

        for hash_family in hash_families:
            signature_cache: dict[int, tuple[np.ndarray, float]] = {}

            for config in lsh_configs:
                num_hashes = config["num_hashes"]

                if num_hashes not in signature_cache:
                    signature_start = perf_counter()
                    signatures = compute_signature_matrix(
                        shingle_sets=shingle_sets,
                        num_hashes=num_hashes,
                        seed=hash_seed,
                        hash_family=hash_family,
                    )
                    signature_cache[num_hashes] = (
                        signatures,
                        perf_counter() - signature_start,
                    )

                signatures, signature_time = signature_cache[num_hashes]

                lsh_start = perf_counter()
                candidates = lsh_candidates(
                    signatures=signatures,
                    num_bands=config["num_bands"],
                    rows_per_band=config["rows_per_band"],
                )
                lsh_time = perf_counter() - lsh_start
                metrics = evaluate_candidates(candidates, ground_truth)
                candidate_fraction = len(candidates) / total_possible_pairs

                results.append(
                    {
                        "dataset": dataset_name,
                        "documents": sample_size,
                        "k": k,
                        "min_shingles": min_shingles,
                        "threshold": threshold,
                        "sample_seed": sample_seed,
                        "hash_seed": hash_seed,
                        "hash_family": hash_family,
                        "num_hashes": num_hashes,
                        "num_bands": config["num_bands"],
                        "rows_per_band": config["rows_per_band"],
                        "ground_truth_pairs": len(ground_truth),
                        "lsh_candidate_pairs": len(candidates),
                        "total_possible_pairs": total_possible_pairs,
                        "candidate_fraction": candidate_fraction,
                        "candidate_reduction": 1.0 - candidate_fraction,
                        **metrics,
                        "ground_truth_time_seconds": ground_truth_time,
                        "shingling_time_seconds": shingling_time,
                        "signature_time_seconds": signature_time,
                        "lsh_time_seconds": lsh_time,
                        "total_runtime_seconds": (
                            shingling_time + signature_time + lsh_time
                        ),
                    }
                )

    return pd.DataFrame(results)
