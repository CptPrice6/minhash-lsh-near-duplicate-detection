import argparse
import gc
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from near_dup.data import load_wikipedia_sample
from near_dup.lsh import lsh_candidates
from near_dup.minhash import compute_signature_matrix
from near_dup.preprocessing import create_shingle_sets, select_document_indices

SAMPLE_SIZES = [500, 1000, 2000, 3000, 5000, 7500, 10_000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=3)
    return parser.parse_args()


def run_once(
    documents: list[str],
    k: int,
    num_hashes: int,
    num_bands: int,
    rows_per_band: int,
    hash_seed: int,
) -> dict[str, float | int]:
    shingling_start = perf_counter()
    shingle_sets = create_shingle_sets(documents, k)
    shingling_time = perf_counter() - shingling_start

    signature_start = perf_counter()
    signatures = compute_signature_matrix(
        shingle_sets,
        num_hashes,
        seed=hash_seed,
        hash_family="murmur",
    )
    signature_time = perf_counter() - signature_start

    lsh_start = perf_counter()
    candidates = lsh_candidates(signatures, num_bands, rows_per_band)
    lsh_time = perf_counter() - lsh_start

    return {
        "lsh_candidate_pairs": len(candidates),
        "shingling_time_seconds": shingling_time,
        "signature_time_seconds": signature_time,
        "lsh_time_seconds": lsh_time,
        "total_runtime_seconds": shingling_time + signature_time + lsh_time,
    }


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    timing_columns = [
        "shingling_time_seconds",
        "signature_time_seconds",
        "lsh_time_seconds",
        "total_runtime_seconds",
    ]
    rows = []

    for documents, group in raw.groupby("documents", sort=True):
        first = group.iloc[0]
        row = {
            column: first[column]
            for column in [
                "dataset",
                "documents",
                "k",
                "min_shingles",
                "sample_seed",
                "hash_seed",
                "hash_family",
                "num_hashes",
                "num_bands",
                "rows_per_band",
                "total_possible_pairs",
                "lsh_candidate_pairs",
                "candidate_fraction",
                "candidate_reduction",
            ]
        }
        row["repetitions"] = len(group)

        for column in timing_columns:
            values = group[column].to_numpy(dtype=float)
            row[column] = np.median(values)
            row[f"{column}_q25"] = np.percentile(values, 25)
            row[f"{column}_q75"] = np.percentile(values, 75)

        row["documents_per_second"] = documents / row["total_runtime_seconds"]
        rows.append(row)

    return pd.DataFrame(rows)


def plot_runtime(results: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    for column, label in [
        ("shingling_time_seconds", "Shingling"),
        ("signature_time_seconds", "MinHash signatures"),
        ("lsh_time_seconds", "LSH banding"),
        ("total_runtime_seconds", "Total"),
    ]:
        axis.plot(results["documents"], results[column], marker="o", label=label)

    axis.fill_between(
        results["documents"].to_numpy(),
        results["total_runtime_seconds_q25"].to_numpy(),
        results["total_runtime_seconds_q75"].to_numpy(),
        alpha=0.15,
        label="Total runtime IQR",
    )
    axis.set(
        xlabel="Number of documents",
        ylabel="Runtime in seconds",
        title="Wikipedia runtime scalability",
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)


def plot_candidates(results: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(
        results["documents"],
        results["total_possible_pairs"],
        marker="o",
        label="All possible pairs",
    )
    axis.plot(
        results["documents"],
        results["lsh_candidate_pairs"],
        marker="o",
        label="LSH candidate pairs",
    )
    axis.set(
        xlabel="Number of documents",
        ylabel="Number of pairs",
        title="Wikipedia candidate-pair reduction",
    )
    axis.set_yscale("log")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.repetitions <= 0:
        raise ValueError("repetitions must be greater than 0")

    k = 5
    min_shingles = 20
    sample_seed = 42
    hash_seed = 42
    num_hashes = 200
    num_bands = 50
    rows_per_band = 4

    dataset = load_wikipedia_sample(
        max_documents=15_000,
        sample_seed=sample_seed,
    )
    selected_indices = select_document_indices(
        dataset=dataset,
        filter_k=k,
        sample_size=max(SAMPLE_SIZES),
        min_shingles=min_shingles,
        sample_seed=sample_seed,
    )
    documents = [dataset.data[index] for index in selected_indices]

    raw_rows = []
    for sample_size in SAMPLE_SIZES:
        subset = documents[:sample_size]
        total_pairs = sample_size * (sample_size - 1) // 2
        expected_candidates = None
        print(f"\n{sample_size} documents")

        for repetition in range(1, args.repetitions + 1):
            gc.collect()
            measurement = run_once(
                subset,
                k,
                num_hashes,
                num_bands,
                rows_per_band,
                hash_seed,
            )
            candidates = int(measurement["lsh_candidate_pairs"])
            if expected_candidates is None:
                expected_candidates = candidates
            elif candidates != expected_candidates:
                raise RuntimeError("Candidate count changed between repetitions")

            fraction = candidates / total_pairs
            raw_rows.append(
                {
                    "dataset": "wikipedia",
                    "documents": sample_size,
                    "repetition": repetition,
                    "k": k,
                    "min_shingles": min_shingles,
                    "sample_seed": sample_seed,
                    "hash_seed": hash_seed,
                    "hash_family": "murmur",
                    "num_hashes": num_hashes,
                    "num_bands": num_bands,
                    "rows_per_band": rows_per_band,
                    "total_possible_pairs": total_pairs,
                    "candidate_fraction": fraction,
                    "candidate_reduction": 1 - fraction,
                    **measurement,
                }
            )
            print(
                f"  run {repetition}: "
                f"{measurement['total_runtime_seconds']:.2f}s, "
                f"{candidates:,} candidates"
            )

    raw = pd.DataFrame(raw_rows)
    results = summarise(raw)
    tables_dir = Path("results/tables")
    figures_dir = Path("results/figures/scalability")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw.to_csv(
        tables_dir / "wikipedia_scalability_raw.csv",
        index=False,
        float_format="%.6f",
    )
    results.to_csv(
        tables_dir / "wikipedia_scalability.csv",
        index=False,
        float_format="%.6f",
    )
    plot_runtime(results, figures_dir / "wikipedia_runtime_scalability.png")
    plot_candidates(results, figures_dir / "wikipedia_candidate_reduction.png")
    print("\nSaved Wikipedia scalability results.")


if __name__ == "__main__":
    main()
