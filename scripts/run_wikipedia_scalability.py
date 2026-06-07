from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd

from near_dup.data import load_wikipedia_sample
from near_dup.lsh import lsh_candidates
from near_dup.minhash import compute_signature_matrix
from near_dup.preprocessing import create_shingle_sets, get_filtered_documents


def plot_runtime_scalability(
    results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))

    plt.plot(
        results_df["documents"],
        results_df["shingling_time_seconds"],
        marker="o",
        label="Shingling",
    )
    plt.plot(
        results_df["documents"],
        results_df["signature_time_seconds"],
        marker="o",
        label="MinHash signatures",
    )
    plt.plot(
        results_df["documents"],
        results_df["lsh_time_seconds"],
        marker="o",
        label="LSH banding",
    )
    plt.plot(
        results_df["documents"],
        results_df["total_runtime_seconds"],
        marker="o",
        label="Total",
    )

    plt.xlabel("Number of documents")
    plt.ylabel("Runtime in seconds")
    plt.title("Wikipedia runtime scalability")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_candidate_scalability(
    results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))

    plt.plot(
        results_df["documents"],
        results_df["total_possible_pairs"],
        marker="o",
        label="All possible pairs",
    )
    plt.plot(
        results_df["documents"],
        results_df["lsh_candidate_pairs"],
        marker="o",
        label="LSH candidate pairs",
    )

    plt.xlabel("Number of documents")
    plt.ylabel("Number of pairs")
    plt.title("Wikipedia candidate-pair reduction")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    sample_sizes = [1000, 2000, 5000]

    max_sample_size = max(sample_sizes)
    streamed_documents = 7000

    k = 5
    min_shingles = 20
    num_hashes = 100
    num_bands = 50
    rows_per_band = 2
    hash_family = "murmur"
    seed = 42

    print("Streaming Wikipedia articles...")
    print(f"Articles requested from stream: {streamed_documents}")

    load_start = perf_counter()

    dataset = load_wikipedia_sample(
        max_documents=streamed_documents,
        seed=seed,
    )

    load_time = perf_counter() - load_start

    print(f"Streaming completed in {load_time:.2f} seconds.")
    print("Filtering empty or very short articles...")

    filter_start = perf_counter()

    filtered_documents = get_filtered_documents(
        dataset=dataset,
        k=k,
        sample_size=max_sample_size,
        min_shingles=min_shingles,
    )

    filter_time = perf_counter() - filter_start

    if len(filtered_documents) < max_sample_size:
        raise RuntimeError(
            f"Only {len(filtered_documents)} valid documents were found, "
            f"but {max_sample_size} are required."
        )

    print(f"Valid documents available: {len(filtered_documents)}")
    print(f"Filtering completed in {filter_time:.2f} seconds.")

    results = []

    for sample_size in sample_sizes:
        print(f"\nRunning scalability experiment with {sample_size} documents...")

        documents = filtered_documents[:sample_size]

        shingling_start = perf_counter()
        shingle_sets = create_shingle_sets(
            documents=documents,
            k=k,
        )
        shingling_time = perf_counter() - shingling_start

        signature_start = perf_counter()
        signatures = compute_signature_matrix(
            shingle_sets=shingle_sets,
            num_hashes=num_hashes,
            seed=seed,
            hash_family=hash_family,
        )
        signature_time = perf_counter() - signature_start

        lsh_start = perf_counter()
        candidates = lsh_candidates(
            signatures=signatures,
            num_bands=num_bands,
            rows_per_band=rows_per_band,
        )
        lsh_time = perf_counter() - lsh_start

        total_runtime = shingling_time + signature_time + lsh_time
        total_possible_pairs = sample_size * (sample_size - 1) // 2

        candidate_fraction = (
            len(candidates) / total_possible_pairs if total_possible_pairs > 0 else 0.0
        )

        candidate_reduction = 1.0 - candidate_fraction

        results.append(
            {
                "dataset": "wikipedia",
                "documents": sample_size,
                "k": k,
                "min_shingles": min_shingles,
                "hash_family": hash_family,
                "seed": seed,
                "num_hashes": num_hashes,
                "num_bands": num_bands,
                "rows_per_band": rows_per_band,
                "total_possible_pairs": total_possible_pairs,
                "lsh_candidate_pairs": len(candidates),
                "candidate_fraction": candidate_fraction,
                "candidate_reduction": candidate_reduction,
                "shingling_time_seconds": shingling_time,
                "signature_time_seconds": signature_time,
                "lsh_time_seconds": lsh_time,
                "total_runtime_seconds": total_runtime,
            }
        )

        print(f"Shingling time: {shingling_time:.4f} seconds")
        print(f"Signature time: {signature_time:.4f} seconds")
        print(f"LSH time: {lsh_time:.4f} seconds")
        print(f"Total runtime: {total_runtime:.4f} seconds")
        print(f"Possible pairs: {total_possible_pairs:,}")
        print(f"LSH candidates: {len(candidates):,}")
        print(f"Candidate reduction: {candidate_reduction:.6%}")

    results_df = pd.DataFrame(results)

    float_columns = [
        "candidate_fraction",
        "candidate_reduction",
        "shingling_time_seconds",
        "signature_time_seconds",
        "lsh_time_seconds",
        "total_runtime_seconds",
    ]

    results_df[float_columns] = results_df[float_columns].round(6)

    tables_dir = Path("results") / "tables"
    figures_dir = Path("results") / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_path = tables_dir / "wikipedia_scalability.csv"
    runtime_figure_path = figures_dir / "wikipedia_runtime_scalability.png"
    candidate_figure_path = figures_dir / "wikipedia_candidate_reduction.png"

    results_df.to_csv(results_path, index=False)

    plot_runtime_scalability(
        results_df=results_df,
        output_path=runtime_figure_path,
    )

    plot_candidate_scalability(
        results_df=results_df,
        output_path=candidate_figure_path,
    )

    print("\nWikipedia scalability experiment completed.")
    print(f"Results: {results_path}")
    print(f"Runtime figure: {runtime_figure_path}")
    print(f"Candidate figure: {candidate_figure_path}")
    print("\nResults summary:")
    print(results_df)


if __name__ == "__main__":
    main()
