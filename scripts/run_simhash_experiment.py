from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from near_dup.data import load_20newsgroups
from near_dup.preprocessing import (
    create_shingle_sets,
    get_filtered_documents,
)
from near_dup.sampling import sample_pairs_below_threshold
from near_dup.simhash import (
    compute_simhash,
    cosine_similarity,
    estimate_cosine_from_simhash,
    hamming_distance,
    term_frequencies,
)
from near_dup.similarity import (
    compute_ground_truth,
)


def calculate_summary(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    subsets = {
        "all": results_df,
        "similar": results_df[results_df["pair_type"] == "similar"],
        "dissimilar": results_df[results_df["pair_type"] == "dissimilar"],
    }

    rows = []

    for pair_type, subset in subsets.items():
        for num_bits in sorted(subset["num_bits"].unique()):
            bit_subset = subset[subset["num_bits"] == num_bits]

            errors = bit_subset["estimated_cosine"] - bit_subset["exact_cosine"]

            correlation = (
                bit_subset[["exact_cosine", "estimated_cosine"]].corr().iloc[0, 1]
            )

            rows.append(
                {
                    "pair_type": pair_type,
                    "num_bits": num_bits,
                    "number_of_pairs": len(bit_subset),
                    "mean_absolute_error": errors.abs().mean(),
                    "root_mean_squared_error": np.sqrt(np.mean(errors**2)),
                    "maximum_absolute_error": errors.abs().max(),
                    "correlation": correlation,
                }
            )

    return pd.DataFrame(rows)


def plot_simhash_vs_cosine(
    results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))

    for num_bits in sorted(results_df["num_bits"].unique()):
        subset = results_df[results_df["num_bits"] == num_bits]

        plt.scatter(
            subset["exact_cosine"],
            subset["estimated_cosine"],
            label=f"{num_bits} bits",
            alpha=0.65,
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Ideal estimate",
    )

    plt.xlabel("Exact cosine similarity")
    plt.ylabel("SimHash-estimated cosine similarity")
    plt.title("SimHash estimate versus exact cosine similarity")
    plt.xlim(0, 1)
    plt.ylim(-0.15, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    dataset = load_20newsgroups()

    sample_size = 1000
    min_shingles = 20
    k = 5
    seed = 42

    similarity_threshold = 0.05
    max_similar_pairs = 100
    number_of_dissimilar_pairs = 100

    bit_sizes = [64, 128, 256]

    print("Preparing documents...")

    documents = get_filtered_documents(
        dataset=dataset,
        k=k,
        sample_size=sample_size,
        min_shingles=min_shingles,
    )

    shingle_sets = create_shingle_sets(
        documents=documents,
        k=k,
    )

    print(f"Documents used: {len(documents)}")
    print(f"Shingle size k: {k}")

    print("Finding similar pairs...")

    ground_truth = compute_ground_truth(
        shingle_sets=shingle_sets,
        threshold=similarity_threshold,
    )

    similar_pairs = dict(
        sorted(
            ground_truth.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:max_similar_pairs]
    )

    print("Sampling dissimilar pairs...")

    dissimilar_pairs = sample_pairs_below_threshold(
        shingle_sets=shingle_sets,
        excluded_pairs=set(ground_truth.keys()),
        number_of_pairs=number_of_dissimilar_pairs,
        similarity_threshold=similarity_threshold,
        seed=seed,
    )

    validation_pairs: list[tuple[int, int, float, str]] = []

    for (i, j), similarity in similar_pairs.items():
        validation_pairs.append((i, j, similarity, "similar"))

    for (i, j), similarity in dissimilar_pairs.items():
        validation_pairs.append((i, j, similarity, "dissimilar"))

    selected_indices = sorted(
        {index for i, j, _, _ in validation_pairs for index in (i, j)}
    )

    print(
        f"Pairs used: {len(validation_pairs)} "
        f"({len(similar_pairs)} similar, "
        f"{len(dissimilar_pairs)} dissimilar)"
    )

    print("Creating word-frequency vectors...")

    frequency_vectors = {
        index: term_frequencies(documents[index]) for index in selected_indices
    }

    fingerprints: dict[int, dict[int, int]] = {}

    for num_bits in bit_sizes:
        print(f"Computing {num_bits}-bit SimHash fingerprints...")

        fingerprints[num_bits] = {
            index: compute_simhash(
                frequencies=frequency_vectors[index],
                num_bits=num_bits,
                seed=seed,
            )
            for index in selected_indices
        }

    rows = []

    for i, j, exact_jaccard, pair_type in validation_pairs:
        exact_cosine = cosine_similarity(
            frequency_vectors[i],
            frequency_vectors[j],
        )

        for num_bits in bit_sizes:
            fingerprint_i = fingerprints[num_bits][i]
            fingerprint_j = fingerprints[num_bits][j]

            distance = hamming_distance(
                fingerprint_i,
                fingerprint_j,
            )

            estimated_cosine = estimate_cosine_from_simhash(
                fingerprint_a=fingerprint_i,
                fingerprint_b=fingerprint_j,
                num_bits=num_bits,
            )

            rows.append(
                {
                    "document_i": i,
                    "document_j": j,
                    "pair_type": pair_type,
                    "num_bits": num_bits,
                    "exact_jaccard": exact_jaccard,
                    "exact_cosine": exact_cosine,
                    "hamming_distance": distance,
                    "estimated_cosine": estimated_cosine,
                    "absolute_error": abs(estimated_cosine - exact_cosine),
                }
            )

    results_df = pd.DataFrame(rows)
    summary_df = calculate_summary(results_df)

    results_df = results_df.round(6)
    summary_df = summary_df.round(6)

    tables_dir = Path("results") / "tables"
    figures_dir = Path("results") / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_path = tables_dir / "simhash_comparison.csv"
    summary_path = tables_dir / "simhash_summary.csv"
    figure_path = figures_dir / "simhash_vs_cosine.png"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    plot_simhash_vs_cosine(
        results_df=results_df,
        output_path=figure_path,
    )

    print("\nSimHash experiment completed.")
    print(f"Pair-level results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Figure: {figure_path}")
    print("\nSummary:")
    print(summary_df)


if __name__ == "__main__":
    main()
