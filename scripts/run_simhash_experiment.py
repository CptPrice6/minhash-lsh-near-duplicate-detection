from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd

from near_dup.data import load_20newsgroups
from near_dup.preprocessing import prepare_documents
from near_dup.sampling import sample_pairs_below_threshold
from near_dup.simhash import (
    calculate_simhash_summary,
    compute_simhash,
    cosine_similarity,
    estimate_cosine_from_simhash,
    hamming_distance,
    term_frequencies,
)
from near_dup.similarity import compute_ground_truth

BIT_SIZES = [64, 128, 256]


def choose_pairs(
    shingle_sets: list[set[str]],
    threshold: float,
    seed: int,
) -> list[tuple[int, int, float, str]]:
    ground_truth = compute_ground_truth(shingle_sets, threshold)
    positives = sorted(ground_truth.items(), key=lambda item: item[1], reverse=True)[
        :100
    ]
    negatives = sample_pairs_below_threshold(
        shingle_sets=shingle_sets,
        excluded_pairs=set(ground_truth),
        number_of_pairs=100,
        similarity_threshold=threshold,
        seed=seed,
    )

    pairs = [(i, j, value, "similar") for (i, j), value in positives]
    pairs.extend((i, j, value, "dissimilar") for (i, j), value in negatives.items())
    return pairs


def plot_estimates(results: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    for num_bits, subset in results.groupby("num_bits"):
        axis.scatter(
            subset["exact_cosine"],
            subset["estimated_cosine"],
            alpha=0.65,
            label=f"{num_bits} bits",
        )

    axis.plot([0, 1], [0, 1], "--", label="Ideal estimate")
    axis.set(
        xlabel="Exact cosine similarity",
        ylabel="SimHash-estimated cosine similarity",
        title="SimHash estimate versus exact cosine similarity",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.15, 1.02)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)


def plot_errors(summary: pd.DataFrame, output: Path) -> None:
    pivot = summary.pivot(
        index="num_bits",
        columns="pair_type",
        values="mean_absolute_error",
    ).reindex(columns=["all", "similar", "dissimilar"])

    figure, axis = plt.subplots(figsize=(8, 6))
    pivot.plot(kind="bar", ax=axis)
    axis.set(
        xlabel="SimHash fingerprint length",
        ylabel="Mean absolute error",
        title="SimHash estimation error by fingerprint length",
    )
    axis.tick_params(axis="x", rotation=0)
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(title="Pair type")
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)


def main() -> None:
    seed = 42
    documents, shingle_sets = prepare_documents(
        dataset=load_20newsgroups(),
        k=5,
        sample_size=1000,
        min_shingles=20,
        sample_seed=seed,
    )
    pairs = choose_pairs(shingle_sets, threshold=0.05, seed=seed)
    selected_indices = sorted({index for i, j, _, _ in pairs for index in (i, j)})

    vectorization_start = perf_counter()
    frequency_vectors = {
        index: term_frequencies(documents[index]) for index in selected_indices
    }
    vectorization_time = perf_counter() - vectorization_start

    fingerprints = {}
    runtime_rows = []

    for num_bits in BIT_SIZES:
        fingerprint_start = perf_counter()
        fingerprints[num_bits] = {
            index: compute_simhash(frequency_vectors[index], num_bits, seed)
            for index in selected_indices
        }
        fingerprint_time = perf_counter() - fingerprint_start
        runtime_rows.append(
            {
                "num_bits": num_bits,
                "documents_hashed": len(selected_indices),
                "vectorization_time_seconds": vectorization_time,
                "fingerprint_time_seconds": fingerprint_time,
                "fingerprint_time_per_document_ms": (
                    1000 * fingerprint_time / len(selected_indices)
                ),
            }
        )

    rows = []
    for i, j, exact_jaccard, pair_type in pairs:
        exact_cosine = cosine_similarity(frequency_vectors[i], frequency_vectors[j])
        for num_bits in BIT_SIZES:
            fingerprint_i = fingerprints[num_bits][i]
            fingerprint_j = fingerprints[num_bits][j]
            estimated_cosine = estimate_cosine_from_simhash(
                fingerprint_i, fingerprint_j, num_bits
            )
            rows.append(
                {
                    "document_i": i,
                    "document_j": j,
                    "pair_type": pair_type,
                    "num_bits": num_bits,
                    "exact_jaccard": exact_jaccard,
                    "exact_cosine": exact_cosine,
                    "hamming_distance": hamming_distance(fingerprint_i, fingerprint_j),
                    "estimated_cosine": estimated_cosine,
                    "absolute_error": abs(estimated_cosine - exact_cosine),
                }
            )

    results = pd.DataFrame(rows)
    summary = calculate_simhash_summary(results, pd.DataFrame(runtime_rows))

    tables_dir = Path("results/tables")
    figures_dir = Path("results/figures/extensions")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        tables_dir / "simhash_comparison.csv",
        index=False,
        float_format="%.6f",
    )
    summary.to_csv(
        tables_dir / "simhash_summary.csv",
        index=False,
        float_format="%.6f",
    )
    plot_estimates(results, figures_dir / "simhash_vs_cosine.png")
    plot_errors(summary, figures_dir / "simhash_error_by_bits.png")
    print("Saved SimHash tables and figures.")


if __name__ == "__main__":
    main()
