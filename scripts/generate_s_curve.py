from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from near_dup.data import load_20newsgroups
from near_dup.preprocessing import (
    create_shingle_sets,
    get_filtered_documents,
)
from near_dup.similarity import jaccard_similarity
from near_dup.minhash import compute_signature_matrix
from near_dup.lsh import lsh_candidates, theoretical_lsh_probability


def compute_empirical_s_curve(
    shingle_sets: list[set[str]],
    candidates: set[tuple[int, int]],
    bins: np.ndarray,
) -> pd.DataFrame:
    bin_total_counts = np.zeros(len(bins) - 1, dtype=int)
    bin_candidate_counts = np.zeros(len(bins) - 1, dtype=int)

    n = len(shingle_sets)

    for i, j in combinations(range(n), 2):
        sim = jaccard_similarity(shingle_sets[i], shingle_sets[j])
        idx = np.searchsorted(bins, sim, side="right") - 1

        if idx == len(bin_total_counts):
            idx = len(bin_total_counts) - 1

        if 0 <= idx < len(bin_total_counts):
            bin_total_counts[idx] += 1
            if (i, j) in candidates:
                bin_candidate_counts[idx] += 1

    rows = []
    for idx in range(len(bin_total_counts)):
        total = bin_total_counts[idx]
        found = bin_candidate_counts[idx]
        center = (bins[idx] + bins[idx + 1]) / 2
        rows.append(
            {
                "similarity_bin_start": bins[idx],
                "similarity_bin_end": bins[idx + 1],
                "similarity_bin_center": center,
                "total_pairs": total,
                "candidate_pairs": found,
                "empirical_probability": found / total if total > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def plot_s_curve(
    s_curve_df: pd.DataFrame,
    num_bands: int,
    rows_per_band: int,
    output_path: Path,
) -> None:
    s_vals = np.linspace(0, 1, 500)
    theoretical = [
        theoretical_lsh_probability(s, num_bands, rows_per_band) for s in s_vals
    ]

    empirical = s_curve_df.dropna(subset=["empirical_probability"])

    plt.figure(figsize=(8, 6))
    plt.plot(s_vals, theoretical, label="Theoretical probability")
    plt.scatter(
        empirical["similarity_bin_center"],
        empirical["empirical_probability"],
        label="Experimental probability",
    )
    plt.xlabel("Exact Jaccard similarity")
    plt.ylabel("Probability of becoming an LSH candidate")
    plt.title("Theoretical and experimental LSH S-curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    dataset = load_20newsgroups()

    k = 5
    sample_size = 2000
    min_shingles = 20
    num_hashes = 200
    num_bands = 100
    rows_per_band = 2
    hash_family = "murmur"
    seed = 42

    documents = get_filtered_documents(dataset, k, sample_size, min_shingles)
    print(
        f"Documents: {len(documents)}, k={k}, {hash_family}, h={num_hashes}, b={num_bands}, r={rows_per_band}"
    )

    shingle_sets = create_shingle_sets(documents, k=k)

    signatures = compute_signature_matrix(shingle_sets, num_hashes, seed, hash_family)
    candidates = lsh_candidates(signatures, num_bands, rows_per_band)
    print(f"LSH candidate pairs: {len(candidates)}")

    bins = np.linspace(0, 1, 21)
    print("Computing empirical S-curve...")

    s_curve_df = compute_empirical_s_curve(shingle_sets, candidates, bins)
    s_curve_df["theoretical_probability"] = s_curve_df["similarity_bin_center"].apply(
        lambda s: theoretical_lsh_probability(s, num_bands, rows_per_band)
    )

    tables_dir = Path("results") / "tables"
    figures_dir = Path("results") / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    s_curve_df.round(6).to_csv(tables_dir / "s_curve_results.csv", index=False)
    plot_s_curve(
        s_curve_df,
        num_bands,
        rows_per_band,
        figures_dir / "s_curve_theoretical_vs_experimental.png",
    )

    print("Done.")
    print(s_curve_df)


if __name__ == "__main__":
    main()
