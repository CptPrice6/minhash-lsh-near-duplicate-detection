from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from datasketch import MinHash

from near_dup.data import load_20newsgroups
from near_dup.minhash import compute_signature_matrix, estimate_jaccard_from_signatures
from near_dup.preprocessing import prepare_documents
from near_dup.sampling import sample_pairs_below_threshold
from near_dup.similarity import compute_ground_truth
from near_dup.validation import calculate_error_summary

NUM_HASH_VALUES = [50, 100, 200]
HASH_SEEDS = [11, 23, 42, 73, 101]
HASH_FAMILIES = ["linear", "murmur", "tabulation"]


def datasketch_signature(shingles: set[str], num_hashes: int, seed: int) -> MinHash:
    signature = MinHash(num_perm=num_hashes, seed=seed)
    for shingle in sorted(shingles):
        signature.update(shingle.encode("utf-8"))
    return signature


def validation_pairs(
    shingle_sets: list[set[str]],
    threshold: float,
    sample_seed: int,
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
        seed=sample_seed,
    )

    pairs = [(i, j, similarity, "similar") for (i, j), similarity in positives]
    pairs.extend(
        (i, j, similarity, "dissimilar") for (i, j), similarity in negatives.items()
    )
    return pairs


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    similar = summary[summary["pair_type"] == "similar"]
    figure, axis = plt.subplots(figsize=(9, 6))

    for method, subset in similar.groupby("method"):
        subset = subset.sort_values("num_hashes")
        axis.errorbar(
            subset["num_hashes"],
            subset["mean_absolute_error"],
            yerr=subset["mae_standard_deviation"].fillna(0),
            marker="o",
            capsize=4,
            label=method,
        )

    axis.set(
        xlabel="Number of MinHash values",
        ylabel="Mean absolute error on similar pairs",
        title="MinHash error by signature length",
    )
    axis.set_xticks(NUM_HASH_VALUES)
    axis.grid(True, alpha=0.3)
    axis.legend(title="Method")
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)


def main() -> None:
    sample_seed = 42
    documents, shingle_sets = prepare_documents(
        dataset=load_20newsgroups(),
        k=5,
        sample_size=1000,
        min_shingles=20,
        sample_seed=sample_seed,
    )
    pairs = validation_pairs(shingle_sets, threshold=0.05, sample_seed=sample_seed)

    selected_indices = sorted({index for i, j, _, _ in pairs for index in (i, j)})
    selected_sets = [shingle_sets[index] for index in selected_indices]
    local_index = {original: local for local, original in enumerate(selected_indices)}
    rows = []

    print(f"Documents: {len(documents)}, validation pairs: {len(pairs)}")

    for num_hashes in NUM_HASH_VALUES:
        for hash_seed in HASH_SEEDS:
            print(f"h={num_hashes}, seed={hash_seed}")
            custom_signatures = {
                family: compute_signature_matrix(
                    selected_sets,
                    num_hashes,
                    seed=hash_seed,
                    hash_family=family,
                )
                for family in HASH_FAMILIES
            }
            datasketch_signatures = {
                index: datasketch_signature(shingle_sets[index], num_hashes, hash_seed)
                for index in selected_indices
            }

            for i, j, exact_similarity, pair_type in pairs:
                for family in HASH_FAMILIES:
                    estimated_jaccard = estimate_jaccard_from_signatures(
                        custom_signatures[family][local_index[i]],
                        custom_signatures[family][local_index[j]],
                    )
                    rows.append(
                        {
                            "document_i": i,
                            "document_j": j,
                            "pair_type": pair_type,
                            "method": family,
                            "num_hashes": num_hashes,
                            "hash_seed": hash_seed,
                            "exact_jaccard": exact_similarity,
                            "estimated_jaccard": estimated_jaccard,
                            "absolute_error": abs(estimated_jaccard - exact_similarity),
                        }
                    )

                datasketch_estimate = datasketch_signatures[i].jaccard(
                    datasketch_signatures[j]
                )
                rows.append(
                    {
                        "document_i": i,
                        "document_j": j,
                        "pair_type": pair_type,
                        "method": "datasketch",
                        "num_hashes": num_hashes,
                        "hash_seed": hash_seed,
                        "exact_jaccard": exact_similarity,
                        "estimated_jaccard": datasketch_estimate,
                        "absolute_error": abs(datasketch_estimate - exact_similarity),
                    }
                )

    validation = pd.DataFrame(rows)
    summary = calculate_error_summary(validation)

    tables_dir = Path("results/tables")
    figure_path = Path("results/figures/validation/minhash_validation_error.png")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    validation.to_csv(
        tables_dir / "datasketch_validation.csv",
        index=False,
        float_format="%.6f",
    )
    summary.to_csv(
        tables_dir / "datasketch_validation_summary.csv",
        index=False,
        float_format="%.6f",
    )
    plot_summary(summary, figure_path)
    print(f"Saved validation tables and {figure_path}")


if __name__ == "__main__":
    main()
