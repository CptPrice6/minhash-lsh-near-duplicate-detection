from pathlib import Path

import numpy as np
import pandas as pd
from datasketch import MinHash

from near_dup.data import load_20newsgroups
from near_dup.minhash import (
    compute_signature_matrix,
    estimate_jaccard_from_signatures,
)
from near_dup.preprocessing import create_shingle_sets, get_filtered_documents
from near_dup.sampling import sample_pairs_below_threshold
from near_dup.similarity import compute_ground_truth


def create_datasketch_minhash(
    shingles: set[str],
    num_perm: int,
    seed: int,
) -> MinHash:
    minhash = MinHash(
        num_perm=num_perm,
        seed=seed,
    )

    for shingle in sorted(shingles):
        minhash.update(shingle.encode("utf-8"))

    return minhash


def calculate_error_summary(
    validation_df: pd.DataFrame,
) -> pd.DataFrame:
    methods = {
        "linear": "linear_estimate",
        "murmur": "murmur_estimate",
        "tabulation": "tabulation_estimate",
        "datasketch": "datasketch_estimate",
    }

    subsets = {
        "all": validation_df,
        "similar": validation_df[validation_df["pair_type"] == "similar"],
        "dissimilar": validation_df[validation_df["pair_type"] == "dissimilar"],
    }

    rows = []

    for pair_type, subset in subsets.items():
        for method, estimate_column in methods.items():
            errors = subset[estimate_column] - subset["exact_jaccard"]

            rows.append(
                {
                    "pair_type": pair_type,
                    "method": method,
                    "number_of_pairs": len(subset),
                    "mean_absolute_error": errors.abs().mean(),
                    "root_mean_squared_error": np.sqrt(np.mean(errors**2)),
                    "maximum_absolute_error": errors.abs().max(),
                }
            )

    return pd.DataFrame(rows)


def main():
    dataset = load_20newsgroups()

    k = 5
    sample_size = 1000
    min_shingles = 20
    num_hashes = 200
    seed = 42

    positive_threshold = 0.05
    max_positive_pairs = 100
    number_of_negative_pairs = 100

    print("Preparing validation documents...")

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
    print(f"Number of permutations: {num_hashes}")

    print("Finding similar document pairs...")

    ground_truth = compute_ground_truth(
        shingle_sets=shingle_sets,
        threshold=positive_threshold,
    )

    positive_pairs = dict(
        sorted(
            ground_truth.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:max_positive_pairs]
    )

    print("Sampling dissimilar document pairs...")

    negative_pairs = sample_pairs_below_threshold(
        shingle_sets=shingle_sets,
        excluded_pairs=set(ground_truth.keys()),
        number_of_pairs=number_of_negative_pairs,
        similarity_threshold=positive_threshold,
        seed=seed,
    )

    validation_pairs: list[tuple[int, int, float, str]] = []

    for (i, j), similarity in positive_pairs.items():
        validation_pairs.append((i, j, similarity, "similar"))

    for (i, j), similarity in negative_pairs.items():
        validation_pairs.append((i, j, similarity, "dissimilar"))

    selected_document_indices = sorted(
        {document_index for i, j, _, _ in validation_pairs for document_index in (i, j)}
    )

    selected_shingle_sets = [shingle_sets[index] for index in selected_document_indices]

    local_index = {
        original_index: selected_index
        for selected_index, original_index in enumerate(selected_document_indices)
    }

    print(
        f"Validation pairs: {len(validation_pairs)} "
        f"({len(positive_pairs)} similar, {len(negative_pairs)} dissimilar)"
    )

    hash_families = [
        "linear",
        "murmur",
        "tabulation",
    ]

    signature_matrices = {}

    for hash_family in hash_families:
        print(f"Computing our {hash_family} signatures...")

        signature_matrices[hash_family] = compute_signature_matrix(
            shingle_sets=selected_shingle_sets,
            num_hashes=num_hashes,
            seed=seed,
            hash_family=hash_family,
        )

    print("Computing datasketch signatures...")

    datasketch_signatures = {
        original_index: create_datasketch_minhash(
            shingles=shingle_sets[original_index],
            num_perm=num_hashes,
            seed=seed,
        )
        for original_index in selected_document_indices
    }

    rows = []

    for i, j, exact_similarity, pair_type in validation_pairs:
        local_i = local_index[i]
        local_j = local_index[j]

        linear_estimate = estimate_jaccard_from_signatures(
            signature_matrices["linear"][local_i],
            signature_matrices["linear"][local_j],
        )

        murmur_estimate = estimate_jaccard_from_signatures(
            signature_matrices["murmur"][local_i],
            signature_matrices["murmur"][local_j],
        )

        tabulation_estimate = estimate_jaccard_from_signatures(
            signature_matrices["tabulation"][local_i],
            signature_matrices["tabulation"][local_j],
        )

        datasketch_estimate = datasketch_signatures[i].jaccard(datasketch_signatures[j])

        rows.append(
            {
                "document_i": i,
                "document_j": j,
                "pair_type": pair_type,
                "exact_jaccard": exact_similarity,
                "linear_estimate": linear_estimate,
                "murmur_estimate": murmur_estimate,
                "tabulation_estimate": tabulation_estimate,
                "datasketch_estimate": datasketch_estimate,
                "linear_absolute_error": abs(linear_estimate - exact_similarity),
                "murmur_absolute_error": abs(murmur_estimate - exact_similarity),
                "tabulation_absolute_error": abs(
                    tabulation_estimate - exact_similarity
                ),
                "datasketch_absolute_error": abs(
                    datasketch_estimate - exact_similarity
                ),
            }
        )

    validation_df = pd.DataFrame(rows)
    summary_df = calculate_error_summary(validation_df)

    validation_df = validation_df.round(6)
    summary_df = summary_df.round(6)

    output_dir = Path("results") / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_path = output_dir / "datasketch_validation.csv"
    summary_path = output_dir / "datasketch_validation_summary.csv"

    validation_df.to_csv(validation_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nValidation completed.")
    print(f"Pair-level results saved to: {validation_path}")
    print(f"Summary saved to: {summary_path}")
    print("\nError summary:")
    print(summary_df)


if __name__ == "__main__":
    main()
