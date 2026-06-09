import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from near_dup.data import load_20newsgroups, load_reuters
from near_dup.lsh import lsh_candidates
from near_dup.minhash import compute_signature_matrix
from near_dup.preprocessing import prepare_documents
from near_dup.similarity import compute_ground_truth, jaccard_similarity

NUM_HASHES = 200
HASH_FAMILY = "murmur"
CONFIGS = [(20, 10), (40, 5), (50, 4), (100, 2)]
HASH_SEEDS = [11, 23, 42, 73, 101]
TARGETS = np.round(np.arange(0.05, 1.0, 0.05), 2)
BIN_EDGES = np.linspace(0.0, 1.0, 21)

SAMPLE_SEED = 42
CONTROLLED_SAMPLE_SIZE = 1000
NATURAL_SAMPLE_SIZE = 5000
PAIRS_PER_TARGET = 30
NATURAL_RANDOM_PAIRS = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate controlled and natural LSH S-curves."
    )
    parser.add_argument(
        "--dataset",
        choices=["20_newsgroups", "reuters"],
        default="20_newsgroups",
    )
    return parser.parse_args()


def make_controlled_pairs(
    shingle_sets: list[set[str]],
    pairs_per_similarity: int,
    seed: int,
) -> tuple[list[set[str]], pd.DataFrame]:
    required = len(TARGETS) * pairs_per_similarity

    if pairs_per_similarity <= 0:
        raise ValueError("pairs_per_similarity must be greater than 0")
    if len(shingle_sets) < required:
        raise ValueError(f"At least {required} source documents are required")

    rng = np.random.default_rng(seed)
    source_indices = rng.choice(
        len(shingle_sets),
        size=required,
        replace=False,
    )

    controlled_sets: list[set[str]] = []
    rows = []
    pair_id = 0

    for target in TARGETS:
        for _ in range(pairs_per_similarity):
            original = shingle_sets[int(source_indices[pair_id])]
            set_size = len(original)

            retained_count = round(2 * float(target) * set_size / (1 + float(target)))
            retained = set(
                rng.choice(
                    np.array(sorted(original), dtype=object),
                    size=retained_count,
                    replace=False,
                ).tolist()
            )

            noise = {
                f"__noise_{pair_id}_{index}__"
                for index in range(set_size - retained_count)
            }
            modified = retained | noise

            first = len(controlled_sets)
            controlled_sets.extend([original, modified])

            rows.append(
                {
                    "first": first,
                    "second": first + 1,
                    "similarity": jaccard_similarity(original, modified),
                    "group": float(target),
                }
            )
            pair_id += 1

    return controlled_sets, pd.DataFrame(rows)


def sample_natural_pairs(
    shingle_sets: list[set[str]],
    number_of_random_pairs: int,
    seed: int,
) -> pd.DataFrame:
    if number_of_random_pairs <= 0:
        raise ValueError("number_of_random_pairs must be greater than 0")

    print("Finding all natural pairs with Jaccard >= 0.05...")

    similar_pairs = compute_ground_truth(
        shingle_sets=shingle_sets,
        threshold=0.05,
    )

    rows = [
        {
            "first": first,
            "second": second,
            "similarity": similarity,
        }
        for (first, second), similarity in similar_pairs.items()
    ]

    print(f"Natural pairs with Jaccard >= 0.05: {len(similar_pairs):,}")
    print("Sampling dissimilar background pairs...")

    rng = np.random.default_rng(seed)
    num_documents = len(shingle_sets)

    excluded = set(similar_pairs)
    sampled: set[tuple[int, int]] = set()

    while len(sampled) < number_of_random_pairs:
        first = int(rng.integers(0, num_documents))
        second = int(rng.integers(0, num_documents))

        if first == second:
            continue

        pair = (min(first, second), max(first, second))

        if pair in excluded or pair in sampled:
            continue

        sampled.add(pair)

    for first, second in sampled:
        rows.append(
            {
                "first": first,
                "second": second,
                "similarity": jaccard_similarity(
                    shingle_sets[first],
                    shingle_sets[second],
                ),
            }
        )

    pairs = pd.DataFrame(rows)

    pairs["group"] = np.digitize(
        pairs["similarity"].to_numpy(),
        BIN_EDGES[1:-1],
        right=False,
    )

    print("\nNatural pairs by similarity bin:")
    print(pairs.groupby("group").size().rename("num_pairs").to_string())

    return pairs


def evaluate_pairs(
    shingle_sets: list[set[str]],
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    pair_tuples = list(
        zip(
            pairs["first"].astype(int),
            pairs["second"].astype(int),
        )
    )
    groups = pairs["group"].to_numpy()

    seed_rows = []

    for hash_seed in HASH_SEEDS:
        signatures = compute_signature_matrix(
            shingle_sets=shingle_sets,
            num_hashes=NUM_HASHES,
            seed=hash_seed,
            hash_family=HASH_FAMILY,
        )

        for num_bands, rows_per_band in CONFIGS:
            candidates = lsh_candidates(
                signatures,
                num_bands,
                rows_per_band,
            )

            flags = np.fromiter(
                (pair in candidates for pair in pair_tuples),
                dtype=float,
                count=len(pair_tuples),
            )

            for group in np.unique(groups):
                mask = groups == group

                seed_rows.append(
                    {
                        "num_bands": num_bands,
                        "rows_per_band": rows_per_band,
                        "hash_seed": hash_seed,
                        "group": group,
                        "seed_probability": flags[mask].mean(),
                    }
                )

    seed_results = pd.DataFrame(seed_rows)
    summary_rows = []

    for (
        num_bands,
        rows_per_band,
        group,
    ), seed_group in seed_results.groupby(
        ["num_bands", "rows_per_band", "group"],
        sort=True,
    ):
        pair_group = pairs[pairs["group"] == group]
        similarities = pair_group["similarity"].to_numpy(dtype=float)

        theoretical = 1 - (1 - similarities**rows_per_band) ** num_bands

        summary_rows.append(
            {
                "num_hashes": NUM_HASHES,
                "hash_family": HASH_FAMILY,
                "num_bands": int(num_bands),
                "rows_per_band": int(rows_per_band),
                "group": group,
                "mean_similarity": similarities.mean(),
                "num_pairs": len(similarities),
                "num_hash_seeds": seed_group["hash_seed"].nunique(),
                "empirical_probability": seed_group["seed_probability"].mean(),
                "empirical_standard_deviation": seed_group["seed_probability"].std(
                    ddof=1
                ),
                "theoretical_probability": theoretical.mean(),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary["absolute_gap"] = (
        summary["empirical_probability"] - summary["theoretical_probability"]
    ).abs()

    return summary


def plot_curve(
    results: pd.DataFrame,
    title: str,
    output: Path,
    minimum_pairs: int,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 7))
    similarity_grid = np.linspace(0.0, 1.0, 500)

    for num_bands, rows_per_band in CONFIGS:
        theoretical = 1 - (1 - similarity_grid**rows_per_band) ** num_bands

        line = axis.plot(
            similarity_grid,
            theoretical,
            label=f"Theoretical b={num_bands}, r={rows_per_band}",
        )[0]

        subset = results[
            (results["num_bands"] == num_bands)
            & (results["rows_per_band"] == rows_per_band)
            & (results["num_pairs"] >= minimum_pairs)
        ].sort_values("mean_similarity")

        axis.errorbar(
            subset["mean_similarity"],
            subset["empirical_probability"],
            yerr=subset["empirical_standard_deviation"].fillna(0.0),
            fmt="o",
            capsize=3,
            color=line.get_color(),
            label=f"Empirical b={num_bands}, r={rows_per_band}",
        )

    axis.set(
        xlabel="Exact Jaccard similarity",
        ylabel="Probability of becoming an LSH candidate",
        title=title,
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, alpha=0.3)
    axis.legend(
        fontsize=9,
        ncol=2,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )

    figure.tight_layout()
    figure.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()

    dataset = load_20newsgroups() if args.dataset == "20_newsgroups" else load_reuters()
    dataset_label = (
        "20 Newsgroups" if args.dataset == "20_newsgroups" else "Reuters-21578"
    )

    print("Building controlled pairs...")
    _, controlled_source_sets = prepare_documents(
        dataset=dataset,
        k=5,
        sample_size=CONTROLLED_SAMPLE_SIZE,
        min_shingles=20,
        sample_seed=SAMPLE_SEED,
    )
    controlled_sets, controlled_pairs = make_controlled_pairs(
        controlled_source_sets,
        pairs_per_similarity=PAIRS_PER_TARGET,
        seed=SAMPLE_SEED,
    )
    controlled_results = evaluate_pairs(
        controlled_sets,
        controlled_pairs,
    )

    print("Sampling natural pairs...")
    _, natural_sets = prepare_documents(
        dataset=dataset,
        k=5,
        sample_size=NATURAL_SAMPLE_SIZE,
        min_shingles=20,
        sample_seed=SAMPLE_SEED,
    )
    natural_pairs = sample_natural_pairs(
        natural_sets,
        number_of_random_pairs=NATURAL_RANDOM_PAIRS,
        seed=SAMPLE_SEED,
    )
    natural_results = evaluate_pairs(
        natural_sets,
        natural_pairs,
    )

    controlled_results.insert(0, "dataset", args.dataset)
    natural_results.insert(0, "dataset", args.dataset)

    natural_results["bin_lower"] = natural_results["group"].apply(
        lambda value: BIN_EDGES[int(value)]
    )
    natural_results["bin_upper"] = natural_results["group"].apply(
        lambda value: BIN_EDGES[int(value) + 1]
    )

    tables_dir = Path("results/tables")
    figures_dir = Path("results/figures") / f"final_{args.dataset}"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    controlled_table = tables_dir / f"s_curve_controlled_{args.dataset}.csv"
    natural_table = tables_dir / f"s_curve_natural_{args.dataset}.csv"
    controlled_figure = figures_dir / "s_curve_controlled.png"
    natural_figure = figures_dir / "s_curve_natural.png"

    controlled_results.to_csv(
        controlled_table,
        index=False,
        float_format="%.6f",
    )
    natural_results.to_csv(
        natural_table,
        index=False,
        float_format="%.6f",
    )

    plot_curve(
        controlled_results,
        title=f"Controlled LSH S-curves - {dataset_label}",
        output=controlled_figure,
        minimum_pairs=1,
    )
    plot_curve(
        natural_results,
        title=f"Natural-corpus LSH S-curves - {dataset_label}",
        output=natural_figure,
        minimum_pairs=10,
    )

    print(f"Saved {controlled_table}")
    print(f"Saved {natural_table}")
    print(f"Saved {controlled_figure}")
    print(f"Saved {natural_figure}")


if __name__ == "__main__":
    main()
