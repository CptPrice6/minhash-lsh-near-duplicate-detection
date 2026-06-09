import argparse
from pathlib import Path

from near_dup.data import load_20newsgroups, load_reuters
from near_dup.experiments import run_experiment_grid

LSH_CONFIGS = [
    {"num_hashes": 50, "num_bands": 5, "rows_per_band": 10},
    {"num_hashes": 50, "num_bands": 10, "rows_per_band": 5},
    {"num_hashes": 50, "num_bands": 25, "rows_per_band": 2},
    {"num_hashes": 100, "num_bands": 10, "rows_per_band": 10},
    {"num_hashes": 100, "num_bands": 20, "rows_per_band": 5},
    {"num_hashes": 100, "num_bands": 25, "rows_per_band": 4},
    {"num_hashes": 100, "num_bands": 50, "rows_per_band": 2},
    {"num_hashes": 200, "num_bands": 20, "rows_per_band": 10},
    {"num_hashes": 200, "num_bands": 40, "rows_per_band": 5},
    {"num_hashes": 200, "num_bands": 50, "rows_per_band": 4},
    {"num_hashes": 200, "num_bands": 100, "rows_per_band": 2},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["20_newsgroups", "reuters"],
        default="20_newsgroups",
    )
    parser.add_argument("--sample-size", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_20newsgroups() if args.dataset == "20_newsgroups" else load_reuters()

    results = run_experiment_grid(
        dataset=dataset,
        dataset_name=args.dataset,
        sample_size=args.sample_size,
        min_shingles=20,
        threshold=0.2,
        sample_seed=42,
        hash_seed=42,
        k_values=[3, 5, 7],
        hash_families=["linear", "murmur", "tabulation"],
        lsh_configs=LSH_CONFIGS,
    )

    output = Path("results/tables") / f"final_{args.dataset}_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False, float_format="%.6f")

    print(f"\nSaved {len(results)} rows to {output}")


if __name__ == "__main__":
    main()
