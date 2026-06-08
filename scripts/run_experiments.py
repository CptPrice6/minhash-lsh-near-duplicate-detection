import argparse
from pathlib import Path

from near_dup.data import load_20newsgroups, load_reuters
from near_dup.experiments import run_experiment_grid


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MinHash and LSH experiments.")

    parser.add_argument(
        "--dataset",
        choices=["20_newsgroups", "reuters"],
        default="20_newsgroups",
        help="Dataset to use for the experiments.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.dataset == "20_newsgroups":
        dataset = load_20newsgroups()
        sample_size = 3000
        threshold = 0.2
        output_filename = "final_20_newsgroups_results.csv"
    else:
        dataset = load_reuters()
        sample_size = 3000
        threshold = 0.2
        output_filename = "final_reuters_results.csv"

    min_shingles = 20
    seed = 42

    k_values = [3, 5, 7]

    hash_families = [
        "linear",
        "murmur",
        "tabulation",
    ]

    lsh_configs = [
        # 50 hash functions
        {"num_hashes": 50, "num_bands": 5, "rows_per_band": 10},
        {"num_hashes": 50, "num_bands": 10, "rows_per_band": 5},
        {"num_hashes": 50, "num_bands": 25, "rows_per_band": 2},
        # 100 hash functions
        {"num_hashes": 100, "num_bands": 10, "rows_per_band": 10},
        {"num_hashes": 100, "num_bands": 20, "rows_per_band": 5},
        {"num_hashes": 100, "num_bands": 25, "rows_per_band": 4},
        {"num_hashes": 100, "num_bands": 50, "rows_per_band": 2},
        # 200 hash functions
        {"num_hashes": 200, "num_bands": 20, "rows_per_band": 10},
        {"num_hashes": 200, "num_bands": 40, "rows_per_band": 5},
        {"num_hashes": 200, "num_bands": 50, "rows_per_band": 4},
        {"num_hashes": 200, "num_bands": 100, "rows_per_band": 2},
    ]

    results_df = run_experiment_grid(
        dataset=dataset,
        dataset_name=args.dataset,
        sample_size=sample_size,
        min_shingles=min_shingles,
        threshold=threshold,
        seed=seed,
        k_values=k_values,
        hash_families=hash_families,
        lsh_configs=lsh_configs,
    )

    output_path = Path("results") / "tables" / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_path, index=False)

    print("\nExperiments completed.")
    print(f"Results saved to: {output_path}")
    print("\nResults preview:")
    print(results_df)


if __name__ == "__main__":
    main()
