from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLES_DIR = Path("results") / "tables"
FIGURES_DIR = Path("results") / "figures"


def load_combined_results() -> pd.DataFrame:
    newsgroups_path = TABLES_DIR / "experiment_results.csv"
    reuters_path = TABLES_DIR / "reuters_experiment_results.csv"

    if not newsgroups_path.exists():
        raise FileNotFoundError(f"Missing file: {newsgroups_path}")

    if not reuters_path.exists():
        raise FileNotFoundError(f"Missing file: {reuters_path}")

    newsgroups_df = pd.read_csv(newsgroups_path)
    reuters_df = pd.read_csv(reuters_path)

    return pd.concat(
        [newsgroups_df, reuters_df],
        ignore_index=True,
    )


def add_config_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["config"] = (
        "h="
        + df["num_hashes"].astype(str)
        + ", b="
        + df["num_bands"].astype(str)
        + ", r="
        + df["rows_per_band"].astype(str)
    )

    return df


def format_dataset_name(name: str) -> str:
    if name == "20_newsgroups":
        return "20 Newsgroups"

    return name.title()


def plot_best_f1_by_dataset(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    best_indices = df.groupby("dataset")["f1_score"].idxmax()
    best = df.loc[best_indices].copy()

    best["dataset_label"] = best["dataset"].apply(format_dataset_name)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(best["dataset_label"], best["f1_score"])

    plt.xlabel("Dataset")
    plt.ylabel("Best F1-score")
    plt.title("Best F1-score by dataset")
    plt.ylim(0, 1)

    for bar, value in zip(bars, best["f1_score"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / "best_f1_by_dataset.png",
        dpi=300,
    )
    plt.close()


def plot_precision_recall_by_dataset(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    markers = {
        "20_newsgroups": "o",
        "reuters": "s",
    }

    plt.figure(figsize=(8, 6))

    for dataset_name in sorted(df["dataset"].unique()):
        subset = df[df["dataset"] == dataset_name]

        plt.scatter(
            subset["recall"],
            subset["precision"],
            marker=markers.get(dataset_name, "o"),
            label=format_dataset_name(dataset_name),
            alpha=0.75,
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall comparison by dataset")
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "precision_recall_by_dataset.png",
        dpi=300,
    )
    plt.close()


def plot_ground_truth_pairs_by_dataset(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    ground_truth = (
        df[
            [
                "dataset",
                "k",
                "ground_truth_pairs",
            ]
        ]
        .drop_duplicates()
        .sort_values(["dataset", "k"])
    )

    pivot = ground_truth.pivot(
        index="k",
        columns="dataset",
        values="ground_truth_pairs",
    )

    x_positions = np.arange(len(pivot.index))
    dataset_names = list(pivot.columns)
    width = 0.35

    plt.figure(figsize=(8, 5))

    for index, dataset_name in enumerate(dataset_names):
        offset = (index - (len(dataset_names) - 1) / 2) * width

        plt.bar(
            x_positions + offset,
            pivot[dataset_name],
            width=width,
            label=format_dataset_name(dataset_name),
        )

    plt.xlabel("Shingle size k")
    plt.ylabel("Ground-truth near-duplicate pairs")
    plt.title("Ground-truth pairs by dataset and shingle size")
    plt.xticks(x_positions, pivot.index)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "ground_truth_pairs_by_dataset.png",
        dpi=300,
    )
    plt.close()


def plot_candidate_pairs_by_dataset(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    grouped = df.groupby(
        [
            "dataset",
            "num_hashes",
            "num_bands",
            "rows_per_band",
        ],
        as_index=False,
    )["lsh_candidate_pairs"].mean()

    grouped = add_config_label(grouped)

    pivot = grouped.pivot(
        index="config",
        columns="dataset",
        values="lsh_candidate_pairs",
    )

    x_positions = np.arange(len(pivot.index))
    dataset_names = list(pivot.columns)
    width = 0.35

    plt.figure(figsize=(11, 6))

    for index, dataset_name in enumerate(dataset_names):
        offset = (index - (len(dataset_names) - 1) / 2) * width

        plt.bar(
            x_positions + offset,
            pivot[dataset_name],
            width=width,
            label=format_dataset_name(dataset_name),
        )

    plt.xlabel("LSH configuration")
    plt.ylabel("Average LSH candidate pairs")
    plt.title("Average candidate pairs by dataset and LSH configuration")
    plt.xticks(
        x_positions,
        pivot.index,
        rotation=30,
        ha="right",
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "candidate_pairs_by_dataset.png",
        dpi=300,
    )
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    results_df = load_combined_results()

    combined_output_path = TABLES_DIR / "combined_experiment_results.csv"
    results_df.to_csv(combined_output_path, index=False)

    plot_best_f1_by_dataset(results_df, FIGURES_DIR)
    plot_precision_recall_by_dataset(results_df, FIGURES_DIR)
    plot_ground_truth_pairs_by_dataset(results_df, FIGURES_DIR)
    plot_candidate_pairs_by_dataset(results_df, FIGURES_DIR)

    print("Dataset comparison plots generated successfully.")
    print(f"Combined results saved to: {combined_output_path}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
