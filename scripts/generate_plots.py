from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")
RESULT_FILES = {
    "20_newsgroups": TABLES_DIR / "final_20_newsgroups_results.csv",
    "reuters": TABLES_DIR / "final_reuters_results.csv",
}
DATASET_LABELS = {
    "20_newsgroups": "20 Newsgroups",
    "reuters": "Reuters-21578",
}


def save_figure(figure: plt.Figure, output: Path) -> None:
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)


def load_results() -> dict[str, pd.DataFrame]:
    results = {}
    for dataset, path in RESULT_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing results file: {path}")
        results[dataset] = pd.read_csv(path)
    return results


def configuration_labels(df: pd.DataFrame) -> pd.Series:
    return (
        "k="
        + df["k"].astype(str)
        + " | "
        + df["hash_family"]
        + " | h="
        + df["num_hashes"].astype(str)
        + ", b="
        + df["num_bands"].astype(str)
        + ", r="
        + df["rows_per_band"].astype(str)
    )


def plot_precision_recall(df: pd.DataFrame, label: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    for family, subset in df.groupby("hash_family"):
        axis.scatter(subset["recall"], subset["precision"], alpha=0.7, label=family)

    axis.set(
        xlabel="Recall",
        ylabel="Precision",
        title=f"Precision-recall trade-off - {label}",
    )
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, alpha=0.3)
    axis.legend(title="Hash family")
    save_figure(figure, output)


def plot_effect_of_k(df: pd.DataFrame, label: str, output: Path) -> None:
    subset = df[
        (df["hash_family"] == "murmur")
        & (df["num_hashes"] == 200)
        & (df["num_bands"] == 50)
        & (df["rows_per_band"] == 4)
    ].sort_values("k")

    if len(subset) != 3:
        raise RuntimeError("Expected one controlled row for each k value")

    x = np.arange(len(subset))
    width = 0.25
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.bar(x - width, subset["precision"], width, label="Precision")
    axis.bar(x, subset["recall"], width, label="Recall")
    axis.bar(x + width, subset["f1_score"], width, label="F1-score")
    axis.set_xticks(x, subset["k"])
    axis.set(
        xlabel="Shingle size k",
        ylabel="Score",
        title=(f"Effect of shingle size - {label}\nFixed: murmur, h=200, b=50, r=4"),
    )
    axis.set_ylim(0, 1.05)
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    save_figure(figure, output)


def plot_lsh_tradeoff(df: pd.DataFrame, label: str, output: Path) -> None:
    configs = [(20, 10), (40, 5), (50, 4), (100, 2)]
    subset = df[
        (df["k"] == 5) & (df["hash_family"] == "murmur") & (df["num_hashes"] == 200)
    ].copy()
    subset = subset[
        subset.apply(
            lambda row: (row["num_bands"], row["rows_per_band"]) in configs, axis=1
        )
    ]
    subset["order"] = subset.apply(
        lambda row: configs.index((row["num_bands"], row["rows_per_band"])), axis=1
    )
    subset = subset.sort_values("order")
    if len(subset) != len(configs):
        raise RuntimeError("Expected one row for each controlled LSH configuration")

    labels = [f"b={b}, r={r}" for b, r in configs]
    x = np.arange(len(subset))
    width = 0.25
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.bar(x - width, subset["precision"], width, label="Precision")
    axis.bar(x, subset["recall"], width, label="Recall")
    axis.bar(x + width, subset["f1_score"], width, label="F1-score")
    axis.set_xticks(x, labels)
    axis.set(
        xlabel="LSH configuration",
        ylabel="Score",
        title=(f"LSH banding trade-off - {label}\nFixed: k=5, murmur, h=200"),
    )
    axis.set_ylim(0, 1.05)
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    save_figure(figure, output)


def plot_top_configurations(df: pd.DataFrame, label: str, output: Path) -> None:
    top = df.nlargest(10, "f1_score").sort_values("f1_score").copy()
    top["configuration"] = configuration_labels(top)

    figure, axis = plt.subplots(figsize=(11, 7))
    bars = axis.barh(top["configuration"], top["f1_score"])
    axis.set(
        xlabel="F1-score", ylabel="Configuration", title=f"Top configurations - {label}"
    )
    axis.set_xlim(0, 1)
    for bar, value in zip(bars, top["f1_score"]):
        axis.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
        )
    save_figure(figure, output)


def plot_cross_dataset(best: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    labels = best["dataset"].map(DATASET_LABELS)
    bars = axis.bar(labels, best["f1_score"])
    axis.set(
        ylabel="Best observed candidate F1-score",
        title="Best observed configuration by dataset",
    )
    axis.set_ylim(0, 1)
    for bar, value in zip(bars, best["f1_score"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
        )
    save_figure(figure, output)


def plot_best_f1_by_hashes(results: dict[str, pd.DataFrame], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))

    for dataset, df in results.items():
        best = df.groupby("num_hashes")["f1_score"].max().sort_index()
        axis.plot(
            best.index,
            best.values,
            marker="o",
            label=DATASET_LABELS[dataset],
        )

    axis.set(
        xlabel="Number of MinHash values",
        ylabel="Best observed candidate F1-score",
        title="Best observed F1-score by signature length",
    )
    axis.set_xticks([50, 100, 200])
    axis.set_ylim(0, 1)
    axis.grid(True, alpha=0.3)
    axis.legend(title="Dataset")
    save_figure(figure, output)


def plot_f1_vs_candidates(results: dict[str, pd.DataFrame], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))

    for dataset, df in results.items():
        axis.scatter(
            df["lsh_candidate_pairs"],
            df["f1_score"],
            alpha=0.6,
            label=DATASET_LABELS[dataset],
        )

    axis.set_xscale("log")
    axis.set(
        xlabel="LSH candidate pairs (log scale)",
        ylabel="F1-score",
        title="Detection quality versus candidate-set size",
    )
    axis.set_ylim(0, 1)
    axis.grid(True, alpha=0.3)
    axis.legend(title="Dataset")

    save_figure(figure, output)


def main() -> None:
    results = load_results()
    best_rows = []

    for dataset, df in results.items():
        label = DATASET_LABELS[dataset]
        output_dir = FIGURES_DIR / f"final_{dataset}"
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_precision_recall(df, label, output_dir / "precision_recall_tradeoff.png")
        plot_effect_of_k(df, label, output_dir / "effect_of_k.png")
        plot_lsh_tradeoff(df, label, output_dir / "lsh_banding_tradeoff.png")
        plot_top_configurations(df, label, output_dir / "top_10_f1_configurations.png")

        best_row = df.loc[df["f1_score"].idxmax()].copy()
        best_row["dataset"] = dataset
        best_rows.append(best_row)

    best = pd.DataFrame(best_rows).reset_index(drop=True)
    best.to_csv(
        TABLES_DIR / "final_best_configurations.csv", index=False, float_format="%.6f"
    )

    comparison_dir = FIGURES_DIR / "final_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    plot_cross_dataset(best, comparison_dir / "best_f1_by_dataset.png")
    plot_best_f1_by_hashes(results, comparison_dir / "best_f1_by_signature_length.png")
    plot_f1_vs_candidates(results, comparison_dir / "f1_vs_candidate_pairs.png")

    print("Generated final experiment plots.")


if __name__ == "__main__":
    main()
