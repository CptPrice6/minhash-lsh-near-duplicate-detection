from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def add_config_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["config"] = (
        "k="
        + df["k"].astype(str)
        + ", "
        + df["hash_family"].astype(str)
        + ", h="
        + df["num_hashes"].astype(str)
        + ", b="
        + df["num_bands"].astype(str)
        + ", r="
        + df["rows_per_band"].astype(str)
    )
    return df


def plot_top_f1(df: pd.DataFrame, output_dir: Path) -> None:
    top = df.sort_values("f1_score", ascending=False).head(10)

    plt.figure(figsize=(12, 6))
    plt.barh(top["config"], top["f1_score"])
    plt.xlabel("F1-score")
    plt.ylabel("Configuration")
    plt.title("Top 10 configurations by F1-score")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    output_path = output_dir / "top_10_f1_configurations.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_precision_recall(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 6))

    markers = {
        3: "o",
        5: "s",
        7: "^",
    }

    for hash_family in sorted(df["hash_family"].unique()):
        for k in sorted(df["k"].unique()):
            subset = df[(df["hash_family"] == hash_family) & (df["k"] == k)]

            if subset.empty:
                continue

            plt.scatter(
                subset["recall"],
                subset["precision"],
                label=f"{hash_family}, k={k}",
                marker=markers.get(k, "o"),
            )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall trade-off")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "precision_recall_tradeoff.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_candidate_pairs(df: pd.DataFrame, output_dir: Path) -> None:
    grouped = df.groupby(["num_hashes", "num_bands", "rows_per_band"], as_index=False)[
        "lsh_candidate_pairs"
    ].mean()

    grouped["config"] = (
        "h="
        + grouped["num_hashes"].astype(str)
        + ", b="
        + grouped["num_bands"].astype(str)
        + ", r="
        + grouped["rows_per_band"].astype(str)
    )

    avg_ground_truth_pairs = df["ground_truth_pairs"].mean()

    plt.figure(figsize=(10, 5))
    plt.bar(grouped["config"], grouped["lsh_candidate_pairs"])
    plt.axhline(
        y=avg_ground_truth_pairs,
        linestyle="--",
        label="Average ground truth pairs",
    )

    plt.xlabel("LSH configuration")
    plt.ylabel("Average candidate pairs")
    plt.title("Average number of LSH candidate pairs")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "candidate_pairs_by_lsh_config.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_runtime_by_hash_family(df: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        df.groupby("hash_family", as_index=False)["lsh_time_seconds"]
        .mean()
        .sort_values("lsh_time_seconds", ascending=False)
    )

    plt.figure(figsize=(7, 5))
    plt.bar(grouped["hash_family"], grouped["lsh_time_seconds"])
    plt.xlabel("Hash family")
    plt.ylabel("Average LSH time seconds")
    plt.title("Average LSH runtime by hash family (banding only)")
    plt.tight_layout()

    output_path = output_dir / "runtime_by_hash_family.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    input_path = Path("results") / "tables" / "experiment_results.csv"
    output_dir = Path("results") / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = add_config_label(df)

    plot_top_f1(df, output_dir)
    plot_precision_recall(df, output_dir)
    plot_candidate_pairs(df, output_dir)
    plot_runtime_by_hash_family(df, output_dir)

    print("Plots generated successfully.")
    print(f"Saved figures in: {output_dir}")


if __name__ == "__main__":
    main()
