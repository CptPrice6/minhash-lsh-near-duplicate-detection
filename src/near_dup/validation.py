import numpy as np
import pandas as pd

_REQUIRED_COLUMNS = {
    "document_i",
    "document_j",
    "method",
    "pair_type",
    "num_hashes",
    "hash_seed",
    "exact_jaccard",
    "estimated_jaccard",
}


def calculate_error_summary(validation_df: pd.DataFrame) -> pd.DataFrame:
    missing = _REQUIRED_COLUMNS - set(validation_df.columns)
    if missing:
        raise ValueError(f"Validation data is missing columns: {sorted(missing)}")
    if validation_df.empty:
        raise ValueError("Validation data must not be empty")

    rows = []

    for (method, num_hashes), method_group in validation_df.groupby(
        ["method", "num_hashes"], sort=True
    ):
        groups = {
            "all": method_group,
            "similar": method_group[method_group["pair_type"] == "similar"],
            "dissimilar": method_group[method_group["pair_type"] == "dissimilar"],
        }

        for pair_type, subset in groups.items():
            if subset.empty:
                continue

            errors = subset["estimated_jaccard"] - subset["exact_jaccard"]
            seed_mae = (
                subset.assign(absolute_error=errors.abs())
                .groupby("hash_seed")["absolute_error"]
                .mean()
            )
            correlation = np.nan
            if (
                subset["exact_jaccard"].nunique() > 1
                and subset["estimated_jaccard"].nunique() > 1
            ):
                correlation = subset["exact_jaccard"].corr(subset["estimated_jaccard"])

            rows.append(
                {
                    "method": method,
                    "pair_type": pair_type,
                    "num_hashes": int(num_hashes),
                    "num_pairs": len(
                        subset[["document_i", "document_j"]].drop_duplicates()
                    ),
                    "num_hash_seeds": subset["hash_seed"].nunique(),
                    "mean_absolute_error": errors.abs().mean(),
                    "mae_standard_deviation": seed_mae.std(ddof=1),
                    "root_mean_squared_error": np.sqrt(np.mean(errors**2)),
                    "maximum_absolute_error": errors.abs().max(),
                    "correlation": correlation,
                }
            )

    return pd.DataFrame(rows)
