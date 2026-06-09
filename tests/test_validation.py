import pandas as pd
import pytest

from near_dup.validation import calculate_error_summary


def validation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "document_i": [0, 0, 0, 0],
            "document_j": [1, 1, 1, 1],
            "method": ["linear"] * 4,
            "pair_type": ["similar"] * 4,
            "num_hashes": [50, 50, 100, 100],
            "hash_seed": [11, 23, 11, 23],
            "exact_jaccard": [0.5] * 4,
            "estimated_jaccard": [0.4, 0.6, 0.48, 0.52],
        }
    )


def test_groups_by_signature_length():
    summary = calculate_error_summary(validation_frame())
    similar = summary[summary["pair_type"] == "similar"]
    assert set(similar["num_hashes"]) == {50, 100}
    errors = similar.set_index("num_hashes")["mean_absolute_error"]
    assert errors.loc[100] < errors.loc[50]


def test_rejects_empty_data():
    with pytest.raises(ValueError):
        calculate_error_summary(pd.DataFrame(columns=validation_frame().columns))


def test_rejects_missing_columns():
    with pytest.raises(ValueError):
        calculate_error_summary(pd.DataFrame({"method": ["linear"]}))
