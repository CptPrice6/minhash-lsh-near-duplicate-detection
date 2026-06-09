import pytest
from sklearn.utils import Bunch

from near_dup.experiments import run_experiment_grid


def experiment_arguments():
    return {
        "dataset": Bunch(
            data=[
                "alpha beta gamma delta epsilon",
                "alpha beta gamma delta zeta",
                "one two three four five",
            ]
        ),
        "dataset_name": "synthetic",
        "sample_size": 3,
        "min_shingles": 2,
        "threshold": 0.2,
        "sample_seed": 42,
        "hash_seed": 42,
        "k_values": [2],
        "hash_families": ["linear"],
        "lsh_configs": [{"num_hashes": 10, "num_bands": 5, "rows_per_band": 2}],
    }


def test_experiment_returns_one_row():
    results = run_experiment_grid(**experiment_arguments())
    row = results.iloc[0]
    assert len(results) == 1
    assert row["dataset"] == "synthetic"
    assert row["documents"] == 3
    assert row["sample_seed"] == 42
    assert row["hash_seed"] == 42
    assert 0 <= row["candidate_reduction"] <= 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_name", ""),
        ("sample_size", 1),
        ("min_shingles", 0),
        ("threshold", 1.1),
        ("k_values", []),
        ("hash_families", ["unknown"]),
        ("lsh_configs", []),
    ],
)
def test_experiment_validates_inputs(field, value):
    kwargs = experiment_arguments()
    kwargs[field] = value
    with pytest.raises(ValueError):
        run_experiment_grid(**kwargs)


def test_experiment_validates_lsh_factorisation():
    kwargs = experiment_arguments()
    kwargs["lsh_configs"] = [{"num_hashes": 10, "num_bands": 3, "rows_per_band": 2}]
    with pytest.raises(ValueError):
        run_experiment_grid(**kwargs)


def test_experiment_requires_enough_documents():
    kwargs = experiment_arguments()
    kwargs["sample_size"] = 4
    with pytest.raises(RuntimeError):
        run_experiment_grid(**kwargs)
