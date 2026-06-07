import pytest
from sklearn.utils import Bunch

from near_dup.experiments import run_experiment_grid


def test_run_experiment_grid_returns_expected_row():
    dataset = Bunch(
        data=[
            "alpha beta gamma delta epsilon",
            "alpha beta gamma delta zeta",
            "one two three four five",
        ]
    )

    results = run_experiment_grid(
        dataset=dataset,
        dataset_name="synthetic",
        sample_size=3,
        min_shingles=2,
        threshold=0.2,
        seed=42,
        k_values=[2],
        hash_families=["linear"],
        lsh_configs=[
            {
                "num_hashes": 10,
                "num_bands": 5,
                "rows_per_band": 2,
            }
        ],
    )

    assert len(results) == 1

    row = results.iloc[0]

    assert row["dataset"] == "synthetic"
    assert row["documents"] == 3
    assert row["num_hashes"] == 10
    assert 0.0 <= row["candidate_reduction"] <= 1.0
    assert row["signature_time_seconds"] >= 0.0
    assert row["lsh_time_seconds"] >= 0.0
    assert row["total_runtime_seconds"] >= 0.0


def test_invalid_experiment_threshold():
    dataset = Bunch(data=["one two three", "one two four"])

    with pytest.raises(ValueError):
        run_experiment_grid(
            dataset=dataset,
            dataset_name="synthetic",
            sample_size=2,
            min_shingles=1,
            threshold=1.5,
            seed=42,
            k_values=[2],
            hash_families=["linear"],
            lsh_configs=[
                {
                    "num_hashes": 10,
                    "num_bands": 5,
                    "rows_per_band": 2,
                }
            ],
        )


def test_invalid_experiment_lsh_configuration():
    dataset = Bunch(data=["one two three", "one two four"])

    with pytest.raises(ValueError):
        run_experiment_grid(
            dataset=dataset,
            dataset_name="synthetic",
            sample_size=2,
            min_shingles=1,
            threshold=0.2,
            seed=42,
            k_values=[2],
            hash_families=["linear"],
            lsh_configs=[
                {
                    "num_hashes": 10,
                    "num_bands": 3,
                    "rows_per_band": 2,
                }
            ],
        )


def base_arguments():
    return {
        "dataset": Bunch(
            data=[
                "one two three four",
                "one two three five",
            ]
        ),
        "dataset_name": "synthetic",
        "sample_size": 2,
        "min_shingles": 1,
        "threshold": 0.2,
        "seed": 42,
        "k_values": [2],
        "hash_families": ["linear"],
        "lsh_configs": [
            {
                "num_hashes": 10,
                "num_bands": 5,
                "rows_per_band": 2,
            }
        ],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_name", ""),
        ("sample_size", 1),
        ("min_shingles", 0),
        ("k_values", []),
        ("k_values", [0]),
        ("hash_families", []),
        ("hash_families", ["unknown"]),
        ("lsh_configs", []),
    ],
)
def test_invalid_experiment_inputs(field, value):
    arguments = base_arguments()
    arguments[field] = value

    with pytest.raises(ValueError):
        run_experiment_grid(**arguments)


def test_missing_lsh_configuration_key():
    arguments = base_arguments()
    arguments["lsh_configs"] = [
        {
            "num_hashes": 10,
            "num_bands": 5,
        }
    ]

    with pytest.raises(ValueError):
        run_experiment_grid(**arguments)


def test_nonpositive_lsh_configuration_value():
    arguments = base_arguments()
    arguments["lsh_configs"] = [
        {
            "num_hashes": 10,
            "num_bands": 0,
            "rows_per_band": 2,
        }
    ]

    with pytest.raises(ValueError):
        run_experiment_grid(**arguments)


def test_insufficient_valid_documents():
    arguments = base_arguments()
    arguments["sample_size"] = 3

    with pytest.raises(RuntimeError):
        run_experiment_grid(**arguments)
