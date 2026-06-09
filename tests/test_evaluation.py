import pytest

from near_dup.evaluation import evaluate_candidates


def test_evaluation_metrics():
    metrics = evaluate_candidates(
        candidates={(0, 1), (0, 2)},
        ground_truth={(0, 1): 0.8, (1, 2): 0.7},
    )
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1_score"] == pytest.approx(0.5)


def test_perfect_evaluation():
    ground_truth = {(0, 1): 0.9, (2, 3): 0.8}
    metrics = evaluate_candidates(set(ground_truth), ground_truth)
    assert metrics["precision"] == metrics["recall"] == metrics["f1_score"] == 1.0


def test_empty_evaluation():
    metrics = evaluate_candidates(set(), {})
    assert metrics["true_positives"] == 0
    assert metrics["precision"] == metrics["recall"] == metrics["f1_score"] == 0.0
