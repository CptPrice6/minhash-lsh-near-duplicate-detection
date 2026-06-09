def evaluate_candidates(
    candidates: set[tuple[int, int]],
    ground_truth: dict[tuple[int, int], float],
) -> dict[str, float | int]:
    true_pairs = set(ground_truth)
    true_positives = candidates & true_pairs
    false_positives = candidates - true_pairs
    false_negatives = true_pairs - candidates

    precision = len(true_positives) / len(candidates) if candidates else 0.0
    recall = len(true_positives) / len(true_pairs) if true_pairs else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )

    return {
        "true_positives": len(true_positives),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }
