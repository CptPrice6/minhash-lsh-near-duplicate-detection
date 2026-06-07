from itertools import combinations


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two shingle sets."""
    if not set_a and not set_b:
        return 0.0

    union = set_a | set_b
    intersection = set_a & set_b

    return len(intersection) / len(union)


def compute_ground_truth(
    shingle_sets: list[set[str]],
    threshold: float,
) -> dict[tuple[int, int], float]:
    """Find every document pair whose exact Jaccard meets the threshold."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    ground_truth = {}

    for i, j in combinations(range(len(shingle_sets)), 2):
        similarity = jaccard_similarity(
            shingle_sets[i],
            shingle_sets[j],
        )

        if similarity >= threshold:
            ground_truth[(i, j)] = similarity

    return ground_truth
