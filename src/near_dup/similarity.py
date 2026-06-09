from itertools import combinations


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0

    if len(set_a) > len(set_b):
        set_a, set_b = set_b, set_a

    intersection = sum(value in set_b for value in set_a)
    union = len(set_a) + len(set_b) - intersection
    return intersection / union


def compute_ground_truth(
    shingle_sets: list[set[str]],
    threshold: float,
) -> dict[tuple[int, int], float]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    ground_truth = {}
    for i, j in combinations(range(len(shingle_sets)), 2):
        similarity = jaccard_similarity(shingle_sets[i], shingle_sets[j])
        if similarity >= threshold:
            ground_truth[(i, j)] = similarity

    return ground_truth
