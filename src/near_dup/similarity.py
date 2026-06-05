from itertools import combinations


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0

    union = set_a | set_b

    if len(union) == 0:
        return 0.0

    intersection = set_a & set_b

    return len(intersection) / len(union)


def compute_ground_truth(
    shingle_sets: list[set[str]],
    threshold: float,
) -> dict[tuple[int, int], float]:
    ground_truth = {}

    for i, j in combinations(range(len(shingle_sets)), 2):
        similarity = jaccard_similarity(shingle_sets[i], shingle_sets[j])

        if similarity >= threshold:
            ground_truth[(i, j)] = similarity

    return ground_truth
