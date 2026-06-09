import random

from near_dup.similarity import jaccard_similarity


def sample_pairs_below_threshold(
    shingle_sets: list[set[str]],
    excluded_pairs: set[tuple[int, int]],
    number_of_pairs: int,
    similarity_threshold: float,
    seed: int = 42,
) -> dict[tuple[int, int], float]:
    if number_of_pairs <= 0:
        raise ValueError("number_of_pairs must be greater than 0")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")
    if len(shingle_sets) < 2:
        raise ValueError("At least two shingle sets are required")

    rng = random.Random(seed)
    sampled = {}
    max_attempts = max(10_000, number_of_pairs * 1_000)

    for _ in range(max_attempts):
        if len(sampled) == number_of_pairs:
            return sampled

        i, j = sorted(rng.sample(range(len(shingle_sets)), 2))
        pair = (i, j)
        if pair in excluded_pairs or pair in sampled:
            continue

        similarity = jaccard_similarity(shingle_sets[i], shingle_sets[j])
        if similarity < similarity_threshold:
            sampled[pair] = similarity

    raise RuntimeError("Could not sample the requested number of document pairs")
