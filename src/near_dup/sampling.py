import random

from near_dup.similarity import jaccard_similarity


def sample_pairs_below_threshold(
    shingle_sets: list[set[str]],
    excluded_pairs: set[tuple[int, int]],
    number_of_pairs: int,
    similarity_threshold: float,
    seed: int = 42,
) -> dict[tuple[int, int], float]:
    """Randomly sample unique pairs below a Jaccard threshold."""
    if number_of_pairs <= 0:
        raise ValueError("number_of_pairs must be greater than 0")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")
    if len(shingle_sets) < 2:
        raise ValueError("At least two shingle sets are required")

    rng = random.Random(seed)
    sampled_pairs = {}

    max_attempts = number_of_pairs * 1000
    attempts = 0

    while len(sampled_pairs) < number_of_pairs and attempts < max_attempts:
        i, j = sorted(rng.sample(range(len(shingle_sets)), 2))
        pair = (i, j)
        attempts += 1

        if pair in excluded_pairs or pair in sampled_pairs:
            continue

        similarity = jaccard_similarity(
            shingle_sets[i],
            shingle_sets[j],
        )

        if similarity < similarity_threshold:
            sampled_pairs[pair] = similarity

    if len(sampled_pairs) < number_of_pairs:
        raise RuntimeError("Could not sample the requested number of document pairs.")

    return sampled_pairs
