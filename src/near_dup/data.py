from sklearn.datasets import fetch_20newsgroups


def load_20newsgroups():
    dataset = fetch_20newsgroups(
        subset="all",
        shuffle=True,
        random_state=42,
        remove=("headers", "footers", "quotes"),
    )

    return dataset
