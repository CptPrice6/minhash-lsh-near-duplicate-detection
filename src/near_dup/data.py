import nltk
from nltk.corpus import reuters
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import Bunch


def load_20newsgroups() -> Bunch:
    return fetch_20newsgroups(
        subset="all",
        shuffle=True,
        random_state=42,
        remove=("headers", "footers", "quotes"),
    )


def load_reuters() -> Bunch:
    try:
        fileids = reuters.fileids()
    except LookupError:
        nltk.download("reuters", quiet=True)
        fileids = reuters.fileids()

    return Bunch(
        data=[reuters.raw(fileid) for fileid in fileids],
        fileids=fileids,
        categories=[reuters.categories(fileid) for fileid in fileids],
        target_names=sorted(reuters.categories()),
    )


def load_wikipedia_sample(
    max_documents: int,
    sample_seed: int = 42,
    shuffle_buffer: int = 10_000,
) -> Bunch:
    if max_documents <= 0:
        raise ValueError("max_documents must be greater than 0")
    if shuffle_buffer <= 0:
        raise ValueError("shuffle_buffer must be greater than 0")

    from datasets import load_dataset

    stream = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    ).shuffle(seed=sample_seed, buffer_size=shuffle_buffer)

    documents = []
    titles = []
    article_ids = []
    urls = []

    for article in stream:
        text = article.get("text", "").strip()
        if not text:
            continue

        documents.append(text)
        titles.append(article.get("title", ""))
        article_ids.append(article.get("id", ""))
        urls.append(article.get("url", ""))

        if len(documents) == max_documents:
            break

    if len(documents) != max_documents:
        raise RuntimeError(
            f"Loaded {len(documents)} Wikipedia articles; "
            f"{max_documents} were requested"
        )

    return Bunch(
        data=documents,
        titles=titles,
        article_ids=article_ids,
        urls=urls,
    )
