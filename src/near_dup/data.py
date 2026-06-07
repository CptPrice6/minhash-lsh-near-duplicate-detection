import nltk
from nltk.corpus import reuters
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import Bunch
from datasets import load_dataset


def load_20newsgroups():
    dataset = fetch_20newsgroups(
        subset="all",
        shuffle=True,
        random_state=42,
        remove=("headers", "footers", "quotes"),
    )

    return dataset


def load_reuters() -> Bunch:
    try:
        fileids = reuters.fileids()
    except LookupError:
        print("Reuters corpus not found. Downloading it with NLTK...")
        nltk.download("reuters", quiet=False)
        fileids = reuters.fileids()

    documents = [reuters.raw(fileid) for fileid in fileids]
    document_categories = [reuters.categories(fileid) for fileid in fileids]
    target_names = sorted(reuters.categories())

    return Bunch(
        data=documents,
        fileids=fileids,
        categories=document_categories,
        target_names=target_names,
    )


def load_wikipedia_sample(
    max_documents: int = 2000,
    seed: int = 42,
    shuffle_buffer: int = 10_000,
) -> Bunch:
    if max_documents <= 0:
        raise ValueError("max_documents must be greater than 0")

    if shuffle_buffer <= 0:
        raise ValueError("shuffle_buffer must be greater than 0")

    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    dataset = dataset.shuffle(
        seed=seed,
        buffer_size=shuffle_buffer,
    )

    documents = []
    titles = []
    article_ids = []
    urls = []

    for article in dataset:
        text = article.get("text", "").strip()

        if not text:
            continue

        documents.append(text)
        titles.append(article.get("title", ""))
        article_ids.append(article.get("id", ""))
        urls.append(article.get("url", ""))

        if len(documents) == max_documents:
            break

    if len(documents) < max_documents:
        raise RuntimeError(
            f"Only {len(documents)} Wikipedia articles were loaded, "
            f"but {max_documents} were requested."
        )

    return Bunch(
        data=documents,
        titles=titles,
        article_ids=article_ids,
        urls=urls,
    )
