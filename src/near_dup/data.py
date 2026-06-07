import nltk
from nltk.corpus import reuters
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import Bunch


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
