import pandas as pd

from near_dup.data import load_20newsgroups


def main():
    print("Loading 20 Newsgroups dataset...")

    dataset = load_20newsgroups()

    print("\nDataset loaded successfully!")
    print(f"Number of documents: {len(dataset.data)}")
    print(f"Number of categories: {len(dataset.target_names)}")

    print("\nCategories:")
    for category in dataset.target_names:
        print(f"- {category}")

    df = pd.DataFrame(
        {
            "text": dataset.data,
            "target": dataset.target,
        }
    )

    df["category"] = df["target"].apply(lambda x: dataset.target_names[x])
    df["text_length"] = df["text"].apply(len)

    print("\nFirst 5 rows:")
    print(df[["category", "text_length"]].head())

    print("\nExample document:")
    print("=" * 80)
    print(df.iloc[0]["text"][:1000])
    print("=" * 80)

    print("\nCategory distribution:")
    print(df["category"].value_counts())


if __name__ == "__main__":
    main()
