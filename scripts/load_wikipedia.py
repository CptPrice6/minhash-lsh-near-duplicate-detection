import argparse

from near_dup.data import load_wikipedia_sample


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a small sample of English Wikipedia articles."
    )

    parser.add_argument(
        "--max-documents",
        type=int,
        default=100,
        help="Number of Wikipedia articles to load.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("Streaming Wikipedia sample...")
    print(f"Requested documents: {args.max_documents}")

    dataset = load_wikipedia_sample(
        max_documents=args.max_documents,
        seed=42,
    )

    lengths = [len(text) for text in dataset.data]

    print("\nDataset loaded successfully!")
    print(f"Documents loaded: {len(dataset.data)}")
    print(f"Minimum document length: {min(lengths)}")
    print(f"Maximum document length: {max(lengths)}")
    print(f"Average document length: {sum(lengths) / len(lengths):.2f}")

    print("\nExample article:")
    print(f"ID: {dataset.article_ids[0]}")
    print(f"Title: {dataset.titles[0]}")
    print(f"URL: {dataset.urls[0]}")
    print(f"Length: {len(dataset.data[0])} characters")
    print("-" * 80)
    print(dataset.data[0][:1000])
    print("-" * 80)


if __name__ == "__main__":
    main()
