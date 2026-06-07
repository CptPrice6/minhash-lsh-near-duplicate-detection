from near_dup.data import load_reuters


def main():
    print("Loading Reuters-21578 dataset...")

    dataset = load_reuters()

    print("\nDataset loaded successfully!")
    print(f"Number of documents: {len(dataset.data)}")
    print(f"Number of categories: {len(dataset.target_names)}")

    print("\nFirst 10 categories:")
    for category in dataset.target_names[:10]:
        print(f"- {category}")

    print("\nExample document:")
    print(f"File ID: {dataset.fileids[0]}")
    print(f"Categories: {dataset.categories[0]}")
    print(f"Document length: {len(dataset.data[0])} characters")
    print("-" * 80)
    print(dataset.data[0][:1000])
    print("-" * 80)


if __name__ == "__main__":
    main()
