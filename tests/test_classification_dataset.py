from catasta.datasets import ImageClassificationDataset


def main() -> None:
    dataset = ImageClassificationDataset(
        root="tests/data/mnist"
    )


if __name__ == "__main__":
    main()
