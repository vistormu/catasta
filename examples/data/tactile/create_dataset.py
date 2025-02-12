from catasta.utils import split_dataset


split_dataset(
    dataset="data",
    task="regression",
    splits=(0.8, 0.1, 0.1),
    destination=".",
    shuffle=True,
    file_based_split=True,
)
