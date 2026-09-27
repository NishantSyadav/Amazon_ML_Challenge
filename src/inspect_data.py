import pandas as pd

TRAIN_PATH = "dataset/train"

files = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv"
]

for file in files:
    path = f"{TRAIN_PATH}/{file}"

    print("\n" + "=" * 70)
    print(file)
    print("=" * 70)

    df = pd.read_csv(path, sep="\t", nrows=5)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.to_string(index=False))

    print("\nShape information:")
    print("Columns:", len(df.columns))