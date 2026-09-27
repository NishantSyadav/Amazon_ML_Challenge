import pandas as pd


def show_sample(path, n=30):
    df = pd.read_csv(
        path,
        sep="\t",
        nrows=n
    )

    print("\n" + "=" * 80)
    print(path)
    print("=" * 80)

    print(df.to_string(index=False))


show_sample("dataset/train/train_source1.tsv")
show_sample("dataset/train/train_source2.tsv")
show_sample("dataset/train/train_source3.tsv")