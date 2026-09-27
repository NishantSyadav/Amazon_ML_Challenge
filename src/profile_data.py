import pandas as pd


TRAIN_PATH = "dataset/train"

FILES = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
]


def profile_file(filename):
    path = f"{TRAIN_PATH}/{filename}"

    print("\n" + "=" * 70)
    print(filename)
    print("=" * 70)

    total_rows = 0
    missing_name = 0
    missing_address = 0

    country_counts = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        chunksize=100_000,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    ):
        total_rows += len(chunk)

        missing_name += chunk["business_name"].isna().sum()
        missing_address += chunk["business_address"].isna().sum()

        counts = chunk["country"].value_counts(dropna=False)

        for country, count in counts.items():
            country_counts[country] = country_counts.get(country, 0) + count

    print(f"Total rows: {total_rows}")
    print(f"Missing business names: {missing_name}")
    print(f"Missing addresses: {missing_address}")

    print("\nCountry distribution:")
    for country, count in sorted(
        country_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"{country}: {count}")


for filename in FILES:
    profile_file(filename)