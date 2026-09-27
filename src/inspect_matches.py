import pandas as pd


S1_PATH = "dataset/train/train_source1.tsv"
S2_PATH = "dataset/train/train_source2.tsv"
S3_PATH = "dataset/train/train_source3.tsv"
GT_PATH = "dataset/train/train_ground_truth.tsv"


def find_records(path, ids):
    found = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        chunksize=100_000,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ):
        matches = chunk[chunk["entity_id"].isin(ids)]

        if not matches.empty:
            found.append(matches)

        if sum(len(x) for x in found) >= len(ids):
            break

    if found:
        return pd.concat(found, ignore_index=True)

    return pd.DataFrame(
        columns=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )


# --------------------------------------------------
# 1. Read a few ground-truth mappings
# --------------------------------------------------

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    nrows=10
)

print("\nGROUND TRUTH")
print("=" * 80)
print(gt.to_string(index=False))


# --------------------------------------------------
# 2. Collect IDs
# --------------------------------------------------

s1_ids = set(gt["source1_entity_id"])

matched_ids = set()

for value in gt["matched_entity_ids"].dropna():
    for entity_id in str(value).split(","):
        matched_ids.add(entity_id.strip())


s2_ids = {
    x for x in matched_ids
    if x.startswith("S2-")
}

s3_ids = {
    x for x in matched_ids
    if x.startswith("S3-")
}


# --------------------------------------------------
# 3. Find actual records
# --------------------------------------------------

print("\nReading matching S1 records...")
s1 = find_records(S1_PATH, s1_ids)

print("Reading matching S2 records...")
s2 = find_records(S2_PATH, s2_ids)

print("Reading matching S3 records...")
s3 = find_records(S3_PATH, s3_ids)


# --------------------------------------------------
# 4. Display
# --------------------------------------------------

print("\n\nSOURCE 1")
print("=" * 80)
print(s1.to_string(index=False))

print("\n\nSOURCE 2")
print("=" * 80)
print(s2.to_string(index=False))

print("\n\nSOURCE 3")
print("=" * 80)
print(s3.to_string(index=False))