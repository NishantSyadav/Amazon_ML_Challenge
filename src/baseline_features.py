import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", errors="ignore")
        .decode("ascii")
    )

    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def similarity(a, b, scorer):
    if not a or not b:
        return 0.0

    return scorer(a, b) / 100.0


def length_ratio(a, b):
    if not a or not b:
        return 0.0

    return min(len(a), len(b)) / max(len(a), len(b))


def parse_truth(value):
    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    return {
        x.strip()
        for x in value.split(",")
        if x.strip()
    }


def add_features(df):

    print("Normalizing text...")

    s1_names = [
        normalize_text(x)
        for x in df["s1_business_name"]
    ]

    candidate_names = [
        normalize_text(x)
        for x in df["candidate_business_name"]
    ]

    s1_addresses = [
        normalize_text(x)
        for x in df["s1_business_address"]
    ]

    candidate_addresses = [
        normalize_text(x)
        for x in df["candidate_business_address"]
    ]

    print("Computing name features...")

    df["name_ratio"] = [
        similarity(a, b, fuzz.ratio)
        for a, b in zip(
            s1_names,
            candidate_names,
        )
    ]

    df["name_partial_ratio"] = [
        similarity(a, b, fuzz.partial_ratio)
        for a, b in zip(
            s1_names,
            candidate_names,
        )
    ]

    df["name_token_sort"] = [
        similarity(a, b, fuzz.token_sort_ratio)
        for a, b in zip(
            s1_names,
            candidate_names,
        )
    ]

    df["name_token_set"] = [
        similarity(a, b, fuzz.token_set_ratio)
        for a, b in zip(
            s1_names,
            candidate_names,
        )
    ]

    df["name_exact"] = [
        int(bool(a) and a == b)
        for a, b in zip(
            s1_names,
            candidate_names,
        )
    ]

    df["name_length_ratio"] = [
        length_ratio(a, b)
        for a, b in zip(
            s1_names,
            candidate_names,
        )
    ]

    print("Computing address features...")

    df["address_ratio"] = [
        similarity(a, b, fuzz.ratio)
        for a, b in zip(
            s1_addresses,
            candidate_addresses,
        )
    ]

    df["address_partial_ratio"] = [
        similarity(a, b, fuzz.partial_ratio)
        for a, b in zip(
            s1_addresses,
            candidate_addresses,
        )
    ]

    df["address_token_sort"] = [
        similarity(a, b, fuzz.token_sort_ratio)
        for a, b in zip(
            s1_addresses,
            candidate_addresses,
        )
    ]

    df["address_token_set"] = [
        similarity(a, b, fuzz.token_set_ratio)
        for a, b in zip(
            s1_addresses,
            candidate_addresses,
        )
    ]

    df["address_exact"] = [
        int(bool(a) and a == b)
        for a, b in zip(
            s1_addresses,
            candidate_addresses,
        )
    ]

    df["address_length_ratio"] = [
        length_ratio(a, b)
        for a, b in zip(
            s1_addresses,
            candidate_addresses,
        )
    ]

    df["same_country"] = (
        df["s1_country"]
        .fillna("")
        .astype(str)
        .eq(
            df["candidate_country"]
            .fillna("")
            .astype(str)
        )
        .astype(int)
    )

    df["max_name_similarity"] = (
        df[
            [
                "name_ratio",
                "name_partial_ratio",
                "name_token_sort",
                "name_token_set",
            ]
        ]
        .max(axis=1)
    )

    df["max_address_similarity"] = (
        df[
            [
                "address_ratio",
                "address_partial_ratio",
                "address_token_sort",
                "address_token_set",
            ]
        ]
        .max(axis=1)
    )

    df["combined_similarity"] = (
        0.60 * df["max_name_similarity"]
        + 0.40 * df["max_address_similarity"]
    )

    return df


def add_labels(df, ground_truth_path):

    print("Loading ground truth...")

    gt = pd.read_csv(
        ground_truth_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    relevant_ids = set(
        df["source1_entity_id"]
    )

    gt = gt[
        gt["source1_entity_id"].isin(
            relevant_ids
        )
    ]

    truth_map = {
        row.source1_entity_id:
        parse_truth(row.matched_entity_ids)
        for row in gt.itertuples(index=False)
    }

    print("Creating labels...")

    df["label"] = [
        int(
            candidate_id
            in truth_map.get(
                source1_id,
                set(),
            )
        )
        for source1_id, candidate_id in zip(
            df["source1_entity_id"],
            df["candidate_entity_id"],
        )
    ]

    return df


def main(input_path, ground_truth_path, output_path):

    print(f"Loading: {input_path}")

    df = pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"Candidate pairs loaded: {len(df):,}"
    )

    df = add_features(df)

    df = add_labels(
        df,
        ground_truth_path,
    )

    positives = int(df["label"].sum())
    negatives = len(df) - positives

    print("\n" + "=" * 60)
    print("REAL FEATURE DATASET")
    print("=" * 60)

    print(
        f"Candidate pairs : {len(df):,}"
    )

    print(
        f"Positive pairs  : {positives:,}"
    )

    print(
        f"Negative pairs  : {negatives:,}"
    )

    print(
        f"Positive rate   : "
        f"{positives / len(df):.4%}"
    )

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output,
        sep="\t",
        index=False,
    )

    print(
        f"Saved to        : {output}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--ground-truth",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    main(
        args.input,
        args.ground_truth,
        args.output,
    )