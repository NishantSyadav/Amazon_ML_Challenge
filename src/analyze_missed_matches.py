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


def parse_truth(value):
    value = str(value).strip()

    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def sim(a, b, scorer):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return scorer(a, b) / 100.0


def reconstruct_sample_ids(
    split_ids_path,
    sample_size,
):
    ids = pd.read_csv(
        split_ids_path,
        sep="\t",
        dtype=str,
        usecols=["entity_id"],
    )

    if sample_size < len(ids):
        ids = ids.sample(
            n=sample_size,
            random_state=42,
        )

    return set(ids["entity_id"])


def collect_missed_pairs(
    candidate_path,
    ground_truth_path,
    sample_ids,
):

    candidates = pd.read_csv(
        candidate_path,
        sep="\t",
        dtype=str,
        usecols=[
            "source1_entity_id",
            "candidate_entity_id",
        ],
    )

    retrieved = set(
        zip(
            candidates["source1_entity_id"],
            candidates["candidate_entity_id"],
        )
    )

    gt = pd.read_csv(
        ground_truth_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    gt = gt[
        gt["source1_entity_id"].isin(
            sample_ids
        )
    ]

    missed = []

    total_true = 0

    for row in gt.itertuples(index=False):

        for candidate_id in parse_truth(
            row.matched_entity_ids
        ):

            total_true += 1

            pair = (
                row.source1_entity_id,
                candidate_id,
            )

            if pair not in retrieved:

                missed.append({
                    "source1_entity_id":
                        row.source1_entity_id,

                    "candidate_entity_id":
                        candidate_id,
                })

    missed_df = pd.DataFrame(missed)

    print("\n" + "=" * 65)
    print("MISSED TRUE LINKS")
    print("=" * 65)

    print(
        f"Total true links     : "
        f"{total_true:,}"
    )

    print(
        f"Retrieved true links : "
        f"{total_true - len(missed_df):,}"
    )

    print(
        f"Missed true links    : "
        f"{len(missed_df):,}"
    )

    return missed_df


def load_source1_records(
    source1_path,
    s1_ids,
):

    print("\nLoading relevant Source1 records...")

    df = pd.read_csv(
        source1_path,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    return df[
        df["entity_id"].isin(s1_ids)
    ].copy()


def scan_target_records(
    source_path,
    target_ids,
    source_name,
    chunksize=200_000,
):

    print(
        f"\nScanning {source_name} "
        f"for missed truth records..."
    )

    found = []

    rows_scanned = 0

    reader = pd.read_csv(
        source_path,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        chunksize=chunksize,
    )

    for chunk_no, chunk in enumerate(
        reader,
        start=1,
    ):

        rows_scanned += len(chunk)

        matches = chunk[
            chunk["entity_id"].isin(
                target_ids
            )
        ]

        if not matches.empty:
            found.append(matches)

        if chunk_no % 5 == 0:
            print(
                f"  scanned "
                f"{rows_scanned:,} rows | "
                f"found "
                f"{sum(len(x) for x in found):,}"
            )

    if not found:
        return pd.DataFrame()

    result = pd.concat(
        found,
        ignore_index=True,
    )

    print(
        f"{source_name}: found "
        f"{len(result):,} relevant rows."
    )

    return result


def add_similarity_analysis(
    pairs,
    source1,
    targets,
):

    s1 = source1.rename(
        columns={
            "entity_id": "source1_entity_id",
            "business_name": "s1_name",
            "business_address": "s1_address",
            "country": "s1_country",
        }
    )

    target = targets.rename(
        columns={
            "entity_id": "candidate_entity_id",
            "business_name": "candidate_name",
            "business_address": "candidate_address",
            "country": "candidate_country",
        }
    )

    df = (
        pairs
        .merge(
            s1,
            on="source1_entity_id",
            how="left",
        )
        .merge(
            target,
            on="candidate_entity_id",
            how="left",
        )
    )

    print("\nComputing similarity diagnostics...")

    df["name_ratio"] = [
        sim(a, b, fuzz.ratio)
        for a, b in zip(
            df["s1_name"],
            df["candidate_name"],
        )
    ]

    df["name_partial"] = [
        sim(a, b, fuzz.partial_ratio)
        for a, b in zip(
            df["s1_name"],
            df["candidate_name"],
        )
    ]

    df["name_token_set"] = [
        sim(a, b, fuzz.token_set_ratio)
        for a, b in zip(
            df["s1_name"],
            df["candidate_name"],
        )
    ]

    df["name_token_sort"] = [
        sim(a, b, fuzz.token_sort_ratio)
        for a, b in zip(
            df["s1_name"],
            df["candidate_name"],
        )
    ]

    df["address_ratio"] = [
        sim(a, b, fuzz.ratio)
        for a, b in zip(
            df["s1_address"],
            df["candidate_address"],
        )
    ]

    df["address_partial"] = [
        sim(a, b, fuzz.partial_ratio)
        for a, b in zip(
            df["s1_address"],
            df["candidate_address"],
        )
    ]

    df["address_token_set"] = [
        sim(a, b, fuzz.token_set_ratio)
        for a, b in zip(
            df["s1_address"],
            df["candidate_address"],
        )
    ]

    df["max_name"] = df[
        [
            "name_ratio",
            "name_partial",
            "name_token_set",
            "name_token_sort",
        ]
    ].max(axis=1)

    df["max_address"] = df[
        [
            "address_ratio",
            "address_partial",
            "address_token_set",
        ]
    ].max(axis=1)

    return df


def print_diagnostics(df):

    print("\n" + "=" * 65)
    print("MISSED MATCH SIMILARITY ANALYSIS")
    print("=" * 65)

    print(
        f"Analyzed pairs: "
        f"{len(df):,}"
    )

    for column in [
        "max_name",
        "max_address",
    ]:

        print(
            f"\n{column} quantiles:"
        )

        print(
            df[column]
            .quantile(
                [
                    0.10,
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                    0.95,
                ]
            )
        )

    print("\nName similarity coverage:")

    for threshold in [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]:

        pct = (
            df["max_name"]
            .ge(threshold)
            .mean()
        )

        print(
            f"name >= {threshold:.2f}: "
            f"{pct:.2%}"
        )

    print("\nAddress similarity coverage:")

    for threshold in [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]:

        pct = (
            df["max_address"]
            .ge(threshold)
            .mean()
        )

        print(
            f"address >= {threshold:.2f}: "
            f"{pct:.2%}"
        )

    print("\nCombined retrieval opportunities:")

    for threshold in [
        0.50,
        0.60,
        0.70,
        0.80,
    ]:

        pct = (
            (
                df["max_name"] >= threshold
            )
            |
            (
                df["max_address"] >= threshold
            )
        ).mean()

        print(
            f"name OR address >= "
            f"{threshold:.2f}: "
            f"{pct:.2%}"
        )


def main(args):

    sample_ids = reconstruct_sample_ids(
        args.split_ids,
        args.sample_size,
    )

    missed = collect_missed_pairs(
        args.candidates,
        args.ground_truth,
        sample_ids,
    )

    s1_ids = set(
        missed["source1_entity_id"]
    )

    s2_ids = {
        x
        for x in missed[
            "candidate_entity_id"
        ]
        if str(x).startswith("S2-")
    }

    s3_ids = {
        x
        for x in missed[
            "candidate_entity_id"
        ]
        if str(x).startswith("S3-")
    }

    print(
        f"\nMissed S2 IDs: {len(s2_ids):,}"
    )

    print(
        f"Missed S3 IDs: {len(s3_ids):,}"
    )

    source1 = load_source1_records(
        args.source1,
        s1_ids,
    )

    source2 = scan_target_records(
        args.source2,
        s2_ids,
        "S2",
    )

    source3 = scan_target_records(
        args.source3,
        s3_ids,
        "S3",
    )

    targets = pd.concat(
        [
            source2,
            source3,
        ],
        ignore_index=True,
    )

    analysis = add_similarity_analysis(
        missed,
        source1,
        targets,
    )

    print_diagnostics(
        analysis
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    analysis.to_csv(
        output,
        sep="\t",
        index=False,
    )

    print(
        f"\nDetailed analysis saved to: "
        f"{output}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidates",
        required=True,
    )

    parser.add_argument(
        "--split-ids",
        required=True,
    )

    parser.add_argument(
        "--ground-truth",
        required=True,
    )

    parser.add_argument(
        "--source1",
        required=True,
    )

    parser.add_argument(
        "--source2",
        required=True,
    )

    parser.add_argument(
        "--source3",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
    )

    main(
        parser.parse_args()
    )