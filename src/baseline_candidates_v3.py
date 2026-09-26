import argparse
import heapq
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz


# =========================================================
# CONFIG
# =========================================================

NAME_NGRAM = 4
ADDRESS_NGRAM = 5

# An anchor may occur in at most this many sampled S1 entities.
MAX_POSTING = 6

# Only the rarest anchors of each S1 are indexed.
MAX_NAME_ANCHORS = 6
MAX_ADDRESS_ANCHORS = 8

# Cheap first-stage retrieval.
PRE_K_PER_SOURCE = 80

# Expensive RapidFuzz reranking output.
TOP_K_PER_SOURCE = 50


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", errors="ignore")
        .decode("ascii")
    )

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def normalize_series(series):
    s = (
        series.fillna("")
        .astype(str)
        .str.lower()
    )

    try:
        s = (
            s.str.normalize("NFKD")
            .str.encode(
                "ascii",
                errors="ignore",
            )
            .str.decode("ascii")
        )
    except Exception:
        pass

    s = s.str.replace(
        r"[^a-z0-9]+",
        " ",
        regex=True,
    )

    s = s.str.replace(
        r"\s+",
        " ",
        regex=True,
    )

    return s.str.strip()


def compact(value):
    return value.replace(" ", "")


# =========================================================
# N-GRAMS
# =========================================================

def make_ngrams_from_normalized(
    value,
    n,
):
    value = compact(value)

    if len(value) < n:
        return set()

    return {
        value[i:i + n]
        for i in range(
            len(value) - n + 1
        )
    }


# =========================================================
# LOAD SOURCE1 SAMPLE
# =========================================================

def load_source1_sample(
    source1_path,
    split_ids_path,
    sample_size,
):

    print("\nLoading Source1 sample...")

    ids = pd.read_csv(
        split_ids_path,
        sep="\t",
        dtype=str,
        usecols=["entity_id"],
    )

    if (
        sample_size
        and sample_size < len(ids)
    ):
        ids = ids.sample(
            n=sample_size,
            random_state=42,
        )

    selected = set(
        ids["entity_id"]
    )

    s1 = pd.read_csv(
        source1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    s1 = s1[
        s1["entity_id"].isin(
            selected
        )
    ].copy()

    s1["name_norm"] = normalize_series(
        s1["business_name"]
    )

    s1["address_norm"] = normalize_series(
        s1["business_address"]
    )

    print(
        f"Source1 entities loaded: "
        f"{len(s1):,}"
    )

    return s1


# =========================================================
# BUILD RARE SELECTED ANCHORS
# =========================================================

def build_indexes(s1):

    print(
        "\nCounting Source1 n-gram frequencies..."
    )

    name_grams_by_entity = {}
    address_grams_by_entity = {}

    name_df = Counter()
    address_df = Counter()

    for row in s1.itertuples(
        index=False
    ):

        country = str(
            row.country
        ).strip()

        name_grams = (
            make_ngrams_from_normalized(
                row.name_norm,
                NAME_NGRAM,
            )
        )

        address_grams = (
            make_ngrams_from_normalized(
                row.address_norm,
                ADDRESS_NGRAM,
            )
        )

        name_grams_by_entity[
            row.entity_id
        ] = (
            country,
            name_grams,
        )

        address_grams_by_entity[
            row.entity_id
        ] = (
            country,
            address_grams,
        )

        for gram in name_grams:
            name_df[
                (country, gram)
            ] += 1

        for gram in address_grams:
            address_df[
                (country, gram)
            ] += 1

    print(
        "Selecting rare anchors..."
    )

    name_index = defaultdict(list)
    address_index = defaultdict(list)

    name_anchor_count = {}
    address_anchor_count = {}

    for row in s1.itertuples(
        index=False
    ):

        entity_id = row.entity_id

        country, name_grams = (
            name_grams_by_entity[
                entity_id
            ]
        )

        _, address_grams = (
            address_grams_by_entity[
                entity_id
            ]
        )

        usable_name = [
            gram
            for gram in name_grams
            if name_df[
                (country, gram)
            ] <= MAX_POSTING
        ]

        usable_name.sort(
            key=lambda gram: (
                name_df[
                    (country, gram)
                ],
                gram,
            )
        )

        usable_name = (
            usable_name[
                :MAX_NAME_ANCHORS
            ]
        )

        usable_address = [
            gram
            for gram in address_grams
            if address_df[
                (country, gram)
            ] <= MAX_POSTING
        ]

        usable_address.sort(
            key=lambda gram: (
                address_df[
                    (country, gram)
                ],
                gram,
            )
        )

        usable_address = (
            usable_address[
                :MAX_ADDRESS_ANCHORS
            ]
        )

        name_anchor_count[
            entity_id
        ] = len(
            usable_name
        )

        address_anchor_count[
            entity_id
        ] = len(
            usable_address
        )

        for gram in usable_name:

            name_index[
                (country, gram)
            ].append(
                entity_id
            )

        for gram in usable_address:

            address_index[
                (country, gram)
            ].append(
                entity_id
            )

    # -----------------------------------------------------
    # Exact blocking — preserve our safe V1 candidates
    # -----------------------------------------------------

    exact_name = defaultdict(list)
    exact_address = defaultdict(list)

    for row in s1.itertuples(
        index=False
    ):

        country = str(
            row.country
        ).strip()

        if row.name_norm:

            exact_name[
                (
                    country,
                    row.name_norm,
                )
            ].append(
                row.entity_id
            )

        if row.address_norm:

            exact_address[
                (
                    country,
                    row.address_norm,
                )
            ].append(
                row.entity_id
            )

    print(
        f"Selected name anchors   : "
        f"{sum(len(x) for x in name_index.values()):,}"
    )

    print(
        f"Selected address anchors: "
        f"{sum(len(x) for x in address_index.values()):,}"
    )

    return (
        name_index,
        address_index,
        name_anchor_count,
        address_anchor_count,
        exact_name,
        exact_address,
    )


# =========================================================
# CHEAP PRE-K
# =========================================================

def push_pre_candidate(
    heaps,
    source1_id,
    source_label,
    cheap_score,
    name_hits,
    address_hits,
    row,
):

    key = (
        source1_id,
        source_label,
    )

    # candidate ID ensures deterministic tie-breaking.
    item = (
        cheap_score,
        row.entity_id,
        name_hits,
        address_hits,
        row.business_name,
        row.business_address,
        row.country,
    )

    heap = heaps[key]

    if (
        len(heap)
        < PRE_K_PER_SOURCE
    ):

        heapq.heappush(
            heap,
            item,
        )

    elif (
        item[0]
        > heap[0][0]
    ):

        heapq.heapreplace(
            heap,
            item,
        )


# =========================================================
# SCAN SOURCE
# =========================================================

def scan_source(
    source_path,
    source_label,
    name_index,
    address_index,
    name_anchor_count,
    address_anchor_count,
    exact_name,
    exact_address,
    pre_heaps,
    exact_records,
    chunksize=200_000,
):

    print(
        f"\nScanning {source_label}..."
    )

    total_rows = 0
    anchor_candidates = 0

    reader = pd.read_csv(
        source_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        chunksize=chunksize,
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):

        chunk["name_norm"] = (
            normalize_series(
                chunk[
                    "business_name"
                ]
            )
        )

        chunk["address_norm"] = (
            normalize_series(
                chunk[
                    "business_address"
                ]
            )
        )

        for row in chunk.itertuples(
            index=False
        ):

            total_rows += 1

            country = str(
                row.country
            ).strip()

            # ------------------------------------------------
            # Exact V1-style matches are always retained.
            # ------------------------------------------------

            exact_hits = defaultdict(
                set
            )

            if row.name_norm:

                for source1_id in (
                    exact_name.get(
                        (
                            country,
                            row.name_norm,
                        ),
                        [],
                    )
                ):

                    exact_hits[
                        source1_id
                    ].add(
                        "exact_name"
                    )

            if row.address_norm:

                for source1_id in (
                    exact_address.get(
                        (
                            country,
                            row.address_norm,
                        ),
                        [],
                    )
                ):

                    exact_hits[
                        source1_id
                    ].add(
                        "exact_address"
                    )

            for (
                source1_id,
                methods,
            ) in exact_hits.items():

                pair = (
                    source1_id,
                    row.entity_id,
                )

                exact_records[pair] = {
                    "source1_entity_id":
                        source1_id,

                    "candidate_entity_id":
                        row.entity_id,

                    "candidate_source":
                        source_label,

                    "retrieval_score":
                        1.0,

                    "anchor_score":
                        1.0,

                    "name_anchor_hits":
                        0,

                    "address_anchor_hits":
                        0,

                    "retrieval_method":
                        "+".join(
                            sorted(methods)
                        ),

                    "candidate_business_name":
                        row.business_name,

                    "candidate_business_address":
                        row.business_address,

                    "candidate_country":
                        row.country,
                }

            # ------------------------------------------------
            # Rare selected-anchor hits
            # ------------------------------------------------

            name_grams = (
                make_ngrams_from_normalized(
                    row.name_norm,
                    NAME_NGRAM,
                )
            )

            address_grams = (
                make_ngrams_from_normalized(
                    row.address_norm,
                    ADDRESS_NGRAM,
                )
            )

            hits = defaultdict(
                lambda: [0, 0]
            )

            for gram in name_grams:

                for source1_id in (
                    name_index.get(
                        (
                            country,
                            gram,
                        ),
                        [],
                    )
                ):

                    hits[
                        source1_id
                    ][0] += 1

            for gram in address_grams:

                for source1_id in (
                    address_index.get(
                        (
                            country,
                            gram,
                        ),
                        [],
                    )
                ):

                    hits[
                        source1_id
                    ][1] += 1

            for (
                source1_id,
                counts,
            ) in hits.items():

                name_hits = counts[0]
                address_hits = counts[1]

                name_total = max(
                    name_anchor_count.get(
                        source1_id,
                        0,
                    ),
                    1,
                )

                address_total = max(
                    address_anchor_count.get(
                        source1_id,
                        0,
                    ),
                    1,
                )

                name_coverage = (
                    name_hits
                    / name_total
                )

                address_coverage = (
                    address_hits
                    / address_total
                )

                cheap_score = max(
                    name_coverage,
                    address_coverage,
                )

                # Conservative cheap filter.
                #
                # Keep:
                #   >=2 selected anchor hits
                # OR
                #   >=50% of selected anchors matched.
                #
                # No RapidFuzz here.
                if not (
                    name_hits >= 2
                    or address_hits >= 2
                    or cheap_score >= 0.50
                ):
                    continue

                anchor_candidates += 1

                push_pre_candidate(
                    pre_heaps,
                    source1_id,
                    source_label,
                    cheap_score,
                    name_hits,
                    address_hits,
                    row,
                )

        if (
            chunk_number % 5 == 0
        ):

            stored = sum(
                len(heap)
                for heap in (
                    pre_heaps.values()
                )
            )

            print(
                f"  scanned "
                f"{total_rows:,} rows | "
                f"cheap-qualified "
                f"{anchor_candidates:,} | "
                f"stored Pre-K "
                f"{stored:,} | "
                f"exact "
                f"{len(exact_records):,}"
            )

    print(
        f"{source_label} complete: "
        f"{total_rows:,} rows."
    )


# =========================================================
# FUZZY RERANK ONLY PRE-K
# =========================================================

def fuzzy_scores(
    a,
    b,
):
    if not a or not b:
        return 0.0

    return max(
        fuzz.ratio(a, b),
        fuzz.partial_ratio(a, b),
        fuzz.token_set_ratio(a, b),
    ) / 100.0


def rerank_pre_candidates(
    pre_heaps,
    s1_lookup,
):

    print(
        "\nRapidFuzz reranking bounded Pre-K candidates..."
    )

    final_rows = []

    total_scored = 0

    for (
        source1_id,
        source_label,
    ), heap in pre_heaps.items():

        s1 = s1_lookup[
            source1_id
        ]

        s1_name = (
            normalize_text(
                s1["business_name"]
            )
        )

        s1_address = (
            normalize_text(
                s1[
                    "business_address"
                ]
            )
        )

        scored = []

        for item in heap:

            (
                cheap_score,
                candidate_id,
                name_hits,
                address_hits,
                candidate_name,
                candidate_address,
                candidate_country,
            ) = item

            candidate_name_norm = (
                normalize_text(
                    candidate_name
                )
            )

            candidate_address_norm = (
                normalize_text(
                    candidate_address
                )
            )

            name_score = fuzzy_scores(
                s1_name,
                candidate_name_norm,
            )

            address_score = (
                fuzzy_scores(
                    s1_address,
                    candidate_address_norm,
                )
            )

            fuzzy_score = max(
                name_score,
                address_score,
            )

            # Fuzzy quality dominates;
            # anchor coverage breaks close cases.
            retrieval_score = (
                0.85 * fuzzy_score
                + 0.15 * cheap_score
            )

            scored.append(
                (
                    retrieval_score,
                    {
                        "source1_entity_id":
                            source1_id,

                        "candidate_entity_id":
                            candidate_id,

                        "candidate_source":
                            source_label,

                        "retrieval_score":
                            retrieval_score,

                        "anchor_score":
                            cheap_score,

                        "name_anchor_hits":
                            name_hits,

                        "address_anchor_hits":
                            address_hits,

                        "retrieval_method":
                            "rare_ngram_topk",

                        "candidate_business_name":
                            candidate_name,

                        "candidate_business_address":
                            candidate_address,

                        "candidate_country":
                            candidate_country,
                    },
                )
            )

            total_scored += 1

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        for rank, (
            _,
            record,
        ) in enumerate(
            scored[
                :TOP_K_PER_SOURCE
            ],
            start=1,
        ):

            record[
                "retrieval_rank"
            ] = rank

            final_rows.append(
                record
            )

    print(
        f"RapidFuzz comparisons performed: "
        f"{total_scored:,}"
    )

    return pd.DataFrame(
        final_rows
    )


# =========================================================
# FINAL MERGE
# =========================================================

def build_final_candidates(
    approximate_df,
    exact_records,
    s1_lookup,
):

    exact_rows = []

    for record in (
        exact_records.values()
    ):

        record = record.copy()

        record[
            "retrieval_rank"
        ] = 0

        exact_rows.append(
            record
        )

    exact_df = pd.DataFrame(
        exact_rows
    )

    frames = []

    if not exact_df.empty:
        frames.append(
            exact_df
        )

    if not approximate_df.empty:
        frames.append(
            approximate_df
        )

    if not frames:
        return pd.DataFrame()

    candidates = pd.concat(
        frames,
        ignore_index=True,
    )

    # Add S1 fields for the feature pipeline.
    s1_rows = []

    for source1_id in (
        candidates[
            "source1_entity_id"
        ]
    ):

        row = s1_lookup[
            source1_id
        ]

        s1_rows.append(
            (
                row[
                    "business_name"
                ],
                row[
                    "business_address"
                ],
                row["country"],
            )
        )

    candidates[
        "s1_business_name"
    ] = [
        x[0]
        for x in s1_rows
    ]

    candidates[
        "s1_business_address"
    ] = [
        x[1]
        for x in s1_rows
    ]

    candidates[
        "s1_country"
    ] = [
        x[2]
        for x in s1_rows
    ]

    # Put exact records before approximate duplicates.
    candidates[
        "_exact_priority"
    ] = (
        candidates[
            "retrieval_rank"
        ]
        .eq(0)
        .astype(int)
    )

    candidates = (
        candidates
        .sort_values(
            [
                "_exact_priority",
                "retrieval_score",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .drop_duplicates(
            subset=[
                "source1_entity_id",
                "candidate_entity_id",
            ],
            keep="first",
        )
        .drop(
            columns=[
                "_exact_priority"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return candidates


# =========================================================
# RECALL
# =========================================================

def evaluate_recall(
    candidates,
    ground_truth_path,
    source1_ids,
):

    gt = pd.read_csv(
        ground_truth_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    gt = gt[
        gt[
            "source1_entity_id"
        ].isin(source1_ids)
    ]

    true_pairs = set()
    truth_map = {}

    for row in gt.itertuples(
        index=False
    ):

        truth = {
            x.strip()
            for x in str(
                row.matched_entity_ids
            ).split(",")
            if x.strip()
        }

        truth_map[
            row.source1_entity_id
        ] = truth

        for candidate_id in truth:

            true_pairs.add(
                (
                    row.source1_entity_id,
                    candidate_id,
                )
            )

    retrieved_pairs = set(
        zip(
            candidates[
                "source1_entity_id"
            ],
            candidates[
                "candidate_entity_id"
            ],
        )
    )

    found = (
        true_pairs
        & retrieved_pairs
    )

    candidate_map = (
        candidates
        .groupby(
            "source1_entity_id"
        )[
            "candidate_entity_id"
        ]
        .apply(set)
        .to_dict()
    )

    entities_with_truth = 0
    any_match = 0
    complete = 0

    for (
        source1_id,
        truth,
    ) in truth_map.items():

        if not truth:
            continue

        entities_with_truth += 1

        retrieved = (
            candidate_map.get(
                source1_id,
                set(),
            )
        )

        if truth & retrieved:
            any_match += 1

        if truth.issubset(
            retrieved
        ):
            complete += 1

    print("\n" + "=" * 65)
    print("V3.1 CANDIDATE RECALL")
    print("=" * 65)

    print(
        f"True links              : "
        f"{len(true_pairs):,}"
    )

    print(
        f"Retrieved true links    : "
        f"{len(found):,}"
    )

    print(
        f"Candidate link recall   : "
        f"{len(found) / max(len(true_pairs), 1):.4%}"
    )

    print(
        f"Entities with truth     : "
        f"{entities_with_truth:,}"
    )

    print(
        f"Entities >=1 match      : "
        f"{any_match:,}"
    )

    print(
        f"Any-match entity recall : "
        f"{any_match / max(entities_with_truth, 1):.4%}"
    )

    print(
        f"Fully covered entities  : "
        f"{complete:,}"
    )

    print(
        f"Complete entity recall  : "
        f"{complete / max(entities_with_truth, 1):.4%}"
    )

    print(
        f"Final candidate pairs   : "
        f"{len(candidates):,}"
    )

    print(
        f"Average candidates / S1 : "
        f"{len(candidates) / max(len(source1_ids), 1):.2f}"
    )


# =========================================================
# MAIN
# =========================================================

def main(args):

    s1 = load_source1_sample(
        args.source1,
        args.split_ids,
        args.sample_size,
    )

    (
        name_index,
        address_index,
        name_anchor_count,
        address_anchor_count,
        exact_name,
        exact_address,
    ) = build_indexes(s1)

    s1_lookup = (
        s1.set_index(
            "entity_id"
        )[
            [
                "business_name",
                "business_address",
                "country",
            ]
        ]
        .to_dict("index")
    )

    pre_heaps = defaultdict(
        list
    )

    exact_records = {}

    scan_source(
        args.source2,
        "S2",
        name_index,
        address_index,
        name_anchor_count,
        address_anchor_count,
        exact_name,
        exact_address,
        pre_heaps,
        exact_records,
    )

    scan_source(
        args.source3,
        "S3",
        name_index,
        address_index,
        name_anchor_count,
        address_anchor_count,
        exact_name,
        exact_address,
        pre_heaps,
        exact_records,
    )

    approximate_df = (
        rerank_pre_candidates(
            pre_heaps,
            s1_lookup,
        )
    )

    candidates = (
        build_final_candidates(
            approximate_df,
            exact_records,
            s1_lookup,
        )
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_csv(
        output,
        sep="\t",
        index=False,
    )

    print("\n" + "=" * 65)
    print("V3.1 GENERATION COMPLETE")
    print("=" * 65)

    print(
        f"Source1 sample       : "
        f"{len(s1):,}"
    )

    print(
        f"Final candidate pairs: "
        f"{len(candidates):,}"
    )

    print(
        f"Saved to             : "
        f"{output}"
    )

    evaluate_recall(
        candidates,
        args.ground_truth,
        set(
            s1["entity_id"]
        ),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

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
        "--split-ids",
        required=True,
    )

    parser.add_argument(
        "--ground-truth",
        required=True,
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    main(
        parser.parse_args()
    )