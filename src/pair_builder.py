from typing import Dict, Set

import pandas as pd


REQUIRED_CANDIDATE_COLUMNS = {
    "source1_entity_id",
    "candidate_entity_id",
}


def parse_match_ids(value) -> Set[str]:
    """
    Convert ground-truth matched_entity_ids into a set.
    """

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    return {
        entity_id.strip()
        for entity_id in value.split(",")
        if entity_id.strip()
    }


def validate_candidates(candidate_df: pd.DataFrame) -> None:
    """
    Validate the minimum contract expected from
    the candidate-generation stage.
    """

    missing_columns = (
        REQUIRED_CANDIDATE_COLUMNS
        - set(candidate_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Candidate data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if candidate_df[
        ["source1_entity_id", "candidate_entity_id"]
    ].isna().any().any():

        raise ValueError(
            "Candidate entity IDs contain missing values."
        )

    invalid_candidates = ~candidate_df[
        "candidate_entity_id"
    ].astype(str).str.startswith(("S2-", "S3-"))

    if invalid_candidates.any():
        raise ValueError(
            "candidate_entity_id must contain only "
            "S2- or S3- IDs."
        )


def build_truth_map(
    ground_truth_df: pd.DataFrame,
) -> Dict[str, Set[str]]:
    """
    Build:
        S1 ID -> set of true S2/S3 matches
    """

    required = {
        "source1_entity_id",
        "matched_entity_ids",
    }

    missing = required - set(ground_truth_df.columns)

    if missing:
        raise ValueError(
            f"Ground truth missing columns: "
            f"{sorted(missing)}"
        )

    truth_map = {}

    for row in ground_truth_df.itertuples(index=False):

        truth_map[row.source1_entity_id] = (
            parse_match_ids(row.matched_entity_ids)
        )

    return truth_map


def add_training_labels(
    candidate_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Label candidate pairs.

    label = 1:
        candidate is a true match.

    label = 0:
        candidate is not in the ground truth.
    """

    validate_candidates(candidate_df)

    truth_map = build_truth_map(
        ground_truth_df
    )

    result = candidate_df.copy()

    result["label"] = [
        int(
            candidate_id
            in truth_map.get(
                source1_id,
                set(),
            )
        )
        for source1_id, candidate_id
        in zip(
            result["source1_entity_id"],
            result["candidate_entity_id"],
        )
    ]

    return result


def print_pair_statistics(
    labeled_df: pd.DataFrame,
) -> None:
    """
    Print simple pair-label statistics.
    """

    total = len(labeled_df)

    positives = int(
        labeled_df["label"].sum()
    )

    negatives = total - positives

    positive_rate = (
        positives / total * 100
        if total
        else 0.0
    )

    print("\n" + "=" * 60)
    print("PAIR LABEL STATISTICS")
    print("=" * 60)

    print(f"Candidate pairs : {total:,}")
    print(f"Positive pairs  : {positives:,}")
    print(f"Negative pairs  : {negatives:,}")
    print(
        f"Positive rate   : "
        f"{positive_rate:.4f}%"
    )