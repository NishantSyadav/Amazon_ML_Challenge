import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.model_pipeline import (
    train_catboost,
    score_candidate_pairs,
    get_feature_importance,
    save_model,
)

from src.thresholding import (
    threshold_search,
    score_entity_predictions,
)


FEATURE_COLUMNS = [
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort",
    "name_token_set",
    "name_exact",
    "name_length_ratio",

    "address_ratio",
    "address_partial_ratio",
    "address_token_sort",
    "address_token_set",
    "address_exact",
    "address_length_ratio",

    "same_country",

    "max_name_similarity",
    "max_address_similarity",
    "combined_similarity",
]


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


def reconstruct_sample_ids(
    split_ids_path,
    sample_size,
):
    """
    Reconstruct the same 5000-ID sample used by
    baseline_candidates.py (random_state=42).

    This is important because entities with zero candidates
    must still remain in validation scoring.
    """

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

    return ids["entity_id"].tolist()


def build_oracle_predictions(
    candidate_df,
    ground_truth,
):
    """
    Best prediction theoretically possible using ONLY
    candidates retrieved by the current blocker.

    The oracle selects every retrieved candidate that is
    actually present in the ground truth.
    """

    candidate_map = (
        candidate_df
        .groupby("source1_entity_id")[
            "candidate_entity_id"
        ]
        .apply(set)
        .to_dict()
    )

    rows = []

    for row in ground_truth.itertuples(
        index=False
    ):

        truth = parse_truth(
            row.matched_entity_ids
        )

        candidates = candidate_map.get(
            row.source1_entity_id,
            set(),
        )

        oracle_matches = (
            truth & candidates
        )

        rows.append({
            "source1_entity_id":
                row.source1_entity_id,

            "matched_entity_ids":
                ",".join(
                    sorted(oracle_matches)
                ),
        })

    return pd.DataFrame(rows)


def main(args):

    print("\nLoading real feature dataset...")

    df = pd.read_csv(
        args.features,
        sep="\t",
        keep_default_na=False,
    )

    print(
        f"Candidate pairs loaded : "
        f"{len(df):,}"
    )

    sample_ids = reconstruct_sample_ids(
        args.split_ids,
        args.sample_size,
    )

    print(
        f"Source1 universe       : "
        f"{len(sample_ids):,}"
    )

    # --------------------------------------------------
    # Internal entity-safe diagnostic split
    # --------------------------------------------------

    train_ids, val_ids = train_test_split(
        sample_ids,
        test_size=0.20,
        random_state=123,
    )

    train_ids = set(train_ids)
    val_ids = set(val_ids)

    train_df = df[
        df["source1_entity_id"].isin(
            train_ids
        )
    ].copy()

    val_df = df[
        df["source1_entity_id"].isin(
            val_ids
        )
    ].copy()

    overlap = (
        set(train_df["source1_entity_id"])
        &
        set(val_df["source1_entity_id"])
    )

    assert len(overlap) == 0

    print("\n" + "=" * 65)
    print("REAL DIAGNOSTIC SPLIT")
    print("=" * 65)

    print(
        f"Train S1 universe : "
        f"{len(train_ids):,}"
    )

    print(
        f"Val S1 universe   : "
        f"{len(val_ids):,}"
    )

    print(
        f"Train pairs       : "
        f"{len(train_df):,}"
    )

    print(
        f"Val pairs         : "
        f"{len(val_df):,}"
    )

    print(
        f"Train positives   : "
        f"{int(train_df['label'].sum()):,}"
    )

    print(
        f"Val positives     : "
        f"{int(val_df['label'].sum()):,}"
    )

    # --------------------------------------------------
    # Ground truth for ALL validation S1 IDs
    # including entities with zero candidates
    # --------------------------------------------------

    print("\nLoading validation ground truth...")

    gt = pd.read_csv(
        args.ground_truth,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    gt_val = gt[
        gt["source1_entity_id"].isin(
            val_ids
        )
    ][
        [
            "source1_entity_id",
            "matched_entity_ids",
        ]
    ].copy()

    print(
        f"Validation GT entities: "
        f"{len(gt_val):,}"
    )

    assert len(gt_val) == len(val_ids)

    # --------------------------------------------------
    # Candidate oracle ceiling
    # --------------------------------------------------

    oracle_predictions = (
        build_oracle_predictions(
            val_df,
            gt_val,
        )
    )

    oracle_score = (
        score_entity_predictions(
            oracle_predictions,
            gt_val,
        )
    )

    print("\n" + "=" * 65)
    print("CANDIDATE ORACLE CEILING")
    print("=" * 65)

    print(
        f"Macro F0.5 oracle : "
        f"{oracle_score:.6f}"
    )

    # --------------------------------------------------
    # Train CatBoost
    # --------------------------------------------------

    model, used_features = (
        train_catboost(
            train_df=train_df,
            val_df=val_df,
            feature_columns=FEATURE_COLUMNS,
            iterations=600,
            depth=7,
            learning_rate=0.05,
            random_seed=42,
            auto_class_weights=None,
        )
    )

    # --------------------------------------------------
    # Score validation candidate pairs
    # --------------------------------------------------

    scored_val = (
        score_candidate_pairs(
            model,
            val_df,
            used_features,
        )
    )

    # --------------------------------------------------
    # Coarse threshold search
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("COARSE THRESHOLD SEARCH")
    print("=" * 65)

    coarse_thresholds = np.arange(
        0.05,
        0.951,
        0.05,
    )

    coarse_results = threshold_search(
        scored_val,
        gt_val,
        coarse_thresholds,
    )

    coarse_best = float(
        coarse_results.iloc[0][
            "threshold"
        ]
    )

    # --------------------------------------------------
    # Fine search around coarse best
    # --------------------------------------------------

    lower = max(
        0.01,
        coarse_best - 0.05,
    )

    upper = min(
        0.99,
        coarse_best + 0.05,
    )

    fine_thresholds = np.arange(
        lower,
        upper + 0.001,
        0.01,
    )

    print("\n" + "=" * 65)
    print("FINE THRESHOLD SEARCH")
    print("=" * 65)

    fine_results = threshold_search(
        scored_val,
        gt_val,
        fine_thresholds,
    )

    best_threshold = float(
        fine_results.iloc[0][
            "threshold"
        ]
    )

    best_score = float(
        fine_results.iloc[0][
            "macro_f0.5"
        ]
    )

    print("\nBest fine thresholds:")
    print(
        fine_results.head(10)
    )

    # --------------------------------------------------
    # Feature importance
    # --------------------------------------------------

    importance = get_feature_importance(
        model,
        used_features,
    )

    print("\n" + "=" * 65)
    print("FEATURE IMPORTANCE")
    print("=" * 65)

    print(
        importance.head(15)
    )

    # --------------------------------------------------
    # Final diagnostic summary
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("REAL AMAZON DIAGNOSTIC RESULT")
    print("=" * 65)

    print(
        f"Best threshold     : "
        f"{best_threshold:.3f}"
    )

    print(
        f"Model macro F0.5   : "
        f"{best_score:.6f}"
    )

    print(
        f"Candidate oracle   : "
        f"{oracle_score:.6f}"
    )

    if oracle_score > 0:
        print(
            f"Model/oracle ratio : "
            f"{best_score / oracle_score:.2%}"
        )

    # --------------------------------------------------
    # Save artifacts
    # --------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    coarse_results.to_csv(
        output_dir
        / "coarse_thresholds.tsv",
        sep="\t",
        index=False,
    )

    fine_results.to_csv(
        output_dir
        / "fine_thresholds.tsv",
        sep="\t",
        index=False,
    )

    importance.to_csv(
        output_dir
        / "feature_importance.tsv",
        sep="\t",
        index=False,
    )

    scored_val.to_csv(
        output_dir
        / "scored_validation_pairs.tsv",
        sep="\t",
        index=False,
    )

    oracle_predictions.to_csv(
        output_dir
        / "oracle_predictions.tsv",
        sep="\t",
        index=False,
    )

    save_model(
        model,
        str(
            output_dir
            / "baseline_v1_catboost.cbm"
        ),
    )

    print(
        "\nREAL AMAZON EXPERIMENT COMPLETE ✅"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features",
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
        "--output-dir",
        default=(
            "artifacts/experiments/"
            "baseline_v1"
        ),
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
    )

    main(
        parser.parse_args()
    )