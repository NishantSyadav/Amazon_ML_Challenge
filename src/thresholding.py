from typing import Iterable
import pandas as pd

from src.evaluate import entity_f05


def probabilities_to_predictions(
    scored_pairs: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:

    required = {
        "source1_entity_id",
        "candidate_entity_id",
        "match_probability",
    }

    missing = required - set(scored_pairs.columns)

    if missing:
        raise ValueError(
            f"Missing scored-pair columns: {sorted(missing)}"
        )

    all_source_ids = (
        scored_pairs["source1_entity_id"]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    selected = scored_pairs[
        scored_pairs["match_probability"] >= threshold
    ].copy()

    if selected.empty:
        return pd.DataFrame({
            "source1_entity_id": all_source_ids,
            "matched_entity_ids": "",
        })

    aggregated = (
        selected
        .groupby("source1_entity_id")["candidate_entity_id"]
        .agg(lambda x: ",".join(dict.fromkeys(x)))
        .reset_index()
        .rename(
            columns={
                "candidate_entity_id": "matched_entity_ids"
            }
        )
    )

    output = pd.DataFrame({
        "source1_entity_id": all_source_ids
    })

    output = output.merge(
        aggregated,
        on="source1_entity_id",
        how="left",
    )

    output["matched_entity_ids"] = (
        output["matched_entity_ids"].fillna("")
    )

    return output


def _parse_ids(value):

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    return {
        item.strip()
        for item in value.split(",")
        if item.strip()
    }


def score_entity_predictions(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> float:

    merged = ground_truth.merge(
        predictions,
        on="source1_entity_id",
        how="left",
        suffixes=("_true", "_pred"),
    )

    merged["matched_entity_ids_pred"] = (
        merged["matched_entity_ids_pred"].fillna("")
    )

    total_score = 0.0

    for row in merged.itertuples(index=False):

        truth = _parse_ids(
            row.matched_entity_ids_true
        )

        prediction = _parse_ids(
            row.matched_entity_ids_pred
        )

        total_score += entity_f05(
            truth,
            prediction,
        )

    return (
        total_score / len(merged)
        if len(merged)
        else 0.0
    )


def threshold_search(
    scored_pairs: pd.DataFrame,
    ground_truth: pd.DataFrame,
    thresholds: Iterable[float],
) -> pd.DataFrame:

    results = []

    for threshold in thresholds:

        predictions = probabilities_to_predictions(
            scored_pairs,
            threshold,
        )

        score = score_entity_predictions(
            predictions,
            ground_truth,
        )

        results.append({
            "threshold": threshold,
            "macro_f0.5": score,
        })

        print(
            f"Threshold {threshold:.3f}"
            f" -> F0.5 {score:.6f}"
        )

    return (
        pd.DataFrame(results)
        .sort_values(
            "macro_f0.5",
            ascending=False,
        )
        .reset_index(drop=True)
    )