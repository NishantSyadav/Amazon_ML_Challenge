from typing import Iterable, Set, Dict
import pandas as pd


BETA = 0.5
BETA_SQ = BETA ** 2


def parse_ids(value) -> Set[str]:
    """
    Convert comma-separated matched IDs into a Python set.

    Examples:
        "S2-1,S3-5" -> {"S2-1", "S3-5"}
        ""           -> set()
        NaN          -> set()
    """

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if value == "":
        return set()

    return {
        item.strip()
        for item in value.split(",")
        if item.strip()
    }


def entity_f05(
    true_ids: Iterable[str],
    predicted_ids: Iterable[str]
) -> float:
    """
    Compute F0.5 for ONE Source1 entity.
    """

    truth = set(true_ids)
    pred = set(predicted_ids)

    # Correct singleton prediction
    if not truth and not pred:
        return 1.0

    # One side empty but the other is not
    if not truth or not pred:
        return 0.0

    true_positive = len(truth & pred)

    precision = true_positive / len(pred)
    recall = true_positive / len(truth)

    denominator = (
        BETA_SQ * precision
        + recall
    )

    if denominator == 0:
        return 0.0

    return (
        (1 + BETA_SQ)
        * precision
        * recall
        / denominator
    )


def evaluate_predictions(
    ground_truth_path: str,
    predictions_path: str
) -> Dict[str, float]:
    """
    Evaluate a prediction TSV using Amazon's
    entity-level macro F0.5 metric.

    Expected ground-truth columns:
        source1_entity_id
        matched_entity_ids

    Expected prediction columns:
        source1_entity_id
        matched_entity_ids
    """

    gt = pd.read_csv(
        ground_truth_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    pred = pd.read_csv(
        predictions_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    required = {
        "source1_entity_id",
        "matched_entity_ids"
    }

    if not required.issubset(gt.columns):
        raise ValueError(
            f"Invalid ground-truth columns: {gt.columns.tolist()}"
        )

    if not required.issubset(pred.columns):
        raise ValueError(
            f"Invalid prediction columns: {pred.columns.tolist()}"
        )

    # Duplicate protection
    if gt["source1_entity_id"].duplicated().any():
        raise ValueError(
            "Duplicate Source1 IDs found in ground truth."
        )

    if pred["source1_entity_id"].duplicated().any():
        raise ValueError(
            "Duplicate Source1 IDs found in predictions."
        )

    gt_ids = set(gt["source1_entity_id"])
    pred_ids = set(pred["source1_entity_id"])

    missing_ids = gt_ids - pred_ids
    extra_ids = pred_ids - gt_ids

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids):,} Source1 IDs "
            f"are missing from predictions."
        )

    if extra_ids:
        raise ValueError(
            f"{len(extra_ids):,} unknown Source1 IDs "
            f"exist in predictions."
        )

    merged = gt.merge(
        pred,
        on="source1_entity_id",
        suffixes=("_true", "_pred")
    )

    macro_score_sum = 0.0

    total_tp = 0
    total_predicted = 0
    total_true = 0

    for row in merged.itertuples(index=False):

        truth = parse_ids(
            row.matched_entity_ids_true
        )

        prediction = parse_ids(
            row.matched_entity_ids_pred
        )

        score = entity_f05(
            truth,
            prediction
        )

        macro_score_sum += score

        total_tp += len(truth & prediction)
        total_predicted += len(prediction)
        total_true += len(truth)

    macro_f05 = (
        macro_score_sum / len(merged)
        if len(merged)
        else 0.0
    )

    micro_precision = (
        total_tp / total_predicted
        if total_predicted
        else 0.0
    )

    micro_recall = (
        total_tp / total_true
        if total_true
        else 0.0
    )

    return {
        "macro_f0.5": macro_f05,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "entities": len(merged)
    }