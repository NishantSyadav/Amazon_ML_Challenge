# blocking-quality metrics: candidate recall, candidates-per-S1, per-source breakdown.
#
# blocking recall = true matches that show up in our candidate set / total true matches
#
# this is the number that matters most for the retrieval side - it's a hard
# ceiling on what Om's model can ever recover. a match that never made it
# into the candidate set can't be scored later, no matter how good the model is.
import pandas as pd


def _explode_ground_truth(ground_truth: pd.DataFrame) -> pd.DataFrame:
    gt = ground_truth.copy()
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
    gt["matched_entity_id"] = gt["matched_entity_ids"].str.split(",")
    gt = gt.explode("matched_entity_id")
    gt = gt[gt["matched_entity_id"] != ""]
    return gt[["source1_entity_id", "matched_entity_id"]]


def candidate_recall(candidates: pd.DataFrame, ground_truth: pd.DataFrame) -> dict:
    true_pairs = _explode_ground_truth(ground_truth)
    total_true = len(true_pairs)
    if total_true == 0:
        return {"blocking_recall": None, "total_true_links": 0}

    cand_pairs = set(zip(candidates["source1_entity_id"], candidates["candidate_entity_id"]))
    found = list(zip(true_pairs["source1_entity_id"], true_pairs["matched_entity_id"]))
    found = pd.Series(found).isin(cand_pairs)

    return {
        "blocking_recall": found.mean(),
        "total_true_links": total_true,
        "true_links_found": int(found.sum()),
        "true_links_missed": int((~found).sum()),
    }


def candidate_recall_by_source(candidates: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    true_pairs = _explode_ground_truth(ground_truth)
    true_pairs["source"] = true_pairs["matched_entity_id"].str.split("-").str[0]
    cand_pairs = set(zip(candidates["source1_entity_id"], candidates["candidate_entity_id"]))
    found = list(zip(true_pairs["source1_entity_id"], true_pairs["matched_entity_id"]))
    true_pairs["found"] = pd.Series(found).isin(cand_pairs).values
    return (
        true_pairs.groupby("source")["found"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "recall", "count": "n_true_links"})
    )


def avg_candidates_per_s1(candidates: pd.DataFrame, all_s1_ids) -> dict:
    counts = candidates.groupby("source1_entity_id").size().reindex(all_s1_ids, fill_value=0)
    return {
        "avg_candidates_per_s1": counts.mean(),
        "median_candidates_per_s1": counts.median(),
        "max_candidates_per_s1": counts.max(),
        "s1_with_zero_candidates": int((counts == 0).sum()),
    }
