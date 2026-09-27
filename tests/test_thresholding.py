import pandas as pd

from src.thresholding import (
    probabilities_to_predictions,
    score_entity_predictions,
    threshold_search,
)


pairs = pd.DataFrame({
    "source1_entity_id": [
        "S1-1",
        "S1-1",
        "S1-1",
        "S1-2",
        "S1-2",
        "S1-3",
    ],

    "candidate_entity_id": [
        "S2-10",
        "S3-20",
        "S2-99",
        "S2-30",
        "S3-88",
        "S2-50",
    ],

    "match_probability": [
        0.97,
        0.92,
        0.20,
        0.89,
        0.30,
        0.10,
    ],
})


ground_truth = pd.DataFrame({
    "source1_entity_id": [
        "S1-1",
        "S1-2",
        "S1-3",
    ],

    "matched_entity_ids": [
        "S2-10,S3-20",
        "S2-30",
        "",
    ],
})


predictions = probabilities_to_predictions(
    pairs,
    threshold=0.80,
)

score = score_entity_predictions(
    predictions,
    ground_truth,
)

print("\nPredictions:")
print(predictions)

print("\nF0.5:", score)

assert abs(score - 1.0) < 1e-9


results = threshold_search(
    pairs,
    ground_truth,
    thresholds=[
        0.20,
        0.50,
        0.80,
        0.95,
    ],
)

print("\nThreshold results:")
print(results)

assert results.iloc[0]["macro_f0.5"] >= 0.99

print("\nALL THRESHOLD TESTS PASSED ✅")