import pandas as pd

from src.pair_builder import (
    add_training_labels,
    validate_candidates,
)


# Fake Amazon-style ground truth
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


# Fake candidate generation output
candidates = pd.DataFrame({
    "source1_entity_id": [
        "S1-1",
        "S1-1",
        "S1-1",
        "S1-2",
        "S1-2",
        "S1-3",
    ],

    "candidate_entity_id": [
        "S2-10",   # true
        "S3-20",   # true
        "S2-99",   # false
        "S2-30",   # true
        "S3-88",   # false
        "S2-50",   # false - singleton
    ],

    "retrieval_score": [
        0.95,
        0.91,
        0.60,
        0.93,
        0.55,
        0.40,
    ],
})


validate_candidates(candidates)

result = add_training_labels(
    candidates,
    ground_truth,
)


expected_labels = [
    1,
    1,
    0,
    1,
    0,
    0,
]


assert (
    result["label"].tolist()
    == expected_labels
)


# Make sure existing candidate features survive
assert "retrieval_score" in result.columns


print(result)

print(
    "\nALL PAIR BUILDER TESTS PASSED ✅"
)