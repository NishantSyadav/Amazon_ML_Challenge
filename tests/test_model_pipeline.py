import pandas as pd

from src.model_pipeline import (
    train_catboost,
    score_candidate_pairs,
    get_feature_importance,
)


train_df = pd.DataFrame({

    "source1_entity_id": [
        "S1-1",
        "S1-1",
        "S1-2",
        "S1-2",
        "S1-3",
        "S1-3",
        "S1-4",
        "S1-4",
        "S1-5",
        "S1-5",
        "S1-6",
        "S1-6",
    ],

    "candidate_entity_id": [
        "S2-1", "S3-1",
        "S2-2", "S3-2",
        "S2-3", "S3-3",
        "S2-4", "S3-4",
        "S2-5", "S3-5",
        "S2-6", "S3-6",
    ],

    "name_similarity": [
        0.97, 0.20,
        0.95, 0.25,
        0.93, 0.30,
        0.91, 0.15,
        0.89, 0.35,
        0.94, 0.18,
    ],

    "address_similarity": [
        0.95, 0.10,
        0.92, 0.20,
        0.90, 0.25,
        0.88, 0.15,
        0.87, 0.30,
        0.91, 0.12,
    ],

    "retrieval_score": [
        0.96, 0.20,
        0.94, 0.30,
        0.92, 0.35,
        0.90, 0.18,
        0.88, 0.40,
        0.93, 0.22,
    ],

    "label": [
        1, 0,
        1, 0,
        1, 0,
        1, 0,
        1, 0,
        1, 0,
    ],
})


val_df = pd.DataFrame({

    "source1_entity_id": [
        "S1-100",
        "S1-100",
        "S1-101",
        "S1-101",
        "S1-102",
        "S1-102",
    ],

    "candidate_entity_id": [
        "S2-100",
        "S3-100",
        "S2-101",
        "S3-101",
        "S2-102",
        "S3-102",
    ],

    "name_similarity": [
        0.96,
        0.22,
        0.90,
        0.28,
        0.94,
        0.15,
    ],

    "address_similarity": [
        0.93,
        0.15,
        0.88,
        0.30,
        0.92,
        0.10,
    ],

    "retrieval_score": [
        0.95,
        0.25,
        0.89,
        0.32,
        0.93,
        0.18,
    ],

    "label": [
        1,
        0,
        1,
        0,
        1,
        0,
    ],
})


model, features = train_catboost(
    train_df=train_df,
    val_df=val_df,
    iterations=100,
    depth=4,
    learning_rate=0.10,
)


scored = score_candidate_pairs(
    model,
    val_df,
    features,
)


print("\nScored pairs:")
print(
    scored[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "match_probability",
        ]
    ]
)


assert "match_probability" in scored.columns

assert scored[
    "match_probability"
].between(
    0,
    1,
).all()


importance = get_feature_importance(
    model,
    features,
)


print("\nFeature importance:")
print(importance)


assert len(importance) == len(features)


print(
    "\nALL MODEL PIPELINE TESTS PASSED ✅"
)