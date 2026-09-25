import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.train_model import (
    prepare_training_data,
    train_classifier,
    predict_probabilities,
)


data = pd.DataFrame({
    "source1_entity_id": [
        "S1-1",
        "S1-1",
        "S1-2",
        "S1-2",
        "S1-3",
        "S1-3",
    ],

    "candidate_entity_id": [
        "S2-10",
        "S2-11",
        "S3-20",
        "S3-21",
        "S2-30",
        "S3-31",
    ],

    "name_similarity": [
        0.98,
        0.20,
        0.91,
        0.35,
        0.88,
        0.10,
    ],

    "address_similarity": [
        0.95,
        0.15,
        0.89,
        0.25,
        0.85,
        0.20,
    ],

    "retrieval_score": [
        0.97,
        0.30,
        0.92,
        0.40,
        0.86,
        0.25,
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


X, y, features = prepare_training_data(data)

assert len(X) == 6
assert len(y) == 6

assert "label" not in features
assert "source1_entity_id" not in features
assert "candidate_entity_id" not in features


model = LogisticRegression()

model, features = train_classifier(
    data,
    model,
)

probabilities = predict_probabilities(
    model,
    data,
    features,
)

assert len(probabilities) == 6

assert all(
    0 <= p <= 1
    for p in probabilities
)


print("\nFeatures:", features)
print("Probabilities:", probabilities)

print("\nALL TRAIN MODEL TESTS PASSED ✅")