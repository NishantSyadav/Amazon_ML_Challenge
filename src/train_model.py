from typing import List, Optional, Tuple
import pandas as pd


LABEL_COLUMN = "label"

NON_FEATURE_COLUMNS = {
    "source1_entity_id",
    "candidate_entity_id",
    "label",
}


def validate_training_data(df: pd.DataFrame) -> None:
    required = {
        "source1_entity_id",
        "candidate_entity_id",
        LABEL_COLUMN,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Training data missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Training dataframe is empty.")

    labels = set(df[LABEL_COLUMN].dropna().unique())

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"Labels must contain only 0/1. Found: {labels}"
        )


def detect_feature_columns(df: pd.DataFrame) -> List[str]:
    features = []

    for column in df.columns:
        if column in NON_FEATURE_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            features.append(column)

    if not features:
        raise ValueError("No numeric features found.")

    return features


def prepare_training_data(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:

    validate_training_data(df)

    if feature_columns is None:
        feature_columns = detect_feature_columns(df)

    missing = set(feature_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing feature columns: {sorted(missing)}"
        )

    X = df[feature_columns].copy()
    y = df[LABEL_COLUMN].astype(int).copy()

    return X, y, feature_columns


def train_classifier(
    df: pd.DataFrame,
    model,
    feature_columns: Optional[List[str]] = None,
):
    X, y, feature_columns = prepare_training_data(
        df,
        feature_columns,
    )

    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    print(f"Training pairs : {len(df):,}")
    print(f"Features       : {len(feature_columns)}")
    print(f"Positive pairs : {int(y.sum()):,}")
    print(f"Negative pairs : {int((y == 0).sum()):,}")

    print("\nFeatures:")
    for feature in feature_columns:
        print(f" - {feature}")

    model.fit(X, y)

    print("\nTraining complete.")

    return model, feature_columns


def predict_probabilities(
    model,
    df: pd.DataFrame,
    feature_columns: List[str],
):
    missing = set(feature_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Prediction data missing features: {sorted(missing)}"
        )

    X = df[feature_columns]

    return model.predict_proba(X)[:, 1]