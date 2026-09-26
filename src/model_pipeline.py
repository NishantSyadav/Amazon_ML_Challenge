from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from catboost import CatBoostClassifier

from src.train_model import (
    detect_feature_columns,
    validate_training_data,
)


ID_COLUMNS = {
    "source1_entity_id",
    "candidate_entity_id",
    "label",
}


def build_catboost_model(
    iterations: int = 500,
    depth: int = 7,
    learning_rate: float = 0.05,
    random_seed: int = 42,
    auto_class_weights: Optional[str] = None,
) -> CatBoostClassifier:
    """
    Create CatBoost binary classifier.

    auto_class_weights can later be:
        None
        "Balanced"

    We keep it configurable because candidate-pair
    class imbalance may be significant.
    """

    return CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_seed,
        auto_class_weights=auto_class_weights,
        verbose=False,
        allow_writing_files=False,
    )


def prepare_feature_columns(
    train_df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> List[str]:
    """
    Determine which numeric feature columns are used
    by the ML model.
    """

    validate_training_data(train_df)

    if feature_columns is None:
        feature_columns = detect_feature_columns(
            train_df
        )

    missing = (
        set(feature_columns)
        - set(train_df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing training features: "
            f"{sorted(missing)}"
        )

    return feature_columns


def train_catboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    iterations: int = 500,
    depth: int = 7,
    learning_rate: float = 0.05,
    random_seed: int = 42,
    auto_class_weights: Optional[str] = None,
) -> Tuple[
    CatBoostClassifier,
    List[str]
]:
    """
    Train CatBoost on already separated train/validation
    candidate pairs.

    IMPORTANT:
    train_df and val_df must already be separated by
    Source1 entity IDs to prevent leakage.
    """

    validate_training_data(train_df)
    validate_training_data(val_df)

    feature_columns = prepare_feature_columns(
        train_df,
        feature_columns,
    )

    # Safety check: all features must also exist in validation
    missing_val_features = (
        set(feature_columns)
        - set(val_df.columns)
    )

    if missing_val_features:
        raise ValueError(
            f"Validation data missing features: "
            f"{sorted(missing_val_features)}"
        )

    # Entity leakage protection
    train_entities = set(
        train_df["source1_entity_id"]
    )

    val_entities = set(
        val_df["source1_entity_id"]
    )

    overlap = train_entities & val_entities

    if overlap:
        raise ValueError(
            f"Entity leakage detected: "
            f"{len(overlap):,} Source1 IDs occur "
            f"in both train and validation."
        )

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        "label"
    ].astype(int)

    X_val = val_df[
        feature_columns
    ]

    y_val = val_df[
        "label"
    ].astype(int)

    print("\n" + "=" * 65)
    print("CATBOOST TRAINING")
    print("=" * 65)

    print(
        f"Train candidate pairs : "
        f"{len(train_df):,}"
    )

    print(
        f"Validation pairs      : "
        f"{len(val_df):,}"
    )

    print(
        f"Train S1 entities     : "
        f"{train_df['source1_entity_id'].nunique():,}"
    )

    print(
        f"Validation S1 entities: "
        f"{val_df['source1_entity_id'].nunique():,}"
    )

    print(
        f"Positive train pairs  : "
        f"{int(y_train.sum()):,}"
    )

    print(
        f"Negative train pairs  : "
        f"{int((y_train == 0).sum()):,}"
    )

    print(
        f"Features              : "
        f"{len(feature_columns)}"
    )

    model = build_catboost_model(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        random_seed=random_seed,
        auto_class_weights=auto_class_weights,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )

    print(
        f"\nBest iteration: "
        f"{model.get_best_iteration()}"
    )

    print("Training complete ✅")

    return model, feature_columns


def score_candidate_pairs(
    model: CatBoostClassifier,
    candidate_df: pd.DataFrame,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Score candidate pairs with probability of being
    the same business entity.
    """

    required = {
        "source1_entity_id",
        "candidate_entity_id",
    }

    missing = (
        required
        - set(candidate_df.columns)
    )

    if missing:
        raise ValueError(
            f"Candidate data missing columns: "
            f"{sorted(missing)}"
        )

    missing_features = (
        set(feature_columns)
        - set(candidate_df.columns)
    )

    if missing_features:
        raise ValueError(
            f"Candidate data missing model features: "
            f"{sorted(missing_features)}"
        )

    scored = candidate_df.copy()

    probabilities = model.predict_proba(
        scored[feature_columns]
    )[:, 1]

    scored[
        "match_probability"
    ] = probabilities

    return scored


def get_feature_importance(
    model: CatBoostClassifier,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Return feature importance table.
    """

    importance = model.get_feature_importance()

    result = pd.DataFrame({
        "feature": feature_columns,
        "importance": importance,
    })

    return (
        result
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def save_model(
    model: CatBoostClassifier,
    output_path: str,
) -> None:
    """
    Save trained CatBoost model.
    """

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(
        str(path)
    )

    print(
        f"Model saved to: {path}"
    )