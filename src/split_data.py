from pathlib import Path
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


def create_source1_split(
    source1_path: str,
    output_dir: str,
    val_size: float = 0.20,
    random_state: int = 42,
):
    """
    Create a reproducible Source1-level train/validation split.

    IMPORTANT:
    We split Source1 ENTITY IDs, not candidate pairs.
    This prevents the same business entity from appearing
    in both training and validation.
    """

    print("\nLoading Source1 IDs...")

    df = pd.read_csv(
        source1_path,
        sep="\t",
        usecols=["entity_id", "country"],
        dtype=str,
    )

    print(f"Total Source1 entities: {len(df):,}")

    # Basic safety checks
    if df["entity_id"].isna().any():
        raise ValueError("Missing entity_id values found.")

    if df["entity_id"].duplicated().any():
        raise ValueError("Duplicate Source1 entity IDs found.")

    print("\nCountry distribution:")
    print(df["country"].value_counts())

    # Stratify by country so train/validation have
    # approximately the same US/India proportions.
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        stratify=df["country"],
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_file = output_path / "train_source1_ids.tsv"
    val_file = output_path / "val_source1_ids.tsv"

    train_df.to_csv(
        train_file,
        sep="\t",
        index=False,
    )

    val_df.to_csv(
        val_file,
        sep="\t",
        index=False,
    )

    # Leakage check
    train_ids = set(train_df["entity_id"])
    val_ids = set(val_df["entity_id"])

    overlap = train_ids.intersection(val_ids)

    if overlap:
        raise RuntimeError(
            f"DATA LEAKAGE DETECTED: "
            f"{len(overlap):,} IDs appear in both splits."
        )

    print("\n" + "=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)

    print(f"Train entities:      {len(train_df):,}")
    print(f"Validation entities: {len(val_df):,}")

    print("\nTrain countries:")
    print(train_df["country"].value_counts())

    print("\nValidation countries:")
    print(val_df["country"].value_counts())

    print("\nLeakage check: PASS")
    print("Shared Source1 IDs: 0")

    print(f"\nSaved train IDs to:\n{train_file}")
    print(f"\nSaved validation IDs to:\n{val_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create Source1-level ML train/validation split."
    )

    parser.add_argument(
        "--source1",
        required=True,
        help="Path to train_source1.tsv",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/splits",
        help="Directory for generated split files",
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.20,
        help="Validation fraction",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    create_source1_split(
        source1_path=args.source1,
        output_dir=args.output_dir,
        val_size=args.val_size,
        random_state=args.seed,
    )