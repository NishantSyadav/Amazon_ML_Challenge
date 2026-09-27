"""Join long candidate pairs to source records and write model-ready features.

Run from the repository root with ``python -m src.build_features --help``.
Only entities referenced by the candidate file are retained in memory.
"""

from __future__ import annotations

import argparse
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_pair_features


CANDIDATE_COLUMNS = (
    "source1_entity_id", "candidate_entity_id", "candidate_source",
    "retrieval_score", "retrieval_rank",
)
SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
VALID_SOURCES = {"S2", "S3"}


def _require_columns(path: Path, required: tuple[str, ...]) -> None:
    columns = pd.read_csv(path, sep="\t", nrows=0).columns
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")


def validate_candidate_schema(path: Path) -> None:
    _require_columns(path, CANDIDATE_COLUMNS)


def candidate_chunks(path: Path, chunk_size: int, max_pairs: int | None):
    remaining = max_pairs
    for chunk in pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False,
        usecols=list(CANDIDATE_COLUMNS), chunksize=chunk_size,
    ):
        if remaining is not None:
            chunk = chunk.iloc[:remaining]
            remaining -= len(chunk)
        if not chunk.empty:
            yield chunk
        if remaining == 0:
            break


def _validate_chunk(chunk: pd.DataFrame, start_row: int) -> None:
    bad_sources = chunk.loc[~chunk["candidate_source"].isin(VALID_SOURCES), "candidate_source"]
    if not bad_sources.empty:
        counts = bad_sources.value_counts(dropna=False).to_dict()
        raise ValueError(f"malformed candidate_source values at/after row {start_row}: {counts}")
    for column in ("source1_entity_id", "candidate_entity_id"):
        if chunk[column].str.strip().eq("").any():
            raise ValueError(f"empty {column} at/after row {start_row}")
    for column in ("retrieval_score", "retrieval_rank"):
        values = pd.to_numeric(chunk[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite numeric values at/after row {start_row}")
        if column == "retrieval_rank" and ((values < 1) | (values % 1 != 0)).any():
            raise ValueError(f"retrieval_rank must contain positive integers at/after row {start_row}")


def collect_required_ids(path: Path, chunk_size: int, max_pairs: int | None, db: sqlite3.Connection):
    """Scan candidates once; use a disk-backed unique index for global duplicate checks."""
    db.execute("CREATE TABLE pairs (s1 TEXT NOT NULL, candidate TEXT NOT NULL, PRIMARY KEY (s1, candidate)) WITHOUT ROWID")
    required = {"S1": set(), "S2": set(), "S3": set()}
    pair_count = duplicate_count = 0
    for chunk in candidate_chunks(path, chunk_size, max_pairs):
        _validate_chunk(chunk, pair_count + 1)
        before = db.total_changes
        db.executemany(
            "INSERT OR IGNORE INTO pairs VALUES (?, ?)",
            chunk[["source1_entity_id", "candidate_entity_id"]].itertuples(index=False, name=None),
        )
        duplicate_count += len(chunk) - (db.total_changes - before)
        required["S1"].update(chunk["source1_entity_id"])
        for source in VALID_SOURCES:
            required[source].update(chunk.loc[chunk["candidate_source"] == source, "candidate_entity_id"])
        pair_count += len(chunk)
    db.commit()
    if duplicate_count:
        raise ValueError(f"duplicate (source1_entity_id, candidate_entity_id) pairs: {duplicate_count}")
    return required, pair_count, duplicate_count


def load_required_entities(path: Path, required_ids: set[str], chunk_size: int) -> dict[str, tuple[str, str, str]]:
    """Scan a source TSV in chunks and retain only referenced records."""
    _require_columns(path, SOURCE_COLUMNS)
    found: dict[str, tuple[str, str, str]] = {}
    if not required_ids:
        return found
    for chunk in pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False,
        usecols=list(SOURCE_COLUMNS), chunksize=chunk_size,
    ):
        for entity_id, name, address, country in chunk[list(SOURCE_COLUMNS)].itertuples(index=False, name=None):
            if entity_id in required_ids:
                if entity_id in found:
                    raise ValueError(f"{path}: duplicate required entity_id {entity_id!r}")
                found[entity_id] = (name, address, country)
    return found


def build_feature_chunk(chunk: pd.DataFrame, entities: dict[str, dict[str, tuple[str, str, str]]]) -> pd.DataFrame:
    rows = []
    for s1_id, candidate_id, source, score_text, rank_text in chunk[list(CANDIDATE_COLUMNS)].itertuples(index=False, name=None):
        name1, address1, country1 = entities["S1"][s1_id]
        name2, address2, country2 = entities[source][candidate_id]
        score, rank = float(score_text), int(float(rank_text))
        features = build_pair_features(
            name1, address1, country1, name2, address2, country2,
            candidate_source=source, retrieval_score=score, retrieval_rank=rank,
        )
        # Retain the original metadata once; the feature function returns these two keys too.
        features.pop("retrieval_score")
        features.pop("retrieval_rank")
        rows.append(dict(zip(CANDIDATE_COLUMNS, (s1_id, candidate_id, source, score, rank))) | features)
    result = pd.DataFrame(rows)
    numeric = result.drop(columns=["source1_entity_id", "candidate_entity_id", "candidate_source"])
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("feature output contains NaN or infinite numeric values")
    return result


def generate_feature_file(
    candidates: Path, source1: Path, source2: Path, source3: Path, out: Path,
    candidate_chunk_size: int = 10_000, source_chunk_size: int = 100_000,
    max_pairs: int | None = None,
) -> dict[str, int]:
    if candidate_chunk_size < 1 or source_chunk_size < 1 or (max_pairs is not None and max_pairs < 1):
        raise ValueError("chunk sizes and max_pairs must be positive")
    validate_candidate_schema(candidates)
    with tempfile.TemporaryDirectory(prefix="candidate_validation_") as temp_dir:
        db = sqlite3.connect(str(Path(temp_dir) / "pairs.sqlite"))
        try:
            required, pair_count, duplicate_count = collect_required_ids(
                candidates, candidate_chunk_size, max_pairs, db,
            )
        finally:
            db.close()

    entities = {
        "S1": load_required_entities(source1, required["S1"], source_chunk_size),
        "S2": load_required_entities(source2, required["S2"], source_chunk_size),
        "S3": load_required_entities(source3, required["S3"], source_chunk_size),
    }
    missing = {source: len(required[source] - entities[source].keys()) for source in entities}
    if any(missing.values()):
        raise ValueError(f"missing source records: S1={missing['S1']}, S2={missing['S2']}, S3={missing['S3']}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tmp", prefix="features_", dir=out.parent, delete=False) as handle:
        temp_output = Path(handle.name)
    written = 0
    source_counts = {"S2": 0, "S3": 0}
    feature_count = 0
    try:
        for chunk in candidate_chunks(candidates, candidate_chunk_size, max_pairs):
            frame = build_feature_chunk(chunk, entities)
            frame.to_csv(temp_output, sep="\t", index=False, mode="a", header=written == 0)
            written += len(frame)
            feature_count = len(frame.columns) - 3  # Numeric model inputs, including retrieval score/rank.
            for source in source_counts:
                source_counts[source] += int((chunk["candidate_source"] == source).sum())
        if written == 0:
            sample = build_pair_features("", "", "", "", "", "", "S2", 0, 1)
            columns = list(CANDIDATE_COLUMNS) + [k for k in sample if k not in ("retrieval_score", "retrieval_rank")]
            pd.DataFrame(columns=columns).to_csv(temp_output, sep="\t", index=False)
            feature_count = len(columns) - 3
        if written != pair_count:
            raise ValueError(f"output row count {written} differs from input pair count {pair_count}")
        temp_output.replace(out)
    finally:
        temp_output.unlink(missing_ok=True)
    return {
        "input_pairs": pair_count, "output_rows": written, "s2_pairs": source_counts["S2"],
        "s3_pairs": source_counts["S3"], "duplicates": duplicate_count,
        "missing_s1": missing["S1"], "missing_s2": missing["S2"], "missing_s3": missing["S3"],
        "nan_or_infinite": 0, "feature_columns": feature_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidates", "source1", "source2", "source3", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--candidate-chunk-size", type=int, default=10_000)
    parser.add_argument("--source-chunk-size", type=int, default=100_000)
    parser.add_argument("--max-pairs", type=int)
    args = parser.parse_args()
    summary = generate_feature_file(
        args.candidates, args.source1, args.source2, args.source3, args.out,
        args.candidate_chunk_size, args.source_chunk_size, args.max_pairs,
    )
    print("feature generation:", ", ".join(f"{key}={value}" for key, value in summary.items()))


if __name__ == "__main__":
    main()
