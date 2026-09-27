"""Local CSV bookkeeping for experiments and leaderboard submissions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


REQUIRED_COLUMNS = (
    "experiment_id",
    "date",
    "git_commit",
    "branch",
    "model_name",
    "model_config",
    "feature_config",
    "candidate_config",
    "decision_threshold",
    "validation_f0_5",
    "validator_status",
    "submission_file",
    "candidate_file",
    "submission_number_for_day",
    "public_leaderboard_score",
    "private_leaderboard_score",
    "status",
    "notes",
)


def validate_tracker_schema(columns: Iterable[str]) -> None:
    """Raise if the CSV cannot store all required tracking fields."""
    available = list(columns)
    missing = sorted(set(REQUIRED_COLUMNS) - set(available))
    if missing:
        raise ValueError(f"Tracker is missing columns: {', '.join(missing)}")
    if len(available) != len(set(available)):
        raise ValueError("Tracker has duplicate column names")


def _read_tracker(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        validate_tracker_schema(columns)
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError("Tracker contains a row with the wrong number of fields")
    return columns, rows


def load_tracker(path: str | Path) -> list[dict[str, str]]:
    """Read existing rows without changing them."""
    return _read_tracker(path)[1]


def _is_submission(row: Mapping[str, str]) -> bool:
    return bool(
        row.get("submission_number_for_day")
        or row.get("submission_file")
        or row.get("status") in {"submitted", "final_selected"}
    )


def count_daily_submissions(rows: Iterable[Mapping[str, str]], date: str) -> int:
    """Count recorded uploads on a date; validation-only rows do not count."""
    return sum(row.get("date") == date and _is_submission(row) for row in rows)


def can_submit_today(rows: Iterable[Mapping[str, str]], date: str) -> bool:
    """Return whether fewer than five uploads are recorded for the date."""
    return count_daily_submissions(rows, date) < 5


def find_experiment(rows: Iterable[Mapping[str, str]], experiment_id: str) -> Mapping[str, str] | None:
    """Find an experiment by its unique ID."""
    return next((row for row in rows if row.get("experiment_id") == experiment_id), None)


def append_experiment(path: str | Path, record: Mapping[str, str]) -> None:
    """Append one new experiment without editing earlier rows or uploading files."""
    columns, rows = _read_tracker(path)
    unknown_columns = set(record) - set(columns)
    if unknown_columns:
        raise ValueError(f"Unknown tracker columns: {', '.join(sorted(unknown_columns))}")

    experiment_id = record.get("experiment_id", "")
    if not experiment_id:
        raise ValueError("experiment_id is required")
    if find_experiment(rows, experiment_id) is not None:
        raise ValueError(f"Duplicate experiment ID: {experiment_id}")
    date = record.get("date", "")
    if not date:
        raise ValueError("date is required")

    number = record.get("submission_number_for_day", "")
    if _is_submission(record):
        if not number or not number.isdigit() or not 1 <= int(number) <= 5:
            raise ValueError("submission_number_for_day must be an integer from 1 to 5")
        if not can_submit_today(rows, date):
            raise ValueError(f"Five submissions are already recorded for {date}")
        if any(
            row.get("date") == date and row.get("submission_number_for_day") == number
            for row in rows
        ):
            raise ValueError(f"Submission number {number} is already recorded for {date}")
        required = (
            "git_commit", "branch", "model_name", "model_config", "feature_config",
            "candidate_config", "decision_threshold", "submission_file", "candidate_file",
        )
        missing = [name for name in required if not record.get(name)]
        if missing:
            raise ValueError(f"Submission is missing fields: {', '.join(missing)}")
        if record.get("validator_status", "").upper() != "PASS":
            raise ValueError("Official validator must be PASS before recording a submission")
        if Path(record["submission_file"]).name != "matching_results.tsv":
            raise ValueError("submission_file must be matching_results.tsv")
        if Path(record["candidate_file"]).name != "candidate_pairs.tsv":
            raise ValueError("candidate_file must be candidate_pairs.tsv")
    elif number:
        raise ValueError("Validation-only experiments must not have a submission number")

    with Path(path).open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writerow(record)
