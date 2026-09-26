"""Local CSV tracker tests with temporary files only."""

import csv

from src.experiment_tracker import (
    REQUIRED_COLUMNS,
    append_experiment,
    can_submit_today,
    count_daily_submissions,
    find_experiment,
    load_tracker,
)


def _tracker(tmp_path):
    path = tmp_path / "experiment_tracker.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(REQUIRED_COLUMNS)
    return path


def _record(experiment_id, number="", **overrides):
    record = {
        "experiment_id": experiment_id,
        "date": "2026-09-26",
        "git_commit": "abc123",
        "branch": "sayem/qa-tracking",
        "model_name": "local-model",
        "model_config": "config-a",
        "feature_config": "features-a",
        "candidate_config": "candidates-a",
        "decision_threshold": "0.5",
        "validator_status": "PASS",
        "submission_file": "output/matching_results.tsv" if number else "",
        "candidate_file": "output/candidate_pairs.tsv" if number else "",
        "submission_number_for_day": number,
        "status": "submitted" if number else "validated",
    }
    record.update(overrides)
    return record


def _expect_value_error(action, message):
    try:
        action()
    except ValueError as exc:
        assert message in str(exc)
    else:
        raise AssertionError(f"Expected ValueError containing {message!r}")


def test_duplicate_experiment_id_rejected(tmp_path):
    path = _tracker(tmp_path)
    append_experiment(path, _record("E001"))
    _expect_value_error(lambda: append_experiment(path, _record("E001")), "Duplicate experiment ID")
    assert len(load_tracker(path)) == 1


def test_daily_submission_count_and_validation_only(tmp_path):
    path = _tracker(tmp_path)
    append_experiment(path, _record("E001"))
    append_experiment(path, _record("E002", "1"))
    rows = load_tracker(path)
    assert count_daily_submissions(rows, "2026-09-26") == 1
    assert count_daily_submissions(rows, "2026-09-27") == 0
    assert can_submit_today(rows, "2026-09-26")
    assert find_experiment(rows, "E002")["submission_number_for_day"] == "1"


def test_submission_numbers_one_through_five_accepted(tmp_path):
    path = _tracker(tmp_path)
    for number in range(1, 6):
        append_experiment(path, _record(f"E{number:03}", str(number)))
    rows = load_tracker(path)
    assert count_daily_submissions(rows, "2026-09-26") == 5
    assert not can_submit_today(rows, "2026-09-26")


def test_submission_number_above_five_rejected(tmp_path):
    path = _tracker(tmp_path)
    _expect_value_error(lambda: append_experiment(path, _record("E001", "6")), "integer from 1 to 5")
    assert load_tracker(path) == []


def test_repeated_submission_number_rejected(tmp_path):
    path = _tracker(tmp_path)
    append_experiment(path, _record("E001", "1"))
    _expect_value_error(
        lambda: append_experiment(path, _record("E002", "1")),
        "already recorded",
    )


def test_existing_rows_are_preserved(tmp_path):
    path = _tracker(tmp_path)
    append_experiment(path, _record("E001", notes="First observation"))
    before = load_tracker(path)[0].copy()
    append_experiment(path, _record("E002", "1"))
    after = load_tracker(path)
    assert after[0] == before
    assert after[0]["notes"] == "First observation"
    assert len(after) == 2
