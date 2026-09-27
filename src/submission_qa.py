"""Local pre-submission checks; the official challenge validator is final authority."""

from __future__ import annotations

import csv
from pathlib import Path


def _result(errors: list[str]) -> dict[str, object]:
    return {"passed": not errors, "errors": errors, "warnings": []}


def _read_tsv(path: str | Path) -> tuple[list[str], list[dict[str, str]], list[str]]:
    path = Path(path)
    if not path.is_file():
        return [], [], [f"{path}: file does not exist"]
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t", strict=True)
            columns = reader.fieldnames or []
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        return [], [], [f"{path}: cannot read as TSV: {exc}"]
    if not columns:
        return [], [], [f"{path}: missing TSV header"]
    if any(None in row or None in row.values() for row in rows):
        return columns, [], [f"{path}: row has a different number of fields than the TSV header"]
    return columns, rows, []


def _read_test_ids(path: str | Path, source: int) -> tuple[set[str], list[str]]:
    columns, rows, errors = _read_tsv(path)
    if errors:
        return set(), errors
    id_column = next(
        (name for name in (f"source{source}_entity_id", "entity_id") if name in columns),
        None,
    )
    if id_column is None:
        return set(), [f"{path}: missing entity ID column for Source {source}"]
    ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        entity_id = row[id_column]
        if entity_id is None or not entity_id.strip():
            errors.append(f"{path}: row {row_number}: empty entity ID")
        else:
            ids.add(entity_id)
    return ids, errors


def _read_output(
    path: str | Path,
    list_column: str,
    test_ids: dict[int, set[str]] | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    columns, rows, errors = _read_tsv(path)
    if errors:
        return {}, errors
    expected = ["source1_entity_id", list_column]
    if len(columns) != len(expected) or set(columns) != set(expected):
        return {}, [f"{path}: expected TSV columns {expected}; found {columns}"]

    values: dict[str, list[str]] = {}
    for row_number, row in enumerate(rows, start=2):
        s1 = row["source1_entity_id"]
        if s1 is None or not s1.strip():
            errors.append(f"{path}: row {row_number}: empty source1_entity_id")
            continue
        if not s1.startswith("S1-"):
            errors.append(f"{path}: row {row_number}: invalid S1 prefix: {s1}")
        if s1 in values:
            errors.append(f"{path}: row {row_number}: duplicate S1 row: {s1}")

        raw = row[list_column]
        ids = [] if raw is None or raw == "" else raw.split(",")
        seen: set[str] = set()
        for entity_id in ids:
            if entity_id in seen:
                errors.append(f"{path}: row {row_number}: duplicate listed ID: {entity_id}")
            seen.add(entity_id)
            if not entity_id.startswith(("S2-", "S3-")):
                errors.append(f"{path}: row {row_number}: invalid listed ID prefix: {entity_id!r}")
            elif test_ids is not None:
                source = 2 if entity_id.startswith("S2-") else 3
                if source in test_ids and entity_id not in test_ids[source]:
                    errors.append(f"{path}: row {row_number}: unknown Source {source} ID: {entity_id}")
        values[s1] = ids

    if test_ids is not None and 1 in test_ids:
        missing = sorted(test_ids[1] - values.keys())
        extra = sorted(values.keys() - test_ids[1])
        if missing:
            errors.append(f"{path}: missing required S1 IDs: {', '.join(missing)}")
        if extra:
            errors.append(f"{path}: unknown S1 IDs: {', '.join(extra)}")
    return values, errors


def _source_ids(
    test_source1: str | Path | None,
    test_source2: str | Path | None,
    test_source3: str | Path | None,
) -> tuple[dict[int, set[str]], list[str]]:
    result: dict[int, set[str]] = {}
    errors: list[str] = []
    for source, path in enumerate((test_source1, test_source2, test_source3), start=1):
        if path is not None:
            result[source], source_errors = _read_test_ids(path, source)
            errors.extend(source_errors)
    return result, errors


def validate_matching_results(
    path: str | Path,
    test_source1: str | Path | None = None,
    test_source2: str | Path | None = None,
    test_source3: str | Path | None = None,
) -> dict[str, object]:
    """Validate the matching TSV, optionally against local test-source TSVs."""
    test_ids, errors = _source_ids(test_source1, test_source2, test_source3)
    _, output_errors = _read_output(path, "matched_entity_ids", test_ids)
    return _result(errors + output_errors)


def validate_candidate_pairs(
    path: str | Path,
    test_source1: str | Path | None = None,
    test_source2: str | Path | None = None,
    test_source3: str | Path | None = None,
) -> dict[str, object]:
    """Validate the final candidate TSV, optionally against local test sources."""
    test_ids, errors = _source_ids(test_source1, test_source2, test_source3)
    _, output_errors = _read_output(path, "candidate_entity_ids", test_ids)
    return _result(errors + output_errors)


def _cross_file_errors(
    matches: dict[str, list[str]], candidates: dict[str, list[str]]
) -> list[str]:
    errors: list[str] = []
    for s1, matched_ids in matches.items():
        candidate_ids = set(candidates.get(s1, []))
        for entity_id in matched_ids:
            if entity_id not in candidate_ids:
                errors.append(f"{s1}: matched ID absent from candidate set: {entity_id}")
    return errors


def validate_cross_file_consistency(
    matching_path: str | Path, candidate_path: str | Path
) -> dict[str, object]:
    """Check that every matched ID occurs in the corresponding final candidate set."""
    matches, match_errors = _read_output(matching_path, "matched_entity_ids")
    candidates, candidate_errors = _read_output(candidate_path, "candidate_entity_ids")
    return _result(match_errors + candidate_errors + _cross_file_errors(matches, candidates))


def validate_submission_outputs(
    matching_path: str | Path,
    candidate_path: str | Path,
    test_source1: str | Path | None = None,
    test_source2: str | Path | None = None,
    test_source3: str | Path | None = None,
) -> dict[str, object]:
    """Run all local checks and return passed, errors, and warnings."""
    test_ids, errors = _source_ids(test_source1, test_source2, test_source3)
    matches, match_errors = _read_output(matching_path, "matched_entity_ids", test_ids)
    candidates, candidate_errors = _read_output(candidate_path, "candidate_entity_ids", test_ids)
    return _result(errors + match_errors + candidate_errors + _cross_file_errors(matches, candidates))
