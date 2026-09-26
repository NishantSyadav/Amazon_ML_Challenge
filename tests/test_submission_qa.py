"""Small local fixtures for the pre-submission QA checks."""

from src.submission_qa import (
    validate_candidate_pairs,
    validate_cross_file_consistency,
    validate_matching_results,
    validate_submission_outputs,
)


def _write(path, header, rows):
    path.write_text("\t".join(header) + "\n" + "\n".join("\t".join(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _fixtures(tmp_path):
    matching = _write(
        tmp_path / "matching_results.tsv",
        ["source1_entity_id", "matched_entity_ids"],
        [("S1-1", "S2-1,S3-1"), ("S1-2", "")],
    )
    candidates = _write(
        tmp_path / "candidate_pairs.tsv",
        ["source1_entity_id", "candidate_entity_ids"],
        [("S1-1", "S2-1,S3-1"), ("S1-2", "")],
    )
    sources = [
        _write(tmp_path / f"test_source{number}.tsv", ["entity_id"], [(f"S{number}-{i}",) for i in ids])
        for number, ids in ((1, (1, 2)), (2, (1,)), (3, (1,)))
    ]
    return matching, candidates, sources


def _validate(matching, candidates, sources):
    return validate_submission_outputs(matching, candidates, *sources)


def _has_error(result, fragment):
    assert not result["passed"]
    assert any(fragment in error for error in result["errors"]), result["errors"]


def test_valid_outputs(tmp_path):
    matching, candidates, sources = _fixtures(tmp_path)
    assert _validate(matching, candidates, sources) == {"passed": True, "errors": [], "warnings": []}
    assert validate_matching_results(matching, *sources)["passed"]
    assert validate_candidate_pairs(candidates, *sources)["passed"]
    assert validate_cross_file_consistency(matching, candidates)["passed"]


def test_duplicate_s1_row(tmp_path):
    matching, candidates, sources = _fixtures(tmp_path)
    _write(matching, ["source1_entity_id", "matched_entity_ids"], [("S1-1", "S2-1"), ("S1-1", "")])
    _has_error(_validate(matching, candidates, sources), "duplicate S1 row")


def test_invalid_s1_prefix(tmp_path):
    matching, candidates, sources = _fixtures(tmp_path)
    _write(matching, ["source1_entity_id", "matched_entity_ids"], [("X1-1", "S2-1"), ("S1-2", "")])
    _has_error(_validate(matching, candidates, sources), "invalid S1 prefix")


def test_duplicate_id_inside_list(tmp_path):
    matching, candidates, sources = _fixtures(tmp_path)
    _write(matching, ["source1_entity_id", "matched_entity_ids"], [("S1-1", "S2-1,S2-1"), ("S1-2", "")])
    _has_error(_validate(matching, candidates, sources), "duplicate listed ID")


def test_invalid_candidate_prefix(tmp_path):
    matching, candidates, sources = _fixtures(tmp_path)
    _write(candidates, ["source1_entity_id", "candidate_entity_ids"], [("S1-1", "S1-2"), ("S1-2", "")])
    _has_error(_validate(matching, candidates, sources), "invalid listed ID prefix")


def test_match_absent_from_candidate_set(tmp_path):
    matching, candidates, sources = _fixtures(tmp_path)
    _write(candidates, ["source1_entity_id", "candidate_entity_ids"], [("S1-1", "S2-1"), ("S1-2", "")])
    _has_error(_validate(matching, candidates, sources), "matched ID absent from candidate set: S3-1")


def test_missing_required_s1(tmp_path):
    matching, candidates, sources = _fixtures(tmp_path)
    _write(matching, ["source1_entity_id", "matched_entity_ids"], [("S1-1", "S2-1,S3-1")])
    _has_error(_validate(matching, candidates, sources), "missing required S1 IDs: S1-2")


def test_unknown_s2_and_s3_ids(tmp_path):
    matching, candidates, sources = _fixtures(tmp_path)
    _write(candidates, ["source1_entity_id", "candidate_entity_ids"], [("S1-1", "S2-9,S3-9"), ("S1-2", "")])
    result = _validate(matching, candidates, sources)
    _has_error(result, "unknown Source 2 ID: S2-9")
    _has_error(result, "unknown Source 3 ID: S3-9")


def test_unknown_s1_and_missing_file(tmp_path):
    matching, candidates, sources = _fixtures(tmp_path)
    _write(candidates, ["source1_entity_id", "candidate_entity_ids"], [("S1-1", "S2-1,S3-1"), ("S1-9", "")])
    _has_error(_validate(matching, candidates, sources), "unknown S1 IDs: S1-9")
    _has_error(validate_matching_results(tmp_path / "absent.tsv"), "file does not exist")


def test_empty_source_id_and_wrong_columns(tmp_path):
    matching, candidates, sources = _fixtures(tmp_path)
    _write(matching, ["source1_entity_id", "matched_entity_ids"], [("", "S2-1"), ("S1-2", "")])
    _has_error(_validate(matching, candidates, sources), "empty source1_entity_id")
    _write(candidates, ["source1_entity_id", "wrong_column"], [("S1-1", "S2-1")])
    _has_error(validate_candidate_pairs(candidates), "expected TSV columns")
