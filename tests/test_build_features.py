"""Small, self-contained checks for the candidate-to-feature interface."""

from pathlib import Path

import pandas as pd
import pytest

from src.build_features import CANDIDATE_COLUMNS, generate_feature_file
from src.features import build_pair_features
from src.normalize import normalize_name_ascii


@pytest.fixture
def sample_files(tmp_path: Path):
    paths = {name: tmp_path / f"{name}.tsv" for name in ("candidates", "source1", "source2", "source3", "out")}
    pd.DataFrame([
        ["S1-1", "S2-1", "S2", "0.91", "1"],
        ["S1-1", "S3-1", "S3", "0.73", "2"],
    ], columns=CANDIDATE_COLUMNS).to_csv(paths["candidates"], sep="\t", index=False)
    columns = ["entity_id", "business_name", "business_address", "country"]
    pd.DataFrame([["S1-1", "Café Bleu", "12 Rue de Paris", "France"]], columns=columns).to_csv(paths["source1"], sep="\t", index=False)
    pd.DataFrame([["S2-1", "Cafe Bleu", "12 Rue de Paris", "France"]], columns=columns).to_csv(paths["source2"], sep="\t", index=False)
    pd.DataFrame([["S3-1", "Café Bleu", "", "France"]], columns=columns).to_csv(paths["source3"], sep="\t", index=False)
    return paths


def run_sample(paths, **kwargs):
    return generate_feature_file(
        paths["candidates"], paths["source1"], paths["source2"],
        paths["source3"], paths["out"], candidate_chunk_size=1,
        source_chunk_size=1, **kwargs,
    )


def test_s2_s3_features_and_metadata(sample_files):
    summary = run_sample(sample_files)
    result = pd.read_csv(sample_files["out"], sep="\t")
    assert summary["input_pairs"] == summary["output_rows"] == 2
    assert summary["s2_pairs"] == summary["s3_pairs"] == 1
    assert summary["feature_columns"] == len(result.columns) - 3 == 26
    assert result["candidate_entity_id"].tolist() == ["S2-1", "S3-1"]
    assert result["candidate_source"].tolist() == ["S2", "S3"]
    assert result["retrieval_score"].tolist() == pytest.approx([0.91, 0.73])
    assert result["retrieval_rank"].tolist() == [1, 2]
    assert result["candidate_is_s2"].tolist() == [1, 0]
    assert result["candidate_is_s3"].tolist() == [0, 1]
    assert result["same_country"].tolist() == [1, 1]
    assert result["name_ascii_exact"].tolist() == [1, 1]
    assert result["address2_missing"].tolist() == [0, 1]
    assert "business_name" not in result.columns
    assert "business_address" not in result.columns
    assert not result.isna().any().any()
    assert all(pd.api.types.is_numeric_dtype(result[c]) for c in result.columns[3:])
    assert normalize_name_ascii("Café Bleu") == "cafe bleu"


def test_missing_source_record_fails(sample_files):
    pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"]).to_csv(
        sample_files["source3"], sep="\t", index=False,
    )
    with pytest.raises(ValueError, match="S3=1"):
        run_sample(sample_files)
    assert not sample_files["out"].exists()


def test_bad_candidate_source_fails(sample_files):
    frame = pd.read_csv(sample_files["candidates"], sep="\t", dtype=str)
    frame.loc[1, "candidate_source"] = "S4"
    frame.to_csv(sample_files["candidates"], sep="\t", index=False)
    with pytest.raises(ValueError, match="malformed candidate_source"):
        run_sample(sample_files)


def test_duplicate_across_chunks_fails(sample_files):
    frame = pd.read_csv(sample_files["candidates"], sep="\t", dtype=str)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(sample_files["candidates"], sep="\t", index=False)
    with pytest.raises(ValueError, match="duplicate .* pairs: 1"):
        run_sample(sample_files)


def test_missing_column_and_non_numeric_metadata_fail(sample_files):
    frame = pd.read_csv(sample_files["candidates"], sep="\t", dtype=str)
    frame.drop(columns=["retrieval_rank"]).to_csv(sample_files["candidates"], sep="\t", index=False)
    with pytest.raises(ValueError, match="missing required columns: retrieval_rank"):
        run_sample(sample_files)
    frame.loc[0, "retrieval_score"] = "bad"
    frame.to_csv(sample_files["candidates"], sep="\t", index=False)
    with pytest.raises(ValueError, match="retrieval_score must contain finite numeric"):
        run_sample(sample_files)


def test_max_pairs_and_existing_feature_function(sample_files):
    summary = run_sample(sample_files, max_pairs=1)
    result = pd.read_csv(sample_files["out"], sep="\t")
    assert summary["output_rows"] == 1
    expected = build_pair_features(
        "Café Bleu", "12 Rue de Paris", "France",
        "Cafe Bleu", "12 Rue de Paris", "France", "S2", 0.91, 1,
    )
    for name, value in expected.items():
        assert result.loc[0, name] == pytest.approx(value)
