# Amazon ML Challenge 2026 — business entity resolution

The production entry point is `python -m src.pipeline`. It uses only challenge
records, a bounded disk-backed candidate index, the saved V3.1 CatBoost classifier,
and streaming TSV writers. Every S1 record remains in both outputs, including
France, new country labels, and zero-candidate/zero-match entities.

## Environment

Verified locally with Python 3.12. Install from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The submission package includes the trained model at
`artifacts/experiments/baseline_v31/baseline_v1_catboost.cbm`. Repository users must
retain their existing model artifact; model binaries and datasets are Git-ignored.
Dependencies are pinned to the tested environment. For development and the legacy
TF-IDF experiment, install `requirements-dev.txt` instead.

## Reproduce both outputs

Set `$TestDir` to the challenge directory containing `test_source1.tsv`,
`test_source2.tsv`, and `test_source3.tsv`. All files use tabs and UTF-8.

```powershell
$TestDir = 'C:\path\to\student_resource\dataset\test'
python -m src.pipeline run --test-dir $TestDir --index-dir work/test_index --output-dir output --threshold 0.560 --batch-size 500 --threads 1 --workers 4
python -m src.official_validation --test-dir $TestDir --output-dir output --scratch work/official_validation
```

Use a fresh output directory to regenerate an existing submission. The program
will not overwrite a completed output. A failed run leaves `.partial` files;
rerun into a fresh output directory. Target indexes are reusable only when source
paths, sizes, SHA-256 hashes, and configuration match. An interrupted index build
requires a fresh index directory. Allow at least 15 GB free disk for each target
index, outputs, packaging, and scratch files. The two historical/validation index
builds are not required to reproduce test inference.

Individual stages and scalability checks:

```powershell
python -m src.pipeline index --test-dir $TestDir --index-dir work/test_index
python -m src.pipeline infer --test-dir $TestDir --index-dir work/test_index --output-dir work/benchmark_1000 --limit 1000
python -m src.pipeline validate --test-dir $TestDir --index-dir work/test_index --output-dir work/benchmark_1000 --limit 1000
python -m src.pipeline infer --test-dir $TestDir --index-dir work/test_index --output-dir work/benchmark_10000 --limit 10000
python -m src.pipeline infer --test-dir $TestDir --index-dir work/test_index --output-dir work/benchmark_50000 --limit 50000
python -m src.pipeline infer --test-dir $TestDir --index-dir work/test_index --output-dir output --threshold 0.560
python -m src.pipeline validate --test-dir $TestDir --index-dir work/test_index --output-dir output --expected-rows 1732544
```

`run` builds/reuses the index, scores candidates, and executes strict QA. The
separate official validator command is mandatory before packaging. `run.json`
contains counts, throughput, memory observations, model/input hashes, feature
order, and threshold. `strict_qa.json` binds checks to hashes of both output files.

## Architecture and consistency

- `disk_candidates.py`: streaming S2/S3 ingestion into SQLite, exact name/address
  indexes, fixed memory-mapped n-gram postings. Common/overflowed hashed buckets
  are skipped. Query at most six name and eight address anchors; preselect up to
  80 candidates per source, then fuzzy rerank to 50 per source. Exact full-field,
  name, and address hits are incorporated with a 512-record scan cap per source
  per key. The cap prevents unbounded common-key expansion; counts of overflow
  events are reported. This differs from the historical sampled V3.1 retriever.
- `features.py`: the exact 16 training features, including training-time NFKD
  accent folding. No legal-suffix expansion is introduced into saved-model
  features. UTF-8 inputs and arbitrary country labels are preserved.
- `inference.py`: strict model feature-name/order and positive-class checks.
- `pipeline.py`: bounded batches, one output row per S1, empty lists as empty
  fields, deterministic sorted ID lists, strict disk-backed QA.
- `official_validate_submission.py`: unchanged challenge-supplied validator.
- `submission_qa.py` and `experiment_tracker.py`: preserved teammate QA/tracking
  tools. The former remains useful for small files; production QA streams because
  its original implementation materializes whole files.
- `validate_final_model.py`: separate training-only held-out evaluation. It is
  never imported by test inference and is the only new module that reads truth.

Legacy `blocking.py`/`normalize.py` are exploratory TF-IDF components, not the
production retrieval/feature path. `baseline_candidates_experimental_v2.py` is
excluded from version control and packaging. Existing model training code uses
entity-disjoint validation. Historical validation scores are internal diagnostics,
not leaderboard results. See `SUBMISSION_READINESS.md` for measured final results.

## Tests

```powershell
python -m pip install -r requirements-dev.txt
New-Item -ItemType Directory -Force work | Out-Null
python -m pytest -q --basetemp=work/pytest-reproduction
```

The suite includes normalization/feature parity, model schema rejection,
threshold boundaries, multi-match and empty outputs, France/unseen countries,
per-source candidate caps, index identity, and strict candidate subset checks.
Some legacy tests are import-time assertion scripts and execute during collection.

## Validation and limitations

The strict validator checks every S1 identity and all candidate target IDs using
SQLite, including duplicates, exact headers, readback, country counts, and the
match-subset invariant. The supplied validator loads candidate sets into RAM;
its optional `--check-ids` adds all S2/S3 IDs. The strict streaming check already
verifies target existence with bounded memory. Do not substitute validation success
for a quality score: test ground truth and leaderboard scores are unavailable.

The fixed posting cap, hash collisions, exact-key scan cap, and final top-K can miss
links. France is not represented in model training. The saved model and normalized
feature definitions are retained; final retrieval quality is measured separately.

## Memory-safe official validation

For the full candidate output on this 16 GB machine, run:

```powershell
python -m src.official_validation --test-dir $TestDir --output-dir output --scratch work/official_validation
```

This calls the **unchanged** supplied validator on exhaustive 20,000-row
partitions, checks global Source1 uniqueness and row order, and turns its subset
warning into a hard failure. The original utility materializes every candidate ID
in Python sets; partitioning bounds that memory. The transcript and JSON report
record every partition and bind success to final output hashes. Run strict QA first;
it checks every target ID against the original S2/S3 index with bounded memory.

Production index version 2 includes name 4/6-grams and address 5/8-grams. Longer
anchors address short-gram saturation in the full target universe. Default inference
uses four worker processes, one CatBoost thread each, and at most eight pending
500-entity batches. Pass `--workers 1` for serial execution; outputs are identical.

After source/docs are committed and the working tree is clean, package the verified
run (choose the archive name matching your portal team name):

```powershell
python -m src.package_submission --output-dir output --zip Amazon_ML_Challenge_submission.zip
```

Packaging rejects incomplete/stale validation, embeds the trained model and code,
creates `SUBMISSION_READINESS.md` with the exact source commit, checks ZIP hierarchy
and CRCs, and hashes the archived TSVs/model against validated originals. Generated
readiness/verification records are ignored by Git; the ZIP contains their final state.

Final threshold is **0.560**, selected after full-scale retrieval changed. On the
original 1,000 held-out S1, macro F0.5 improves from 0.829989 at 0.610 to 0.831200
at 0.560. No model retraining was performed; these remain internal tuning results.

The packaging command runs from the Git checkout, where it can verify the clean
source commit and local validation evidence. Recipients of the ZIP can regenerate
both TSVs from the packaged `code/business_entity_resolution/` directory and the
official test data; a Git checkout is not needed for indexing, inference, or QA.
