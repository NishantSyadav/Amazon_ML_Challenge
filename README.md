# Amazon ML Challenge 2026 — Business Entity Resolution

A scalable and reproducible **business entity resolution pipeline** developed for the **Amazon ML Challenge 2026**.

The system matches every entity from **Source1 (S1)** against corresponding records in **Source2 (S2)** and **Source3 (S3)** while handling noisy business names, addresses, abbreviations, spelling variations, missing fields, Unicode text, and unseen country labels.

The production entry point is:

```bash
python -m src.pipeline
```

The final production system uses only challenge-provided records, a bounded disk-backed candidate index, the saved V3.1 CatBoost classifier, deterministic feature generation, streaming TSV writers, strict QA, and the official challenge validation workflow.

Every Source1 entity remains represented in both output files, including entities with:

- zero candidates
- zero predicted matches
- unseen country labels
- France records

---

## Team

| Contributor | Main Contribution |
|---|---|
| **Om Srivastav (`om-srivastav`)** | Production ML pipeline, scalable disk-backed candidate retrieval, CatBoost integration, inference, validation, final integration and submission packaging |
| **Ramnesh (`RAMNESH007`)** | Candidate generation, TF-IDF blocking, candidate-recall analysis and blocking-specific normalization |
| **Nishant (`NishantSyadav`)** | Data normalization, feature engineering and data-inspection utilities |
| **Sayem (`sayem23ai`)** | Experiment tracking, QA workflow, submission checks and supporting documentation |

The repository preserves the work of all team members while keeping the final validated production path isolated from experimental components.

---

## Problem Overview

The challenge requires resolving entities across three business datasets:

```text
Source1
   │
   ▼
Entity Resolution
   │
   ├──► Source2
   └──► Source3
```

Real-world records can contain:

- spelling differences
- punctuation differences
- reordered words
- legal suffix variations
- abbreviated addresses
- incomplete names
- incomplete addresses
- Unicode characters and accents
- duplicated businesses
- multiple records belonging to one entity
- unseen country labels

Training data contains labeled records from **India and the United States**, while the test data additionally includes **France**.

Country handling is therefore implemented as an open string label rather than being hardcoded to only the training countries.

---

# Final Production Architecture

```text
                  Source2 + Source3
                         │
                         ▼
             ┌──────────────────────┐
             │ Disk-backed Index    │
             │ disk_candidates.py   │
             └──────────┬───────────┘
                        │
                        ▼
               Candidate Retrieval
                        │
                        ▼
             ┌──────────────────────┐
             │ Production Features  │
             │ features.py          │
             │ 16 validated inputs  │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ CatBoost Classifier  │
             └──────────┬───────────┘
                        │
                        ▼
                  Threshold 0.560
                        │
                        ▼
          ┌─────────────────────────────┐
          │ matching_results.tsv        │
          │ candidate_pairs.tsv         │
          └─────────────┬───────────────┘
                        │
                        ▼
                Strict QA Validation
                        │
                        ▼
             Official Challenge Validator
```

The production pipeline is deliberately bounded so that candidate retrieval and inference remain feasible on a normal development machine without attempting to materialize the complete Source1 × Source2/Source3 search space.

---

# Environment

The final pipeline was verified locally with:

```text
Python 3.12
```

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install production dependencies:

```powershell
python -m pip install -r requirements.txt
```

For development and retained experimental components, including the historical TF-IDF blocker:

```powershell
python -m pip install -r requirements-dev.txt
```

The validated submission package includes the trained model at:

```text
artifacts/experiments/baseline_v31/baseline_v1_catboost.cbm
```

Model binaries and raw challenge datasets are intentionally excluded from ordinary Git tracking where appropriate.

---

# Reproducing the Outputs

Set `$TestDir` to the challenge directory containing:

```text
test_source1.tsv
test_source2.tsv
test_source3.tsv
```

Example:

```powershell
$TestDir = 'C:\path\to\student_resource\dataset\test'
```

Run the production pipeline:

```powershell
python -m src.pipeline run `
  --test-dir $TestDir `
  --index-dir work/test_index `
  --output-dir output `
  --threshold 0.560 `
  --batch-size 500 `
  --threads 1 `
  --workers 4
```

Then run the official validation wrapper:

```powershell
python -m src.official_validation `
  --test-dir $TestDir `
  --output-dir output `
  --scratch work/official_validation
```

The recorded final competition inference run used **6 worker processes**, while `4` remains a conservative reproduction default for machines with tighter resources.

Use a fresh output directory when regenerating an existing completed run.

The pipeline intentionally refuses to silently overwrite completed output files.

A failed or interrupted run may leave `.partial` files; rerun into a fresh output directory.

Target indexes are reusable only when the original source paths, file sizes, hashes and index configuration remain compatible.

An interrupted index build should be recreated in a fresh index directory.

For the full test workflow, allow sufficient free disk space for:

- index storage
- candidate output
- prediction output
- packaging
- validator scratch data

Approximately **15 GB or more of free disk space** is recommended for the complete workflow.

---

# Individual Pipeline Stages

Build or reuse the candidate index:

```powershell
python -m src.pipeline index `
  --test-dir $TestDir `
  --index-dir work/test_index
```

Run a 1,000-entity scalability check:

```powershell
python -m src.pipeline infer `
  --test-dir $TestDir `
  --index-dir work/test_index `
  --output-dir work/benchmark_1000 `
  --limit 1000
```

Validate that run:

```powershell
python -m src.pipeline validate `
  --test-dir $TestDir `
  --index-dir work/test_index `
  --output-dir work/benchmark_1000 `
  --limit 1000
```

Additional scalability runs:

```powershell
python -m src.pipeline infer `
  --test-dir $TestDir `
  --index-dir work/test_index `
  --output-dir work/benchmark_10000 `
  --limit 10000
```

```powershell
python -m src.pipeline infer `
  --test-dir $TestDir `
  --index-dir work/test_index `
  --output-dir work/benchmark_50000 `
  --limit 50000
```

Full inference:

```powershell
python -m src.pipeline infer `
  --test-dir $TestDir `
  --index-dir work/test_index `
  --output-dir output `
  --threshold 0.560
```

Full strict validation:

```powershell
python -m src.pipeline validate `
  --test-dir $TestDir `
  --index-dir work/test_index `
  --output-dir output `
  --expected-rows 1732544
```

`run` combines index preparation, inference and strict QA.

The resulting `run.json` records execution information such as:

- row counts
- throughput
- memory observations
- model hash
- input hashes
- feature order
- threshold
- runtime configuration

`strict_qa.json` binds validation success to the exact hashes of both generated output files.

---

# Production Candidate Generation

The scalable final candidate retriever is implemented in:

```text
src/disk_candidates.py
```

Source2 and Source3 are ingested sequentially into a disk-backed structure rather than being fully expanded in memory.

The production candidate system uses:

- SQLite-backed record storage
- exact normalized-name indexing
- exact address indexing
- exact combined-field retrieval
- name 4-grams
- name 6-grams
- address 5-grams
- address 8-grams
- fixed-size memory-mapped posting structures
- country-aware filtering
- bounded candidate expansion
- fuzzy reranking
- deterministic tie breaking

Common or overflowing hashed buckets are skipped to prevent unbounded expansion.

For each Source1 record, the retriever queries at most:

```text
6 name anchors
8 address anchors
```

Approximate retrieval first preselects up to:

```text
80 candidates per target source
```

Candidates are then fuzzy-reranked to at most:

```text
50 final candidates per target source
```

Exact name, address and full-field matches are also included with a bounded scan cap.

Each exact-key lookup is limited to:

```text
512 records per source/key
```

The serialized:

```text
output/candidate_pairs.tsv
```

contains the actual final candidate set presented to the model.

It is therefore **not an earlier raw blocking expansion**.

Every predicted entity in `matching_results.tsv` must also appear in the corresponding candidate set.

---

# Production Feature Engineering

The final CatBoost model expects exactly **16 features** in a fixed order:

```text
1.  name_ratio
2.  name_partial_ratio
3.  name_token_sort
4.  name_token_set
5.  name_exact
6.  name_length_ratio

7.  address_ratio
8.  address_partial_ratio
9.  address_token_sort
10. address_token_set
11. address_exact
12. address_length_ratio

13. same_country

14. max_name_similarity
15. max_address_similarity
16. combined_similarity
```

The validated production implementation lives in:

```text
src/features.py
```

and uses training-compatible normalization from:

```text
src/baseline_features.py
```

The saved-model feature schema and order are checked before inference.

This prevents accidental differences between model training and production inference.

Training-time text behavior includes:

- lowercase conversion
- NFKD normalization
- ASCII accent folding
- punctuation-to-space conversion
- collapsed whitespace

No inference-only legal-suffix expansion is introduced into the saved CatBoost model's feature calculations.

---

# Team Normalization Architecture

Two different normalization requirements were developed during the project.

They are intentionally kept separate.

## Nishant — Feature Engineering Normalization

Nishant's richer normalization and feature-engineering utilities are preserved in:

```text
src/normalize.py
src/nishant_features.py
```

The normalization module provides utilities including:

```text
normalize_text
remove_accents
normalize_name
normalize_name_ascii
normalize_address
normalize_address_ascii
tokenize
token_set
extract_numbers
has_value
```

These functions support Nishant's exploratory and extended feature-engineering work.

---

## Ramnesh — Blocking Normalization

Ramnesh's original blocking-specific normalization is preserved independently in:

```text
src/blocking_normalize.py
```

This normalization performs blocking-oriented substitutions such as:

```text
pvt   → private
ltd   → limited
corp  → corporation
inc   → incorporated

rd    → road
st    → street
ave   → avenue
blvd  → boulevard
apt   → apartment
bldg  → building
```

The historical TF-IDF blocker now explicitly imports:

```python
from .blocking_normalize import normalize_name, normalize_address
```

instead of importing from Nishant's feature-normalization module.

This separation prevents one teammate's normalization behavior from unintentionally changing another teammate's pipeline.

The normalization collision was resolved during final repository integration in **PR #5 — Separate feature and blocking normalization**.

---

# Retained TF-IDF Blocking Pipeline

Ramnesh's candidate-generation system remains preserved in:

```text
src/blocking.py
src/blocking_normalize.py
src/run_candidate_generation.py
src/candidate_recall.py
```

The retained system includes:

- country-partitioned candidate generation
- normalized exact matches
- TF-IDF name similarity
- TF-IDF address similarity
- combined similarity retrieval
- candidate-density analysis
- candidate-recall evaluation

This remains valuable historical and experimental project work.

It is **not the candidate retriever used by the frozen final production submission**.

The final scalable submission instead uses:

```text
src/disk_candidates.py
```

This distinction allows teammate work to remain available without changing the already validated production path.

---

# Model

The final production matcher uses a **CatBoost binary classifier**.

Model artifact:

```text
artifacts/experiments/baseline_v31/baseline_v1_catboost.cbm
```

Validated model SHA-256:

```text
101123e89499c693bed5621906854207b2136be6be73b46c0683a1f920764b7d
```

Final decision threshold:

```text
0.560
```

The inference layer verifies:

- expected feature names
- expected feature order
- binary class labels
- positive-class probability handling

before producing predictions.

---

# Internal Held-Out Validation

The final retriever was evaluated on an entity-disjoint held-out training subset.

These metrics are **internal diagnostics only** and are **not leaderboard scores**.

| Metric | Result |
|---|---:|
| Held-out Source1 entities | 1,000 |
| Final candidate pairs | 42,116 |
| Candidate link recall | 84.699607% |
| Any-match entity recall | 96.940928% |
| Complete entity recall | 64.978903% |
| Candidate oracle macro F0.5 | 0.931115 |
| Model macro F0.5 at threshold 0.610 | 0.829989 |
| Model macro F0.5 at threshold 0.560 | **0.831200** |

The final threshold selected for production was:

```text
0.560
```

No model retraining was performed during this final threshold adjustment.

The saved model remained unchanged.

---

# Final Full-Test Run

The production pipeline completed inference over the complete Source1 test dataset.

| Metric | Final Result |
|---|---:|
| Source1 entities | **1,732,544** |
| Final candidate pairs | **66,623,289** |
| Average candidates per S1 | **38.454024** |
| Predicted links | **5,449,697** |
| Zero-match entities | **152,443** |
| Zero-candidate entities | **8,204** |
| United States entities | **663,106** |
| France entities | **259,452** |
| India entities | **809,986** |

The recorded final run used:

```text
Workers: 6
CatBoost threads per worker: 1
Batch size: 500
Threshold: 0.560
```

The system preserves every Source1 row regardless of whether a candidate or match was found.

---

# Output Files

The production pipeline generates:

```text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

## `matching_results.tsv`

Contains final predicted entity matches.

Conceptually:

```text
source1_entity_id    matched_entity_ids
```

Multiple predicted IDs are serialized as comma-separated values.

A Source1 entity with no predicted match remains in the output with an empty match field.

---

## `candidate_pairs.tsv`

Contains the final candidates actually passed to the matching model.

Conceptually:

```text
source1_entity_id    candidate_entity_ids
```

Broad candidates discarded during reranking are not serialized.

The following invariant is enforced:

```text
predicted matches ⊆ final candidates
```

---

# Strict QA

The production QA implementation verifies the completed outputs against the original challenge inputs.

Checks include:

- exact Source1 coverage
- Source1 uniqueness
- exact headers
- row order
- duplicate IDs
- valid Source2 and Source3 target IDs
- empty candidate rows
- empty prediction rows
- country counts
- candidate/prediction subset invariant
- TSV readback
- output SHA-256 hashes

The final run passed strict validation for:

```text
1,732,544 / 1,732,544 Source1 entities
```

Result:

```text
STRICT QA: PASS
```

---

# Official Challenge Validation

The unchanged challenge-supplied validator is retained as:

```text
src/official_validate_submission.py
```

For the complete output, the repository provides a memory-safe wrapper:

```text
src/official_validation.py
```

Run:

```powershell
python -m src.official_validation `
  --test-dir $TestDir `
  --output-dir output `
  --scratch work/official_validation
```

The supplied validator materializes candidate sets in Python memory.

To keep memory bounded, the wrapper validates exhaustive disjoint Source1 partitions while separately enforcing global coverage and uniqueness.

The final validation processed:

```text
Source1 rows: 1,732,544
Partitions:   87
```

Result:

```text
OFFICIAL VALIDATOR: PASS
```

The validator transcript and JSON report are preserved in the verified submission package.

---

# Memory-Safe Validation Strategy

The strict streaming validator checks every Source1 identity and every candidate target ID using disk-backed structures.

The official validator is then executed over exhaustive partitions.

This provides two complementary validation layers:

```text
Strict streaming QA
        +
Official supplied validator
        =
Final validation gate
```

Validation success confirms structural correctness and submission consistency.

It must not be interpreted as a hidden test-quality or leaderboard score.

---

# Tests

Install development dependencies:

```powershell
python -m pip install -r requirements-dev.txt
```

Create a temporary test directory if needed:

```powershell
New-Item -ItemType Directory -Force work | Out-Null
```

Run:

```powershell
python -m pytest -q --basetemp=work/pytest-reproduction
```

The test suite covers functionality including:

- feature parity
- normalization behavior
- model schema rejection
- threshold boundaries
- multi-match entities
- zero-match entities
- France and unseen-country handling
- per-source candidate caps
- index identity
- candidate subset validation
- experiment tracking
- production pipeline behavior

Latest final integration verification:

```text
Production 16-feature import       PASS
Nishant extended-feature import   PASS
Ramnesh blocking import           PASS
Candidate-generation import       PASS
Python compilation checks         PASS
Full pytest suite                 35 / 35 PASS
Normalization collision           RESOLVED
```

---

# Scalability

The production index avoids storing an unconstrained candidate graph.

Index version 2 uses:

```text
Name n-grams:     4 and 6
Address n-grams:  5 and 8
```

Longer anchors help avoid saturation caused by very common short character sequences.

The pipeline performs bounded:

- anchor selection
- posting retrieval
- exact-key scans
- candidate preselection
- fuzzy reranking
- final top-K selection

The production implementation therefore avoids attempting an all-pairs comparison over the complete target universe.

---

# Packaging

After the source and documentation are committed and the working tree is clean, a reproducible package can be built using:

```powershell
python -m src.package_submission `
  --output-dir output `
  --zip Amazon_ML_Challenge_submission.zip
```

The packaging process checks:

- completed inference
- strict QA evidence
- official validation evidence
- source provenance
- output hashes
- model hash
- ZIP structure
- required files
- duplicate archive entries
- CRC integrity
- archived TSV hashes
- archived model hash

It also generates:

```text
SUBMISSION_READINESS.md
SUBMISSION_COMMIT.txt
```

inside the verified submission archive.

---

# Frozen Validated Submission

The already validated competition artifact was produced from:

```text
fe09fd107564af3ebc811df67ae558939fb9f175
```

The validated archive is:

```text
Amazon_ML_Challenge_submission.zip
```

Validated ZIP SHA-256:

```text
cf2050e01b852fd45c5642ef6f50ccb11e1fa69f0a6f61e9f8b770ebe5bf2470
```

Final output hashes:

### `matching_results.tsv`

```text
SHA-256:
69b33086389e2249226cda23a6bb5a87ffe62cc4abbd946eda994dc8145b973e
```

### `candidate_pairs.tsv`

```text
SHA-256:
baf1e2dfa5eee6292662b8be76478b439f8c03440288d4b89d6b179c3868f9ca
```

Later repository integration commits separate and preserve teammate utilities but do not change the already validated production outputs, trained model, or frozen submission archive.

The validated ZIP should therefore **not be regenerated merely because repository documentation or integration history changes after validation**.

---

# Repository Structure

```text
Amazon_ML_Challenge/
│
├── src/
│   ├── pipeline.py
│   ├── disk_candidates.py
│   ├── features.py
│   ├── baseline_features.py
│   ├── inference.py
│   ├── provenance.py
│   ├── package_submission.py
│   ├── official_validation.py
│   ├── official_validate_submission.py
│   ├── submission_qa.py
│   │
│   ├── normalize.py
│   ├── nishant_features.py
│   │
│   ├── blocking_normalize.py
│   ├── blocking.py
│   ├── run_candidate_generation.py
│   ├── candidate_recall.py
│   │
│   ├── experiment_tracker.py
│   ├── validate_final_model.py
│   ├── train_model.py
│   ├── thresholding.py
│   └── ...
│
├── tests/
│   ├── test_final_pipeline.py
│   ├── test_model_pipeline.py
│   ├── test_submission_qa.py
│   ├── test_experiment_tracker.py
│   ├── test_train_model.py
│   ├── test_thresholding.py
│   └── ...
│
├── documentation/
│   ├── methodology.md
│   ├── final_implementation_plan.md
│   ├── submission_checklist.md
│   ├── validator_workflow.md
│   └── experiment_notes.md
│
├── artifacts/
│   └── experiments/
│       └── baseline_v31/
│
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── Documentation_template.md
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

Large challenge datasets, temporary indexes, caches and generated development artifacts are intentionally excluded from version control.

---

# Production vs Experimental Components

The repository contains both final production code and useful historical/experimental team work.

## Production path

```text
disk_candidates.py
        │
        ▼
features.py
        │
        ▼
inference.py
        │
        ▼
pipeline.py
        │
        ▼
matching_results.tsv
candidate_pairs.tsv
```

## Retained Ramnesh blocking path

```text
blocking_normalize.py
        │
        ▼
blocking.py
        │
        ▼
run_candidate_generation.py
        │
        ▼
candidate_recall.py
```

## Retained Nishant feature utilities

```text
normalize.py
        │
        ▼
nishant_features.py
```

These paths are intentionally separated so that experimental work can remain available without silently changing the validated production model interface.

---

# Known Limitations

The candidate-retrieval system is intentionally bounded and therefore does not claim theoretically perfect recall.

Possible failure modes include:

- hashed n-gram collisions
- common-bucket exclusion
- fixed posting caps
- exact-key scan limits
- top-K truncation
- heavily corrupted business names
- heavily corrupted addresses
- businesses with unusually many records
- ambiguous businesses sharing names or addresses
- France not appearing in labeled model-training examples

These are documented algorithmic limitations.

Test ground truth is unavailable, so the repository does not claim knowledge of specific test false positives or false negatives.

---

# Data and Integrity Rules

The final solution:

- uses only challenge-provided records
- uses no external business registries
- uses no geocoding service
- uses no web-scraped entity database
- uses no external identity-resolution dataset
- does not read test ground truth
- preserves all Source1 entities
- supports arbitrary country labels
- separates training/evaluation code from test inference
- binds critical artifacts through SHA-256 hashes
- validates model feature order before inference
- validates prediction/candidate consistency before packaging

Historical validation values are reported only as internal diagnostics.

They are not presented as public leaderboard or hidden-test results.

---

# Final Repository Integration Status

All core teammate contributions are preserved in repository history.

Final integration includes:

```text
Ramnesh candidate/blocking contribution       INTEGRATED
Nishant normalization contribution            INTEGRATED
Sayem QA/tracking contribution                INTEGRATED
Production inference pipeline                 INTEGRATED
Blocking/feature normalization separation     RESOLVED
Pull request #5                               MERGED
Open pull requests                            NONE
```

Later experimental/refinement branches may remain available for research or post-submission development, but they are not required by the frozen validated competition artifact.

---

# Final Status

```text
Full test inference             PASS
Strict QA                       PASS
Official challenge validator    PASS
Production feature import       PASS
Nishant feature utilities       PASS
Ramnesh blocking pipeline       PASS
Candidate-generation imports    PASS
Python compilation              PASS
Test suite                      35 / 35 PASS
Submission ZIP inspection       PASS
Normalization collision         RESOLVED
```

The validated competition artifact is frozen and ready for submission.

---

## Important Submission Note

The validated competition ZIP is intentionally tied to its original verified production commit.

Updating this README or other repository documentation after validation does **not** require rebuilding the already verified submission archive.

For reproducibility, integrity and provenance, the existing validated artifact should remain unchanged unless an actual submission-blocking defect is discovered.
