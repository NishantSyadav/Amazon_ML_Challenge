# Final production architecture

The executable path is `python -m src.pipeline run`. It builds/reuses a source-
identified disk index, scores bounded candidate batches, writes both official TSVs,
and performs strict streaming QA. Use the commands and resource controls in README.

## Data and normalization

Only challenge-provided TSVs are read. Source1 defines the complete output universe.
S2/S3 are ingested sequentially; UTF-8 text, IDs, and arbitrary country labels are
accepted. Saved-model features retain the original lowercase/NFKD ASCII accent
folding and punctuation-to-space normalization. France remains in both outputs.
Legacy suffix-expansion normalization is confined to legacy TF-IDF experiments.

## Candidate generation

Normalized S2/S3 records and exact keys live in SQLite. Name 4/6-grams and address
5/8-grams are hashed into 2^24 buckets. Each bucket stores at most 24 row IDs; counts
saturate at 25, and overflowing buckets are not queried. Index storage is fixed
(~1.56 GiB for postings plus counts) regardless of record count. Hash collisions
can suppress useful anchors and reduce recall, but do not bypass country filtering.

For each S1, select up to six rare name and eight rare address anchors. Retrieve
their bounded union using primary-key lookups, filter to the same country, and
preselect up to 80 per source by normalized anchor-hit coverage. Include exact
name/address/full-record candidates; each exact-key query has a 512-row/source cap.
Both-field exact hits receive first priority in reranking. Remaining ties use
0.6*name_similarity + 0.4*address_similarity, then deterministic name/address
similarities and candidate ID. Each similarity is the maximum of ratio,
partial_ratio, and token_set_ratio. Retain up to 50 candidates per source.

The serialized candidate output contains exactly these final candidates, all of
which are scored. Broad early candidates removed by reranking are not serialized.
No cross-source cardinality assumptions are imposed on final predicted links.

## Features and model

The production schema exactly matches the saved 16-feature CatBoost model:
name/address ratio, partial ratio, token-sort ratio, token-set ratio, nonempty exact
match, length ratio; same country; maximum name/address similarities; weighted
combined similarity. Similarities lie in [0,1]. Empty strings have zero similarity
and are not considered exact matches. Derived features are computed in float64
before one float32 conversion, matching CatBoost's training input conversion.
Model feature names/order and binary class labels [0,1] are checked explicitly.

The model is the existing CatBoost binary classifier, trained by the project using
600 maximum trees, depth 7, learning rate .05, seed42, and early stopping. The
saved model itself is included in the ZIP and bound to SHA-256 in the run report.
Threshold tuning uses macro entity F0.5 and includes zero-match entities. Final
threshold and measured metrics are recorded in the completed methodology template
and generated SUBMISSION_READINESS.md.

## Validation and execution

Original model fitting uses disjoint Source1 IDs: 4,000 training and 1,000 held-out
entities sampled from the validation-ID file. The new training-only diagnostic
reconstructs those exact held-out IDs and never fits the model again. Threshold
selection on that validation set is diagnostic, not a leaderboard/test score.
No ground-truth file is read by test inference.

Four process workers handle bounded 500-entity batches with at most eight pending
batches. The parent writes completed batches in original order. Empty entities
produce empty ID fields. Final files are promoted from `.partial` names only after
inference finishes. On interruption, rerun into a fresh output directory.

Strict QA reads S1 and both outputs in lockstep, uses a disk-backed uniqueness table,
verifies every candidate ID against the original target record index, verifies
match subset, counts countries/empty rows, and hashes final output files. The
unchanged official validator is run over exhaustive disjoint partitions to bound
its Python-set memory; global coverage/uniqueness and all ID-existence checks are
additionally enforced by strict QA. Its subset warning is treated as a hard failure.

## Known limits

N-gram and exact scan caps can miss matches. Top-K can miss businesses with more
than 50 target records in one source. Hash collisions are deterministic and may
lower recall. France is not represented in model training. There are no test
labels or verified leaderboard scores available to this pipeline. Full measured
counts, runtime, QA status, ZIP verification, and source SHA are in the generated
readiness report; no readiness claim is made from unit tests alone.
