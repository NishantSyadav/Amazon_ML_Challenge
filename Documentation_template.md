# ML Challenge 2026: Business Entity Resolution

**Project identifier:** Amazon_ML_Challenge  
**Repository contributors (Git author identifiers):** om-srivastav, RAMNESH007,
sayem23ai, NishantSyadav  
**Submission preparation date:** 2026-09-27

## 1. Executive Summary

We resolve every Source1 entity against Source2 and Source3 with bounded candidate
retrieval and a CatBoost pair classifier. A disk-backed record store and fixed-size
n-gram postings avoid materializing the full candidate expansion. Both official
outputs retain every Source1 row, including singletons and French entities.

## 2. Methodology

### 2.1 Problem Analysis

Names and addresses contain reordered components, spelling errors, abbreviations,
missing fields, and diacritics. Training covers US/India; test additionally covers
France. Country is treated as an open string label. Precision matters because the
metric is macro entity F0.5, including correct and incorrect singleton predictions.

### 2.2 Solution Strategy

**Approach:** bounded blocking + supervised pair classifier + entity aggregation.
All data comes from the challenge. We use no geocoding, external registries, APIs,
web scraping, pretrained entity lookup, or external datasets. Ground truth is read
only by separate training/evaluation modules, never by test inference.

## 3. Candidate Generation

SQLite stores normalized S2/S3 records and indexes `(country, source, name,
address)` and `(country, source, address, name)`. Fixed memory-mapped arrays hold
hashed name 4/6-grams and address 5/8-grams. There are 2^24 buckets with at most
24 postings each; overfull buckets are skipped. Query up to six rare name and eight
rare address anchors, use bounded primary-key record fetches, filter country, and
retain up to 80 approximate candidates per source before fuzzy reranking.

Exact name, exact address, and combined exact matches are also retrieved, bounded
at 512 scanned matches per source/key. Full name-and-address exact matches receive
priority; other candidates rank by 0.6*name + 0.4*address similarity. Each similarity
is the maximum of RapidFuzz ratio, partial ratio, and token-set ratio. Final top-K
is 50 per source, and all final candidates are scored by CatBoost.

`candidate_pairs.tsv` contains this exact scored candidate set, not an earlier
blocking set. Final candidate totals and measured density are reported in
`SUBMISSION_READINESS.md` and `verification/run.json` in the archive.

The original sampled V3.1 anchors alone lost recall when applied to the full target
universe. Longer anchors restored useful rare substrings without increasing posting
capacity. Hash collisions, common-bucket exclusion, and caps can still miss links;
we quantify recall on held-out training entities rather than assuming no loss.

## 4. Matching Model

**Model:** existing CatBoost binary classifier,
`artifacts/experiments/baseline_v31/baseline_v1_catboost.cbm`.
The saved model has 600 trees; training configuration was depth 7, learning rate .05,
seed 42, maximum 600 iterations, early stopping, and no automatic class weighting.
CatBoost is Apache License 2.0; this is a small task-trained tree model, far below the
8-billion-parameter limit. No foundation model is used.

**Feature order (asserted against model.feature_names_):**

1. name_ratio
2. name_partial_ratio
3. name_token_sort
4. name_token_set
5. name_exact
6. name_length_ratio
7. address_ratio
8. address_partial_ratio
9. address_token_sort
10. address_token_set
11. address_exact
12. address_length_ratio
13. same_country
14. max_name_similarity
15. max_address_similarity
16. combined_similarity

Training normalization is retained: lowercase, NFKD accent folding to ASCII,
punctuation-to-space, and collapsed whitespace. Empty strings are not exact matches
and have zero similarity. Derived features use float64 arithmetic before conversion
to CatBoost float32 input. No inference-only suffix substitutions are introduced.

**Final threshold: 0.560.** It was selected by macro F0.5 optimization after the
retrieval distribution changed. On the same 1,000 held-out entities, threshold 0.610
scores 0.829989; threshold 0.560 scores 0.831200. The existing model was not retrained.

## 5. Results and Error Analysis

The original entity split is reconstructed exactly: sample 5,000 IDs from the stored
validation-ID list using random_state 42, then split 80/20 with random_state 123.
The saved model's 4,000 training S1 entities and 1,000 validation S1 entities are
disjoint. Zero-candidate entities remain in the evaluation universe. The final
retriever is evaluated against all 10,320,219 training S2/S3 records.

| Final retriever diagnostic | Value |
|---|---:|
| Held-out S1 entities | 1,000 |
| Final candidate pairs | 42,116 |
| Candidate link recall | 84.699607% |
| Any-match entity recall | 96.940928% |
| Complete entity recall | 64.978903% |
| Candidate oracle macro F0.5 | 0.931115 |
| Model macro F0.5 at 0.610 | 0.829989 |
| Model macro F0.5 at 0.560 | 0.831200 |
| Exact-key overflow events in validation | 0 |

These are **internal sampled validation diagnostics, not leaderboard results**.
The threshold was tuned on this validation set, so its score is not an unbiased
estimate from a separate final test set. France has no labeled training examples.

Historical sampled V3.1 results were link recall 83.8402%, any-match recall 95.5829%,
complete recall 67.9337%, candidate oracle 0.919427, model 0.816031 at 0.610, and roughly
106.06 candidates/S1. Historical candidate diagnostics and current held-out metrics
need not use the same entity subset; do not interpret them as a controlled full-test
leaderboard comparison. Short-only full-scale retrieval was rejected after scoring
0.746901, demonstrating why full-universe validation mattered.

Expected failure modes include similar businesses at shared addresses, misleading
partial-name overlap, highly corrupted records with no retained anchors, and large
many-record entities exceeding top-K. These are algorithmic limitations; test truth
is unavailable, so no claims about specific test false positives/negatives are made.

## 6. Execution, QA, and Reproducibility

The single entry point is `python -m src.pipeline run`. Source2/3 indexing streams
records. Inference uses six worker processes for the final run, one CatBoost thread
each, 500-entity batches, and at most two pending batches per worker. Results are
written in Source1 order, with comma-separated deduplicated IDs and empty fields for
zero matches. Source/model SHA-256 manifests bind outputs to the executed code.

Scalability gates passed at 1,000, 10,000, and 50,000 Source1 entities against the entire
test target universe. The 50,000 run took 70.4 seconds, scored 1,921,941 candidates,
preserved 7,485 France rows, and had no exact-key overflow. Sum-of-process RSS peaked
at 2.61 GiB (shared mapped pages can be counted multiple times); system available RAM
remained at least 4.13 GiB. Full-run counts and memory observations are recorded in the
readiness report once all completion gates pass.

Strict QA checks exact row counts and headers, original S1 identities, global
uniqueness, country coverage, empty rows, duplicate/invalid target IDs, all target-ID
existence, TSV readback, and the candidate-subset invariant. The supplied official
validator is unchanged (SHA-256
`f96f59934383a15095914f507620c078c474e8f9c60e864b2d051ed173a22dfc`).
It runs over exhaustive disjoint 20,000-row partitions to bound its in-memory sets;
global Source1 uniqueness/order are checked separately. Its subset warning is
promoted to a failure. Strict disk-backed QA supplies the optional target-ID check.
Official transcripts and hash-bound reports are included in `verification/`.

## Appendix: Code Artifacts

The ZIP contains `output/matching_results.tsv`, `output/candidate_pairs.tsv`, this
methodology, the readiness report, and `code/business_entity_resolution/` with
`src/`, tests, README, pinned requirements, and the trained model. There is no extra
outer directory. Packaging verifies required entries, CRCs, and archived model/TSV
hashes. Failed experiments, raw data, index caches, and temporary files are excluded.

See the packaged README for exact installation, indexing, inference, validation,
and packaging commands. Retained legacy experiment/training code and teammate QA
history remain available; the scalable production modules are explicitly named.
The project branch is `om/final-submission`; the exact source commit is embedded in
`SUBMISSION_COMMIT.txt` and `SUBMISSION_READINESS.md`.
