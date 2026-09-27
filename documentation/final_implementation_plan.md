# Final submission implementation plan and design

Goal: regenerate both official TSVs for every test S1, with bounded memory, exact
model feature compatibility, audited IDs, official validation, and a reproducible ZIP.
The user's master prompt is the specification and explicitly authorizes autonomous
implementation, direct work in this checkout, commits, and final-branch push.

Architecture: SQLite stores normalized S2/S3 records and indexed exact keys. A
fixed-size, memory-mapped hashed n-gram inverted index retains only bounded posting
lists; overfull buckets are excluded, never expanded. Query rare anchors, retain a
bounded preselection per source, then RapidFuzz rerank and preserve exact candidates.
Exact hits are streamed and reranked with a documented per-source cap to prevent
common names/addresses causing unbounded expansion. Score final candidates in batches
with the saved 16-feature CatBoost, and write one row per input S1 including empties.
Country labels are unrestricted. No test inference code opens training or truth files.

Alternatives considered: existing sparse TF-IDF is known too slow; full V3.1 in-memory
heaps cannot fit this machine; repeated source scans per S1 batch multiply I/O and
normalization cost. A reusable disk index allows bounded queries and restartable runs.

- [x] Preserve Git history, fetch, merge QA; exclude failed experiment.
- [ ] Add failing tests: accents/empty text, exact feature parity, schema rejection,
  France and unseen countries, multiple/zero matches, overflow and subset failures.
- [ ] Implement index, model inference, streamed output, strict disk-backed QA.
- [ ] Run full suite and real 1k/10k/50k benchmarks against full test targets.
- [ ] Measure held-out training retrieval/threshold where runtime permits; do not
  present historical metrics as this retriever's metrics or leaderboard results.
- [ ] Execute full inference, then strict QA and supplied official validator.
- [ ] Complete README, pinned dependencies, methodology, readiness report.
- [ ] Review, commit, push; package model/code/docs/outputs and inspect ZIP.

Review focus: candidate overflow, duplicate source IDs, stale index identity,
interrupted outputs, zero candidates, threshold boundary, feature column order.

Rulings: preserve training-time ASCII accent folding in model features; changing it
would invalidate the saved model. Unicode source text is read as UTF-8; accents fold
consistently and every country passes through. Hash collisions may reduce retrieval
recall; reranking/model and country checks guard candidate quality. Final retrieval
quality must be measured separately from historical V3.1 diagnostics.

Baseline: 16 pytest tests passed, plus legacy import-time assertion scripts collected.
Initial pytest setup failed on inaccessible Windows temp; project-local basetemp fixes it.
Ruling: retained a bounded exact-key scan (512 per source per key) instead of
unbounded streaming reranking. Both-field exact lookup runs before broad keys.
Reason: repeated common names can otherwise create near-Cartesian CPU cost.
Cost: a true match beyond this cap can be missed; overflow count is reported and
must be inspected in real benchmarks. Final output caps at 50 candidates/source.
Review fixes: per-source anchor preselection after country filtering, per-source
exact scans, full-precision derived features, stale QA report invalidation, and
output hashes. All 28 tests pass, including real CatBoost multi-batch inference.
Reviewer's runtime/recall/threshold/packaging judgments remain pending measurements.
Milestone: 32 tests pass. Short-only index built all test targets in 475.5s and all
training targets in 493.5s, RSS ~1.73 GiB. Corrected SQLite planner regression:
including country with a many-row-ID IN list selected a country-wide scan. Fetch
bounded primary keys first, then filter country in Python (1–5ms sampled retrieval).
Short-only held-out result: recall .666480, macro F0.5 .746901 at .610; insufficient
relative to historical V3.1. Ruling: include longer name6/address8 anchors alongside
short grams, keeping the same bounded index capacity. Cost: added indexing time;
benefit to be measured, not assumed. Production source index version is now 2.
Parallel workers are bounded and ordered; serial/parallel byte-equivalence test
passes. Official partition wrapper calls the unmodified supplied validator and
checks global identity; this preserves the full validation rules within RAM limits.
Second review fixes: source-aware exact index keys, isolated validator scratch
subdirectories with collision rejection, source/model fingerprints captured during
inference, and mandatory matching held-out evidence at packaging. Reproduced tests
failed before fixes; complete suite now35 passed. Fresh production imports pass,
and pip check reports no broken requirements. CatBoost metadata confirms Apache2.0.

Final retrieval held-out:42,116 candidates/1000 S1; link recall.84699607; any-match
recall.96940928; complete recall.64978903; oracle.93111451. Model.82998919 at.610;
best.83120008 at.560. Ruling: choose.560 for the changed distribution; risk is
validation-set threshold overfitting, disclosed as internal tuning, not leaderboard.

Scalability: v2 1000=7.2s;10000=25.7s;50000=70.4s with6 workers.50k strict QA passed,
1,921,941 candidates,156,934 links,4,420 empties,232 zero-candidate S1,7,485 France.
Peak summed RSS2.61GiB; minimum observed available RAM4.13GiB. Full run launched
with6 workers, threshold.560, batch500; completion and final validators pending.
