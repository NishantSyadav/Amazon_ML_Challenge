# Final submission validation

Commands were checked against origin/om/final-submission at
fe09fd107564af3ebc811df67ae558939fb9f175. Run them from the selected production
checkout containing these modules; they are absent from this QA-only branch.
The former utils/validate_submission.py path is absent from this repository.
See [production reconciliation](production_reconciliation.md) before selecting
the pipeline. Documentation and unit tests do not constitute final validation.

## 1. Strict internal QA

After generating both final TSVs, run production streaming QA against the original
test sources and their matching index:

    python -m src.pipeline validate --test-dir dataset/test --index-dir work/test_index --output-dir output --expected-rows 1732544

The production run command also invokes strict QA. Retain strict_qa.json: it checks
source coverage, target-ID existence, duplicates, country counts, and match subset,
and binds the result to output hashes. Sayem's src/submission_qa.py exposes
additional small-file checks as Python functions; pass all three test-source paths
for ID/coverage validation. It materializes files and is not the full-scale path.

## 2. Direct supplied validator

One-line PowerShell command:

    python -m src.official_validate_submission --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids

Enable --check-ids for final direct QA and provide all three test-source files.
It requests S2/S3 existence checks and can require substantial RAM. Confirm that
the check actually ran: missing S2/S3 files can cause a warning and skip it even
with this flag. Record inability to finish; never label an incomplete check PASS.

The CLI accepts --matching, --candidate, --test-dir, and --check-ids. Path
arguments have defaults; explicit paths make the run reproducible. Exit code 0
prints "PASS — no blocking issues found. Safe to submit."; errors return 1.
Warnings alone do not produce a failing exit code:

- A missing candidate file can be skipped with a warning.
- Matches absent from the candidate set can produce only a warning.
- Target-ID existence checks are off without --check-ids; missing source files
  can prevent them even when requested.

A direct PASS with unresolved warnings is insufficient readiness evidence. Review
the complete transcript and require both final files and all required checks.

## 3. Production exhaustive wrapper

    python -m src.official_validation --test-dir dataset/test --output-dir output --scratch work/official_validation

This calls the supplied validator over exhaustive 20,000-row partitions, checks
global S1 uniqueness and original row order, requires both outputs, and turns
subset warnings into failures. Its default expected count is 1,732,544 S1 rows.
--expected-rows and --batch-size are configurable; justify any alternate universe.
Scratch must differ from input/output directories and needs free disk space.

The wrapper intentionally uses check_ids=False. It relies on separate strict QA
for target existence and does not itself run or verify that strict QA. Require
strict_qa.json on the same output hashes. Retain official_validation.json and
official_validator.log. "Official validation PASS" appears only after every
partition and the final row count succeed.

Final readiness requires strict QA plus production validation, warning review,
and evidence tied to exact files. Record direct ID-check results separately.
If the full direct check cannot finish due to RAM, leave that check unresolved
and obtain a documented team decision on strict-ID-check plus exhaustive-wrapper
evidence; do not relabel it direct PASS.

## Final TSV contracts

Both files use UTF-8, tabs, exactly two columns, no index column, and exactly one
row per test S1, including France, future country labels, and empty entities.

| File | Exact ordered columns | List value |
| --- | --- | --- |
| matching_results.tsv | source1_entity_id, matched_entity_ids | Comma-separated matches; empty field for no match |
| candidate_pairs.tsv | source1_entity_id, candidate_entity_ids | Comma-separated final scored candidates; empty field for none |

Require unique S1 rows, unique targets within each list, and valid test S2/S3 IDs.
Every match must be in that S1's candidate list. This grouped final candidate
format differs from the five-column long retrieval/feature input.

## Evidence and safeguards

Retain experiment ID, branch/commit, model/configuration, threshold, seeds,
candidate/feature configurations, model SHA-256, output SHA-256 values, and
validation transcripts/reports. Revalidate changed outputs. Follow the
[submission checklist](submission_checklist.md) before packaging and upload.

Use only challenge data: no external entity lookup, geocoding, registries, or
augmentation. Preserve all countries. Check the five-submissions-per-day
allowance before upload and record actual uploads afterward. Public leaderboard
scores are evidence, not ground truth. Unknown results remain blank.
