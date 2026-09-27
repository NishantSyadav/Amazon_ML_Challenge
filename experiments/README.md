# Experiment and submission records

Record each real configuration in experiment_tracker.csv and use
[experiment notes](../documentation/experiment_notes.md) for evidence and details.
Unknown metrics, thresholds, hashes, submission numbers, and leaderboard scores
remain blank. No example or estimated results belong in the live tracker.

## Status and actual uploads

| status | Meaning | Counts toward daily uploads? |
| --- | --- | --- |
| experiment | Work in progress or diagnostic experiment | No |
| prepared | Output files exist; final validation is incomplete | No |
| validated | Required QA/validation completed; no upload confirmed | No |
| submitted | Upload actually completed and acknowledgment retained | Yes |
| final_selected | An already uploaded submission chosen as final | Yes, the same upload once |

A file path, validator PASS, or reserved submission number never proves upload.
Only the exact lowercase statuses submitted and final_selected count.
If a final selection has not been uploaded, retain prepared/validated and note
the selection pending upload. Do not mark it final_selected prematurely.

Fill submission_number_for_day (1 through 5) only after confirmed upload; leave it
blank for experiment/prepared/validated rows. The append helper rejects a number
on an unsubmitted row. Check the portal and daily allowance before uploading;
bookkeeping cannot prove portal state or validate files itself.

## Dates and identity

Use date as YYYY-MM-DD in Asia/Kolkata (IST, UTC+05:30). All daily counts use this
local date, not UTC or a timestamp string. For submitted/final_selected rows it
is the confirmed upload date. For other rows use the run date. Retain run and
upload timestamps separately in notes as ISO 8601 with +05:30, including the
original run date if upload occurs later. Date/time formatting is a convention;
the helper compares date strings and does not convert timezones.

Assign unique experiment IDs (E001, E002, ...). A change to candidates, features,
model configuration, or threshold requires a new ID. Record branch and full
executed Git commit for reproducibility; distinguish later documentation commits.
Never replace a prior uploaded run's configuration, paths, or measured evidence.

A prepared/validated row may later be updated in place with confirmed upload
status/date/number and acknowledgment; retain its original run details in notes.
Do not append the same upload again when selecting it as final. The helper only
appends new IDs; it has no update API. Review any manual lifecycle update for
unique experiment IDs, daily limits, and duplicate upload numbers. A genuine
repeat upload needs a distinct record ID linked to the original configuration.

## Field conventions

Use the explicit fields below for final records. Legacy fields remain compatible;
if also filled, their values must agree with the canonical fields.

| Canonical fields | Record |
| --- | --- |
| git_commit, branch | Full executed SHA and branch |
| candidate_config | Method/version, source identity, K/caps, thresholds, seeds; immutable config path/hash or complete parameters |
| feature_config | Implementation/version, exact numeric names/order, normalization, config path/hash |
| model_name, model_config | Model type, actual parameters/seeds, model artifact path and SHA-256 |
| decision_threshold | Actual selected threshold and tuning evidence reference |
| submission_file, candidate_file | Versioned paths to matching_results.tsv and candidate_pairs.tsv; may be populated before upload |
| validator_status | PASS only when required final QA evidence is complete; otherwise blank/pending/failure with explanation |
| validation_precision, validation_recall, validation_f0_5 | Measured values with aggregation convention/split in notes; F0.5 is macro by S1 |
| blocking_recall, avg_candidates_per_s1 | Measured retrieval metrics with split, denominator, and candidate version |
| runtime, memory_usage | Values with units, scope, hardware, and peak-memory measurement convention |
| public_leaderboard_score, private_leaderboard_score | Actual portal results when available |
| notes | Changes, evidence paths, hashes, timestamps, limitations, and unresolved gates |

Legacy aliases include model -> model_name, model_parameters -> model_config,
threshold -> decision_threshold, and leaderboard_score -> public_leaderboard_score.
features_used/candidate_method/top_k remain useful summaries; configurations hold
the detailed definitions. No new columns are needed for this convention.

Use labeled entries in notes (or link to the matching experiment-notes section):
model_artifact_path, model_sha256, matching_sha256, candidate_sha256,
run_timestamp_ist, upload_timestamp_ist, strict_qa_report, direct_validator_log,
production_validation_report, and upload_acknowledgment. Keep reports/artifacts
in their versioned locations; do not commit datasets, models, or generated TSVs.

Follow [validator workflow](../documentation/validator_workflow.md): a direct
validator PASS with warnings is insufficient. The tracker checks the supplied
PASS text but does not inspect reports, hashes, or portal acknowledgment.
Use local macro F0.5 and treat public leaderboard feedback as experimental
evidence, not ground truth.

See [production reconciliation](../documentation/production_reconciliation.md)
for the pending 26-input versus saved 16-feature model decision and packaging gaps.
