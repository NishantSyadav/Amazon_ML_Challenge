# Final submission checklist

All items start unchecked. Complete them only from evidence for the selected run.
See [validator workflow](validator_workflow.md) and
[production reconciliation](production_reconciliation.md).

## A. matching_results.tsv

- [ ] UTF-8 TSV has exactly source1_entity_id, matched_entity_ids in that order; no extra/index columns or malformed rows.
- [ ] Exactly one row per original test S1; no missing, extra, or duplicate S1 rows.
- [ ] Empty matches use an empty field; no accidental NaN/None strings.
- [ ] Matched IDs are comma-separated, unique within each list, and valid original test S2/S3 IDs.
- [ ] France, other country labels, and zero-match S1 entities are included.
- [ ] Every matched ID belongs to that S1's final candidate list.

## B. candidate_pairs.tsv

- [ ] UTF-8 TSV has exactly source1_entity_id, candidate_entity_ids in that order; this is the grouped final format.
- [ ] Exactly one row per original test S1, including empty lists/all countries; no extra or duplicate S1 rows.
- [ ] Candidate IDs are valid test S2/S3 IDs, with no duplicates within a list or duplicate S1-candidate pairs.
- [ ] Lists contain the actual final candidates scored by the selected model.
- [ ] Generation is reproducible from recorded code, configuration, seeds, and source identity.
- [ ] Measured candidate recall is recorded with labeled validation split, denominator, and retrieval version; no test-label recall is claimed.
- [ ] Average candidates/S1, total count, density/caps, and zero-candidate count are recorded.
- [ ] Measured runtime, RAM, disk needs, and candidate-set scalability are recorded.

## C. Provenance

- [ ] Experiment ID, exact branch, and full executed Git commit are recorded.
- [ ] Candidate configuration/version and source identity are recorded.
- [ ] Feature configuration, numeric names/order, and production interface decision are recorded.
- [ ] Model configuration and artifact path/SHA-256 match the executed run.
- [ ] Actual threshold and all random seeds are recorded.
- [ ] SHA-256 hashes of both final files match retained validation evidence.
- [ ] Run reports/artifacts identify the selected experiment and preserve previous runs.

## D. Validation

- [ ] Strict internal QA PASS covers original source IDs, target existence, countries, duplicates, and match subset.
- [ ] Direct supplied validator PASS with --check-ids; transcript confirms ID checking ran. If it cannot finish, leave unchecked and record the unresolved gate/team decision described in the workflow.
- [ ] Production exhaustive wrapper PASS covers both complete outputs and the intended S1 count.
- [ ] Warnings reviewed; missing candidates, subset violations, and skipped ID checks are not accepted as direct readiness evidence.
- [ ] strict_qa.json, official_validation.json, official_validator.log, and the direct transcript are retained with output hashes.
- [ ] No final file has changed since validation; otherwise rerun checks.

## E. Packaging

- [ ] README, source code, selected production requirements, and completed methodology are included.
- [ ] Correct model artifact is included if required; SHA-256 matches inference.
- [ ] Both output/matching_results.tsv and output/candidate_pairs.tsv are included.
- [ ] Dataset, secrets, caches, scratch files, and unrelated local artifacts are excluded.
- [ ] Source/docs are committed and the production checkout is clean before packaging; executed-code provenance matches the run.
- [ ] Team has decided whether to include experiment_tracker.csv and general documentation/; current packaging omits them.
- [ ] Archive hierarchy, required entries, duplicate members, CRCs, and archived model/output hashes pass inspection.
- [ ] Archive SHA-256, manifest, embedded commit, readiness report, and verification records are retained.

## F. Submission control

- [ ] Before upload, count actual submissions on the Asia/Kolkata date; fewer than five used, reconciled with the portal.
- [ ] Exact filename/version, experiment, and hashes are recorded; retain matching_results.tsv within a versioned directory.
- [ ] Upload actually completed; portal acknowledgment and time retained.
- [ ] Only after confirmed upload, set status=submitted, upload date, and submission_number_for_day (1 through 5).
- [ ] Record leaderboard results afterward when available; unavailable scores stay blank.
- [ ] Preserve earlier submissions/configurations; final_selected marks an already uploaded selection without counting that upload twice.
