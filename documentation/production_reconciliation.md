# Production documentation reconciliation

Reviewed read-only from origin/om/final-submission at
fe09fd107564af3ebc811df67ae558939fb9f175:
README.md, documentation/methodology.md, Documentation_template.md,
requirements.txt, requirements-dev.txt, src/official_validate_submission.py,
src/official_validation.py, and src/package_submission.py.
These implementations are not merged into this QA branch. Their commands must
run from the selected production checkout; this note does not select that branch
or claim any new measured result.

## Pending production-interface decision

Nishant's integration at 07a4ab117fbc529dbe2e663efa8d8fe9d2f9f47e produces
26 numeric inputs plus three identifier/source columns. Om's production documents
describe a saved 16-feature CatBoost classifier with asserted feature names/order
and its own training-compatible normalization.

These interfaces are not automatically interchangeable. Do not feed the 26-input
output into the saved model or document it as the final production path without
Om's explicit compatibility decision. Record the selected implementation, schema,
normalization, model artifact/hash, and real-data validation evidence together.
Nishant's real-data smoke test remains pending; no outcome is invented here.

## Existing production documentation

Reuse Om's README installation, pipeline, validation, and packaging instructions.
Its methodology and Documentation_template.md already describe production
retrieval, model features, held-out evaluation, and limitations. Reported results
there are existing team claims to reconcile with retained artifacts; they have
not been remeasured by this documentation change.

Clarify four workers in README/methodology versus six in the template as defaults
versus actual run configuration using run.json. Ask Om to confirm training and
negative-selection details. Keep historical retrieval metrics distinct from final
retrieval metrics and internal tuning scores distinct from leaderboard results.
Sayem's older methodology headings should be reconciled after pipeline selection,
not populated with a conflicting model narrative now.

## Environment and reproducibility

Use requirements.txt and requirements-dev.txt from the selected production commit.
Om's runtime manifest pins pandas, NumPy, RapidFuzz, CatBoost, scikit-learn, SciPy,
and psutil; the development manifest adds pytest and legacy sparse-dot-topn.
No competing manifest is introduced on this QA branch. Confirm installation on
the intended machine; this task does not install or claim a fresh environment test.

Reproduction additionally requires the actual saved model, model/configuration
provenance, source identity, seeds, feature order, validation reports, and final
output hashes. An installed environment alone does not establish reproducibility.

## Packaging alignment and pending team decision

The inspected production command is:

    python -m src.package_submission --output-dir output --zip Amazon_ML_Challenge_submission.zip

Run it only after final evidence exists on the selected production checkout.
The implementation reads run.json, strict_qa.json, official_validation.json,
and the model; requires a full unlimited run with 1,732,544 S1 rows, passing QA,
matching counts/output hashes, model hash consistency, and a clean Git checkout.
It also invokes provenance verification before creating the package.

The archive contains both output TSVs, tracked Python source/tests, README,
requirements, the saved model, Documentation_template.md, generated readiness and
commit records, and verification reports/logs. It checks hierarchy, duplicate
members, CRCs, and archived TSV/model hashes, then writes a ZIP manifest/hash.
Its explicit file selection excludes the raw dataset and index/scratch caches;
the team must still inspect for secrets and unintended local content.

It does not currently include experiments/experiment_tracker.csv or the general
documentation/ directory. Whether to include those is a pending team packaging
decision; the script is unchanged. It also does not enforce Sayem's separate
direct --check-ids PASS checklist item: track that gate and any documented team
decision separately from what packaging mechanically enforces.

Final output availability, model identity, measured metrics, successful validation,
archive inspection, and actual upload remain evidence-dependent. No placeholder
is a completed readiness gate.
