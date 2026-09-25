# Official validator workflow

Official validator integration pending — add the competition-provided validator when available.

When the official validator is available, record its actual path and command
here. Run it on the final matching output and candidate-pair output before
uploading. Resolve any failures and rerun it on the files intended for upload.

Expected checks:

- Matching output exists.
- Candidate-pair output exists.
- Every required Source1 is represented exactly once.
- There are no duplicate Source1 entities.
- Predicted matches are a subset of candidate pairs.
- Schema and format pass official validation.

Record the validation result with the experiment ID and Git commit. Do not
mark the checklist complete until the competition-provided validator passes.
