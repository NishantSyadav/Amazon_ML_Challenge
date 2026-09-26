# Experiments

This folder contains the experiment tracker, notes, and results. Record each
experiment as one row in `experiment_tracker.csv` and use
`documentation/experiment_notes.md` for detailed observations.

Assign a unique ID to each actual experiment/configuration: E001, E002, and
so on. Make one major change per experiment so its effect can be understood.
Create a new experiment ID whenever the configuration changes, including
candidate generation or the decision threshold. Record the exact Git commit,
branch, model/configuration, feature configuration, candidate-generation
configuration, and threshold. Run internal QA and the official validator on
the actual output files before any leaderboard upload.

Record measured metrics only; never invent them. Validation-only experiments
may have no leaderboard score. If a run fails or a metric is unavailable,
record the status and explain it in notes. The public leaderboard score is
experimental evidence, not final ground truth; final ranking depends on the
private leaderboard. Local evaluation targets macro F0.5.

Each leaderboard submission must record its experiment ID, exact Git commit,
model/configuration, feature configuration, candidate-generation configuration,
decision threshold, validator result, upload file (`matching_results.tsv`),
and `submission_number_for_day` (1 through 5). Record the final
`candidate_pairs.tsv` path in `candidate_file`. Count submissions by date
before uploading: at most five are allowed per day. Preserve the matching
output and its final `candidate_pairs.tsv` together. Never overwrite an old
submission row with a newer result; fill only fields that become available
later, such as leaderboard scores. The final selected submission must be
traceable to its code, configuration, and both output files.

The original tracker columns remain for compatibility. Use the explicit new
columns for submission records; leave values blank until known. In `status`,
use `experiment`, `validated`, `submitted`, or `final_selected` as appropriate.

Each experiment should record:
- Experiment ID
- Candidate generation method
- Top-K
- Features used
- Model
- Threshold
- Precision
- Recall
- Macro F0.5
- Runtime
- Notes

For submitted experiments, also record `branch`, `model_name`, `model_config`,
`feature_config`, `candidate_config`, `decision_threshold`,
`validator_status`, `submission_file`, `candidate_file`, `submission_number_for_day`,
`public_leaderboard_score` when available, and `private_leaderboard_score`
when available.
