# Experiments

This folder contains the experiment tracker, notes, and results. Record each
experiment as one row in `experiment_tracker.csv` and use
`documentation/experiment_notes.md` for detailed observations.

Assign IDs in sequence: E001, E002, and so on. Make one major change per
experiment so its effect can be understood. Record the exact Git commit and
parameters used for every experiment. Run local validation before any
leaderboard submission.

Record measured metrics only; never invent or overwrite them. Leave the
leaderboard score blank when an experiment has not been submitted. If a run
fails or a metric is unavailable, record the status and explain it in notes.

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
