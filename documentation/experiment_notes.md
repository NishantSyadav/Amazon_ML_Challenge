# Experiment Notes

Record important experiments, observations, errors, and decisions here.

An **experiment** is one tested configuration, whether or not submitted. A
**validated experiment** has output files that pass internal QA and the
official validator. A **leaderboard submission** is a validated experiment
whose `matching_results.tsv` was actually uploaded. The **final selected
submission** is the exact experiment and output chosen for the final package.

Use local macro F0.5 as the evaluation target. Do not optimize blindly against
the public leaderboard: it is not final ground truth, and final ranking uses
the private leaderboard. Candidate-generation or threshold changes make a
distinct configuration and require a new experiment ID.

Copy this template for each experiment. Leave unknown metrics blank rather
than estimating them.

## Experiment template

Experiment ID:
Date:
Git Commit:
Branch:

Candidate Method:
Top-K:
Candidate Configuration:
Features:
Feature Configuration:
Model:
Parameters:
Model Configuration:
Threshold:

Blocking Recall:
Average Candidates/S1:
Validation Precision:
Validation Recall:
Validation Macro F0.5:

Runtime:
Memory:

Leaderboard Submitted:
Validator Status:
Matching Output File:
Final Candidate File:
Submission Number for Day:
Public Leaderboard Score:
Private Leaderboard Score:

What changed from previous experiment:
Result:
Observations:
Errors:
Next step:
