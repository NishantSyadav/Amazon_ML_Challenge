# Experiment notes

Use one section per real experiment/configuration. Unknown measured values stay
blank; this file contains a template, not fabricated results. Follow the field
and lifecycle conventions in [experiments README](../experiments/README.md).

Prepared means files exist, validated means required checks passed, and submitted
means an upload actually completed. Only submitted and an already uploaded
final_selected record count toward the daily limit. A path or validator PASS does
not establish upload. Fill the daily number only after confirmed upload.

Use YYYY-MM-DD dates and Asia/Kolkata (IST, UTC+05:30) timestamps. Tracker date
becomes the actual upload date for submitted rows; preserve the original run
date/time here. Keep configuration and evidence for prior runs unchanged.
Macro F0.5 is the local target; leaderboard scores must come from the portal.

## Experiment template

Experiment ID:
Status (experiment / prepared / validated / submitted / final_selected):
Run Date (YYYY-MM-DD, IST):
Run Timestamp (ISO 8601, +05:30):
Executed Git Commit (full SHA):
Executed Branch:
Later Documentation Commit (if different):

Candidate Method / Version:
Candidate Configuration Path / Hash or Parameters:
Candidate Source Identity / Split:
Top-K / Caps / Retrieval Thresholds:
Feature Configuration / Implementation Version:
Exact Numeric Feature Names / Order:
Normalization Version:
Production Interface Decision / Owner Confirmation:
Model Name / Type:
Model Configuration / Parameters:
Random Seeds:
model_artifact_path:
model_sha256:
Decision Threshold:
Threshold Selection Evidence:

Blocking Recall / Definition / Denominator:
Average Candidates per S1:
Total Candidates / Density / Zero-Candidate Count:
Validation Precision / Aggregation:
Validation Recall / Aggregation:
Validation Macro F0.5 / Split:
Runtime / Units / Scope:
Memory / Units / Measurement Method:
Hardware / Disk Requirements:

Matching Output Versioned Path:
Candidate Output Versioned Path:
matching_sha256:
candidate_sha256:
Strict QA Status / Report Path:
Direct Validator Status / ID-Check Confirmation / Transcript:
Production Wrapper Status / Report / Transcript:
Warnings / Resolution / Remaining Gates:
Output Hashes Matched Across Reports:

Upload Actually Completed:
Upload Date (YYYY-MM-DD, IST):
Upload Timestamp (ISO 8601, +05:30):
Portal Acknowledgment:
Submission Number for Day (blank until uploaded):
Public Leaderboard Score (when available):
Private Leaderboard Score (when available):
Final Selection / Archive Path / Hash / Manifest:

What changed from previous experiment:
Measured result:
Observations / Errors:
Pending team decisions:
Next step:
