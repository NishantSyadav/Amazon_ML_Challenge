# Official submission validator workflow

## Run the official validator

Run the validator against the final files intended for submission.

Linux/macOS:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

PowerShell:

```powershell
python utils/validate_submission.py `
  --matching output/matching_results.tsv `
  --candidate output/candidate_pairs.tsv `
  --test-dir dataset/test
```

## `output/matching_results.tsv`

Use TSV format with these columns:

| Column | Requirement |
| --- | --- |
| `source1_entity_id` | Include exactly one row for every test Source 1 entity, with no duplicate Source 1 rows. |
| `matched_entity_ids` | Use comma-separated IDs. Leave the value empty when there is no match. |

Every matched ID must be a valid Source 2 or Source 3 test ID. Match lists must contain no duplicate IDs and no Source 1 IDs.

## `output/candidate_pairs.tsv`

Use TSV format with these columns:

| Column | Requirement |
| --- | --- |
| `source1_entity_id` | Include exactly one row for every test Source 1 entity, with no duplicate Source 1 rows. |
| `candidate_entity_ids` | Use comma-separated IDs. Leave the value empty when no candidates exist. |

Every candidate ID must be a valid Source 2 or Source 3 test ID. Candidate lists must contain no duplicate IDs. This file must contain the **final candidate set actually sent to the matching model**, not an earlier raw blocking output.

For each Source 1 entity, every ID in `matched_entity_ids` must also appear in that entity's `candidate_entity_ids` list.

## Submission QA procedure

1. Generate both TSV files from the final pipeline output.
2. Verify the column names, tab separation, row coverage, ID validity, and comma-separated lists described above.
3. Run the official validator on both final files.
4. If validation fails, do not upload. Fix the issues and rerun the validator until it reports **PASS**.
5. After PASS, record the experiment and submission details, including:
   - experiment ID
   - Git commit
   - model/configuration
   - decision threshold
   - candidate-generation configuration
   - validator result
   - leaderboard result, when available

## Challenge safeguards

- Training countries include the US and India; the test set additionally includes France. Keep country handling open-set and do not hard-code only the US and India.
- Include every test Source 1 entity in the submission, including entities with no matches or candidates.
- Do not use external entity lookup, geocoding APIs, government registry lookup, or internet/external data augmentation.
- Make no more than five leaderboard submissions per day.
- Preserve version history of submissions.
- Do not treat public leaderboard feedback as ground truth.

## Final package checklist

- [ ] `output/matching_results.tsv`
- [ ] `output/candidate_pairs.tsv`
- [ ] `code/business_entity_resolution/src/`
- [ ] `code/business_entity_resolution/README.md`
- [ ] Pinned `requirements.txt` or equivalent
- [ ] Completed `Documentation_template.md` or PDF
