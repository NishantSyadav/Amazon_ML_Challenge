# Candidate pair feature integration

Run from the repository root in PowerShell (one line):

```powershell
python -m src.build_features --candidates output/train_candidates.tsv --source1 dataset/train/train_source1.tsv --source2 dataset/train/train_source2.tsv --source3 dataset/train/train_source3.tsv --out output/train_features.tsv --max-pairs 1000
```

Remove `--max-pairs 1000` for the full candidate file. `--candidate-chunk-size` defaults to 10,000 rows and `--source-chunk-size` to 100,000 rows. The command reads existing candidates; it does not run candidate generation.

Input candidates must have `source1_entity_id`, `candidate_entity_id`, `candidate_source`, `retrieval_score`, and `retrieval_rank`. `candidate_source` must be `S2` or `S3`. Each source TSV must have `entity_id`, `business_name`, `business_address`, and `country`.

Output is a tab-separated file with the five candidate columns followed by the numeric fields returned by `build_pair_features()` (without repeating `retrieval_score` and `retrieval_rank`). Names and addresses are lookup inputs and are not written. The existing feature names include `name1_missing`, `name2_missing`, `address1_missing`, and `address2_missing`.

The builder checks all candidate chunks for malformed metadata and duplicate pair IDs, scans source files in chunks while retaining only referenced entities, and fails if any requested record is missing. It writes feature chunks to a temporary file and replaces the requested output only after validation succeeds. Its summary reports row counts, S2/S3 counts, missing lookups, duplicate pairs, non-finite values, and feature count.
