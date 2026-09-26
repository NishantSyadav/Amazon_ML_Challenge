# CLI for running candidate generation.
# start small (--sample), check recall, then scale up per the team plan:
# 10k -> 50k -> 100k -> 500k -> full.
#
# example:
#   python run_candidate_generation.py \
#       --data-dir dataset/train --prefix train --sample 10000 \
#       --out candidate_pairs.tsv \
#       --ground-truth dataset/train/train_ground_truth.tsv
#
# for the real test run just point --data-dir at dataset/test, drop --sample
# and --ground-truth (test has no ground truth), and use --top-k /
# --lower-bound whatever you landed on after validating on train.

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .blocking import generate_candidates, to_submission_format
from .candidate_recall import candidate_recall, candidate_recall_by_source, avg_candidates_per_s1


def load(path, sample):
    df = pd.read_csv(path, sep="\t", dtype=str)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--prefix", default="train")
    ap.add_argument("--sample", type=int, default=None, help="sample N source1 rows for a quick run")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--lower-bound", type=float, default=0.25)
    ap.add_argument("--out", required=True, help="candidate_pairs.tsv in the agreed 5-column format")
    ap.add_argument("--out-submission", default=None,
                     help="optional - also write the official grouped candidate_pairs.tsv shape for the final zip")
    ap.add_argument("--top-n-final", type=int, default=None, help="cap per-S1 candidates in --out-submission")
    ap.add_argument("--ground-truth", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    t0 = time.time()
    s1 = load(data_dir / f"{args.prefix}_source1.tsv", args.sample)
    # don't sample S2/S3, only S1 - otherwise recall numbers get skewed
    s2 = load(data_dir / f"{args.prefix}_source2.tsv", None)
    s3 = load(data_dir / f"{args.prefix}_source3.tsv", None)
    print(f"loaded S1={len(s1)} S2={len(s2)} S3={len(s3)} in {time.time()-t0:.1f}s")

    t1 = time.time()
    candidates = generate_candidates(s1, s2, s3, top_k=args.top_k, lower_bound=args.lower_bound)
    print(f"got {len(candidates)} candidate pairs in {time.time()-t1:.1f}s")

    zero_cand = set(s1["entity_id"]) - set(candidates["source1_entity_id"])
    print(f"S1 entities with zero candidates: {len(zero_cand)}")

    candidates.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out}")

    if args.out_submission:
        sub = to_submission_format(candidates, s1["entity_id"], top_n_final=args.top_n_final)
        sub.to_csv(args.out_submission, sep="\t", index=False)
        print(f"wrote {args.out_submission}")

    print("candidate density:", avg_candidates_per_s1(candidates, s1["entity_id"]))

    if args.ground_truth:
        gt = pd.read_csv(args.ground_truth, sep="\t", dtype=str)
        gt = gt[gt["source1_entity_id"].isin(s1["entity_id"])]
        print("blocking recall (overall):", candidate_recall(candidates, gt))
        print("blocking recall (by source):")
        print(candidate_recall_by_source(candidates, gt))


if __name__ == "__main__":
    main()
