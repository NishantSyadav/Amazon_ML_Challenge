# candidate generation / blocking
# takes S1 + S2 + S3, spits out candidate pairs for the matching model to score
#
# output columns (agreed with Om so it plugs straight into features.py):
#   source1_entity_id, candidate_entity_id, candidate_source, retrieval_score, retrieval_rank
#
# how it finds candidates:
#   1. exact match on normalized business name -> basically free, catches clean dupes
#   2. TF-IDF (char n-grams) on name
#   3. TF-IDF (char n-grams) on address
#   4. TF-IDF (char n-grams) on name+address glued together
#   all of the above run separately per country, then we union everything and
#   keep the best score whenever the same pair shows up more than once.
#
# country is NOT hardcoded to US/India anywhere here - we just group by
# whatever country values are actually in the data, so France at test time
# works without touching this file.
#
# install sparse-dot-topn if you can (pip install sparse-dot-topn), it's a
# fast C implementation of "top-N cosine similarity per row" which is exactly
# what we need at this scale (millions of rows). if it's not installed we
# still work, just slower (chunked fallback below).

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .normalize import normalize_name, normalize_address

try:
    from sparse_dot_topn import awesome_cossim_topn
    HAVE_TOPN = True
except ImportError:
    HAVE_TOPN = False

FINAL_COLS = ["source1_entity_id", "candidate_entity_id", "candidate_source",
              "retrieval_score", "retrieval_rank"]


def _empty(cols=FINAL_COLS):
    return pd.DataFrame(columns=cols)


def prep(df):
    df = df.copy()
    df["norm_name"] = df["business_name"].map(normalize_name)
    df["norm_address"] = df["business_address"].map(normalize_address)
    df["norm_combined"] = (df["norm_name"] + " " + df["norm_address"]).str.strip()
    return df


def exact_name_candidates(s1, other):
    """same normalized name -> call it a candidate, score 1.0. cheap and surprisingly effective."""
    merged = s1[["entity_id", "norm_name"]].merge(
        other[["entity_id", "norm_name"]], on="norm_name", suffixes=("_s1", "_cand")
    )
    merged = merged[merged["norm_name"] != ""]
    if merged.empty:
        return _empty(["source1_entity_id", "candidate_entity_id", "retrieval_score", "retrieval_method"])
    return pd.DataFrame({
        "source1_entity_id": merged["entity_id_s1"],
        "candidate_entity_id": merged["entity_id_cand"],
        "retrieval_score": 1.0,
        "retrieval_method": "exact_name",
    })


def _topn_sparse(query_vecs, index_vecs, top_n, lower_bound):
    """rows/cols/scores of the top_n cosine matches per query row (vectors already L2-normalized)."""
    if HAVE_TOPN:
        sims = awesome_cossim_topn(query_vecs, index_vecs.T.tocsr(), top_n, lower_bound).tocoo()
        return sims.row, sims.col, sims.data

    # dependency-free fallback, chunked so we don't blow up memory on big inputs
    rows, cols, scores = [], [], []
    chunk = 2000
    for start in range(0, query_vecs.shape[0], chunk):
        end = min(start + chunk, query_vecs.shape[0])
        block = (query_vecs[start:end] @ index_vecs.T).tocsr()
        for i in range(block.shape[0]):
            r = block.getrow(i)
            if r.nnz == 0:
                continue
            data, idx = r.data, r.indices
            keep = data >= lower_bound
            data, idx = data[keep], idx[keep]
            if not len(data):
                continue
            top = np.argsort(-data)[:top_n]
            rows.extend([start + i] * len(top))
            cols.extend(idx[top].tolist())
            scores.extend(data[top].tolist())
    return np.array(rows), np.array(cols), np.array(scores)


def tfidf_candidates(s1, other, text_col, method_name, top_n=20, lower_bound=0.25,
                      ngram_range=(2, 4), min_df=2):
    corpus_s1 = s1[text_col].fillna("")
    corpus_other = other[text_col].fillna("")
    if corpus_s1.empty or corpus_other.empty:
        return _empty(["source1_entity_id", "candidate_entity_id", "retrieval_score", "retrieval_method"])

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, min_df=min_df)
    vec.fit(pd.concat([corpus_s1, corpus_other], ignore_index=True))
    v1 = vec.transform(corpus_s1)
    v2 = vec.transform(corpus_other)

    rows, cols, scores = _topn_sparse(v1, v2, top_n, lower_bound)
    if not len(rows):
        return _empty(["source1_entity_id", "candidate_entity_id", "retrieval_score", "retrieval_method"])

    return pd.DataFrame({
        "source1_entity_id": s1["entity_id"].values[rows],
        "candidate_entity_id": other["entity_id"].values[cols],
        "retrieval_score": scores,
        "retrieval_method": method_name,
    })


def _candidates_for_country(s1_c, s2_c, s3_c, top_k, lower_bound):
    if s1_c.empty:
        return _empty(["source1_entity_id", "candidate_entity_id", "retrieval_score", "retrieval_method"])

    frames = []
    for other, tag in [(s2_c, "S2"), (s3_c, "S3")]:
        if other.empty:
            continue
        frames.append(exact_name_candidates(s1_c, other))
        frames.append(tfidf_candidates(s1_c, other, "norm_name", f"name_{tag}", top_n=top_k))
        frames.append(tfidf_candidates(s1_c, other, "norm_address", f"address_{tag}",
                                        top_n=top_k, lower_bound=lower_bound))
        frames.append(tfidf_candidates(s1_c, other, "norm_combined", f"combined_{tag}", top_n=top_k))

    frames = [f for f in frames if not f.empty]
    if not frames:
        return _empty(["source1_entity_id", "candidate_entity_id", "retrieval_score", "retrieval_method"])

    combined = pd.concat(frames, ignore_index=True)
    # same pair can get flagged by more than one method -> just keep the best score
    best = (
        combined.sort_values("retrieval_score", ascending=False)
        .groupby(["source1_entity_id", "candidate_entity_id"], as_index=False)
        .agg(retrieval_score=("retrieval_score", "max"))
    )
    return best


def generate_candidates(df_s1, df_s2, df_s3, top_k=30, lower_bound=0.25):
    """main entry point. returns the agreed 5-column candidate table, ready for features.py."""
    s1, s2, s3 = prep(df_s1), prep(df_s2), prep(df_s3)

    pieces = []
    for country in pd.unique(s1["country"].dropna()):
        s1_c = s1[s1["country"] == country]
        s2_c = s2[s2["country"] == country]
        s3_c = s3[s3["country"] == country]
        pieces.append(_candidates_for_country(s1_c, s2_c, s3_c, top_k, lower_bound))

    pieces = [p for p in pieces if not p.empty]
    if not pieces:
        return _empty()

    out = pd.concat(pieces, ignore_index=True)
    out["candidate_source"] = out["candidate_entity_id"].str.split("-").str[0]
    out["retrieval_rank"] = (
        out.groupby("source1_entity_id")["retrieval_score"]
        .rank(method="first", ascending=False).astype(int)
    )
    return out[FINAL_COLS].sort_values(["source1_entity_id", "retrieval_rank"]).reset_index(drop=True)


def to_submission_format(candidates, all_s1_ids, top_n_final=None):
    """official candidate_pairs.tsv shape for the final zip - one row per S1,
    candidate ids comma-joined. NOT what Om wants for features.py, that's the
    long format `generate_candidates` already returns."""
    valid = candidates.dropna(subset=["candidate_entity_id"])
    if top_n_final:
        valid = valid.sort_values("retrieval_score", ascending=False).groupby("source1_entity_id").head(top_n_final)
    grouped = (
        valid.sort_values("retrieval_score", ascending=False)
        .groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda ids: ",".join(dict.fromkeys(ids)))
        .reset_index().rename(columns={"candidate_entity_id": "candidate_entity_ids"})
    )
    full = pd.DataFrame({"source1_entity_id": list(all_s1_ids)}).merge(grouped, on="source1_entity_id", how="left")
    full["candidate_entity_ids"] = full["candidate_entity_ids"].fillna("")
    return full
