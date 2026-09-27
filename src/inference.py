"""Strict CatBoost schema and positive-class probability handling."""
import numpy as np
from catboost import CatBoostClassifier, Pool
from src.features import FEATURE_COLUMNS


def load_model(path, threads=4):
    model=CatBoostClassifier(thread_count=threads)
    model.load_model(str(path))
    if list(model.feature_names_) != FEATURE_COLUMNS:
        raise ValueError(f'Model feature order mismatch: {model.feature_names_!r}')
    if list(model.classes_) != [0,1]:
        raise ValueError(f'Expected binary classes [0,1], got {model.classes_!r}')
    return model


def score(model, matrix, threads=4):
    if list(model.feature_names_) != FEATURE_COLUMNS or matrix.shape[1] != 16:
        raise ValueError('Model feature order or matrix width mismatch')
    if not np.isfinite(matrix).all():
        raise ValueError('Nonfinite feature values')
    if not len(matrix):
        return np.empty(0)
    return model.predict_proba(Pool(matrix, feature_names=FEATURE_COLUMNS), thread_count=threads)[:,1]


def select_matches(ids, probabilities, threshold):
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be in [0,1]')
    if len(ids) != len(probabilities):
        raise ValueError('Candidate/probability length mismatch')
    return sorted({eid for eid,p in zip(ids,probabilities) if p >= threshold})


def score_batch(batch,idx,model,threshold,threads):
    from src.baseline_features import normalize_text
    from src.features import feature_matrix
    grouped=[]; rows=[]; overflows=idx.exact_overflows
    for eid,name,address,country in batch:
        if not eid.startswith('S1-'): raise ValueError(f'Invalid Source1 ID {eid}')
        name=normalize_text(name); address=normalize_text(address)
        candidates=idx.retrieve(name,address,country,normalized=True)
        ids=[r[1] for r in candidates]
        if len(ids)!=len(set(ids)): raise ValueError('Duplicate retrieval IDs')
        grouped.append((eid,ids,country))
        rows.extend((name,address,country,r[2],r[3],r[4]) for r in candidates)
    probabilities=score(model,feature_matrix(rows,normalized=True),threads)
    output=[]; offset=0
    for eid,ids,country in grouped:
        matches=select_matches(ids,probabilities[offset:offset+len(ids)],threshold)
        if not set(matches)<=set(ids): raise ValueError('Candidate subset invariant failed')
        output.append((eid,ids,matches,country)); offset+=len(ids)
    return output,idx.exact_overflows-overflows


def _worker_init(index,model_path,threshold,threads):
    import atexit
    from src.disk_candidates import CandidateIndex
    global _worker_index,_worker_model,_worker_threshold,_worker_threads
    _worker_index=CandidateIndex(index)
    _worker_model=load_model(model_path,threads)
    _worker_threshold=threshold; _worker_threads=threads
    atexit.register(_worker_index.__exit__,None,None,None)


def _worker_batch(batch):
    return score_batch(batch,_worker_index,_worker_model,_worker_threshold,_worker_threads)


def scored_batches(batches,index,model_path,threshold,threads,workers):
    """Preserve input order; at most two pending batches per process."""
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor
    from src.disk_candidates import CandidateIndex
    if workers==1:
        with CandidateIndex(index) as idx:
            model=load_model(model_path,threads)
            for batch in batches:
                yield score_batch(batch,idx,model,threshold,threads)
        return
    with ProcessPoolExecutor(max_workers=workers,initializer=_worker_init,
                             initargs=(index,model_path,threshold,threads)) as executor:
        pending=deque()
        for _ in range(workers*2):
            batch=next(batches,None)
            if batch is None: break
            pending.append(executor.submit(_worker_batch,batch))
        while pending:
            yield pending.popleft().result()
            batch=next(batches,None)
            if batch is not None:
                pending.append(executor.submit(_worker_batch,batch))


if __name__ == '__main__':
    from src.pipeline import main
    main()
