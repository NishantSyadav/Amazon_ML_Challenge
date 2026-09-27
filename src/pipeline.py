"""Single production entry point: index, infer, validate, or run end to end."""
import argparse
import csv
import itertools
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path
import psutil
from src.disk_candidates import CandidateIndex,build_index,source_rows,progress,fingerprint
from src.baseline_features import normalize_text
from src.features import feature_matrix,FEATURE_COLUMNS
from src.inference import load_model,score,select_matches
from src.provenance import source_manifest

MATCH_HEADER=['source1_entity_id','matched_entity_ids']
CANDIDATE_HEADER=['source1_entity_id','candidate_entity_ids']


def infer(source1,index,model_path,output,threshold=0.560,batch_size=500,limit=None,threads=1,workers=1):
    from src.inference import scored_batches
    if batch_size<1 or not 0<=threshold<=1 or (limit is not None and limit<1) or not 1<=workers<=8:
        raise ValueError('Invalid batch size, threshold, limit, or workers')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    for name in ('matching_results.tsv','candidate_pairs.tsv','run.json'):
        if (output/name).exists():
            raise ValueError(f'Output already exists: {output/name}; use a fresh directory')
    code_identity=source_manifest(); model_identity=fingerprint(model_path)
    load_model(model_path,threads)  # Fail schema checks before any worker or output.
    meta=json.loads((Path(index)/'metadata.json').read_text())
    started=time.perf_counter(); processed=pairs=links=empty=zero_candidates=overflows=0
    countries=Counter(); peak=0; min_available=psutil.virtual_memory().available
    readers=iter(source_rows(source1))
    if limit: readers=itertools.islice(readers,limit)
    def batches():
        while True:
            batch=list(itertools.islice(readers,batch_size))
            if not batch: return
            yield batch
    with (output/'matching_results.tsv.partial').open('w',encoding='utf-8',newline='') as mf, (output/'candidate_pairs.tsv.partial').open('w',encoding='utf-8',newline='') as cf:
        mw=csv.writer(mf,delimiter='\t',lineterminator='\n'); mw.writerow(MATCH_HEADER)
        cw=csv.writer(cf,delimiter='\t',lineterminator='\n'); cw.writerow(CANDIDATE_HEADER)
        for grouped,batch_overflows in scored_batches(batches(),index,model_path,threshold,threads,workers):
            for eid,ids,matches,country in grouped:
                mw.writerow((eid,','.join(matches))); cw.writerow((eid,','.join(sorted(ids))))
                pairs+=len(ids); links+=len(matches)
                empty+=not matches; zero_candidates+=not ids; countries[country]+=1
            processed+=len(grouped); overflows+=batch_overflows
            mf.flush(); cf.flush()
            process=psutil.Process()
            rss=process.memory_info().rss
            for child in process.children():
                try: rss+=child.memory_info().rss
                except psutil.NoSuchProcess: pass
            peak=max(peak,rss); min_available=min(min_available,psutil.virtual_memory().available)
            elapsed=max(time.perf_counter()-started,1e-9)
            if processed%5000==0 or (limit and processed>=limit):
                progress(f'Inference: {processed:,} S1 | {pairs:,} candidates | {links:,} links | {processed/elapsed:.1f} S1/s')
        report={'source1_rows':processed,'candidate_pairs':pairs,'average_candidates':pairs/max(processed,1),
                'predicted_links':links,'zero_matches':empty,'zero_candidates':zero_candidates,
                'countries':dict(countries),'seconds':time.perf_counter()-started,'peak_batch_rss_bytes':peak,
                'rss_note':'sum of process RSS; shared mapped pages may be counted more than once',
                'minimum_system_available_bytes':min_available,'workers':workers,'threads_per_worker':threads,
                'threshold':threshold,'model':model_identity,'source_code':code_identity,'source1':fingerprint(source1),
                'index_sources':meta['sources'],'index_version':meta['version'],'feature_columns':FEATURE_COLUMNS,
                'exact_key_overflows':overflows,'limit':limit,'batch_size':batch_size,
                'status':'inference_finished_validation_pending'}
    if source_manifest()!=code_identity or fingerprint(model_path)['sha256']!=model_identity['sha256']:
        raise ValueError('Inference source/model changed during the run')
    for name in ('matching_results.tsv','candidate_pairs.tsv'):
        (output/(name+'.partial')).replace(output/name)
    (output/'run.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    progress(f'Inference finished: {processed:,} entities, {report["seconds"]:.1f}s')
    return report


def parse_ids(raw):
    ids=raw.split(',') if raw else []
    if len(ids)!=len(set(ids)) or any(not x.startswith(('S2-','S3-')) or x.strip()!=x for x in ids):
        raise ValueError('Invalid or duplicate listed IDs')
    return set(ids)


def validate_outputs(source1,output,index,expected_rows=1732544,limit=None):
    """Disk-backed identity checks; strict lockstep coverage with the original S1."""
    output=Path(output); started=time.perf_counter()
    (output/'strict_qa.json').unlink(missing_ok=True)
    qa_path=output/'qa_seen.sqlite'
    if qa_path.exists(): qa_path.unlink()
    con=sqlite3.connect(qa_path)
    con.execute('PRAGMA journal_mode=OFF')
    con.execute('PRAGMA cache_size=-32768')
    con.execute('CREATE TABLE seen(eid TEXT PRIMARY KEY) WITHOUT ROWID')
    rows=pairs=links=empty=zero_candidates=0; countries=Counter(); pending=set()
    try:
        with CandidateIndex(index) as idx, (output/'matching_results.tsv').open(encoding='utf-8',newline='') as mf, (output/'candidate_pairs.tsv').open(encoding='utf-8',newline='') as cf:
            mr=csv.reader(mf,delimiter='\t'); cr=csv.reader(cf,delimiter='\t')
            if next(mr,None)!=MATCH_HEADER or next(cr,None)!=CANDIDATE_HEADER:
                raise ValueError('Output header mismatch')
            sr=source_rows(source1)
            if limit: sr=itertools.islice(sr,limit)
            def check_targets():
                ids=list(pending)
                for start in range(0,len(ids),500):
                    chunk=ids[start:start+500]
                    count=idx.con.execute('SELECT count(*) FROM records WHERE eid IN ('+','.join('?'*len(chunk))+')',chunk).fetchone()[0]
                    if count!=len(chunk): raise ValueError('Candidate ID absent from target sources')
                pending.clear()
            for s,m,c in itertools.zip_longest(sr,mr,cr):
                if s is None or m is None or c is None:
                    raise ValueError('Source1/output row count mismatch')
                if len(m)!=2 or len(c)!=2 or m[0]!=s[0] or c[0]!=s[0]:
                    raise ValueError('Source1 identity/order or TSV field mismatch')
                if not s[0].startswith('S1-'): raise ValueError('Invalid Source1 ID')
                try: con.execute('INSERT INTO seen VALUES(?)',(s[0],))
                except sqlite3.IntegrityError as exc: raise ValueError('Duplicate Source1 ID') from exc
                mids=parse_ids(m[1]); cids=parse_ids(c[1])
                if not mids<=cids: raise ValueError('Matching results violate candidate subset invariant')
                pending.update(cids)
                rows+=1; pairs+=len(cids); links+=len(mids); empty+=not mids; zero_candidates+=not cids
                countries[s[3]]+=1
                if len(pending)>=50000: check_targets()
                if rows%100000==0:
                    con.commit(); progress(f'Strict QA: {rows:,} S1 verified')
            check_targets()
        if rows!=expected_rows:
            raise ValueError(f'Expected {expected_rows:,} Source1 rows, found {rows:,}')
        result={'passed':True,'source1_rows':rows,'candidate_pairs':pairs,'average_candidates':pairs/max(rows,1),
                'predicted_links':links,'zero_matches':empty,'zero_candidates':zero_candidates,
                'countries':dict(countries),'seconds':time.perf_counter()-started,
                'matching_file':fingerprint(output/'matching_results.tsv'),
                'candidate_file':fingerprint(output/'candidate_pairs.tsv'),
                'checks':['exact headers','TSV readback','S1 complete unique identity','target ID existence',
                          'no duplicate listed IDs','matches subset of candidates','country coverage','empty rows']}
        (output/'strict_qa.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        progress(f'Strict QA passed: {rows:,} rows')
        return result
    finally:
        con.close()
        qa_path.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['index','infer','validate','run'])
    p.add_argument('--test-dir',type=Path,required=True)
    p.add_argument('--index-dir',type=Path,default=Path('work/test_index'))
    p.add_argument('--output-dir',type=Path,default=Path('output'))
    p.add_argument('--model',type=Path,default=Path('artifacts/experiments/baseline_v31/baseline_v1_catboost.cbm'))
    p.add_argument('--threshold',type=float,default=.560)
    p.add_argument('--batch-size',type=int,default=500)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit',type=int)
    p.add_argument('--expected-rows',type=int,default=1732544)
    a=p.parse_args()
    s1=a.test_dir/'test_source1.tsv'
    if a.stage in ('index','run'):
        build_index([a.test_dir/'test_source2.tsv',a.test_dir/'test_source3.tsv'],a.index_dir)
    if a.stage in ('infer','validate','run'):
        # Explicit identity check prevents using a train index for test inference.
        meta=json.loads((a.index_dir/'metadata.json').read_text())
        targets=[fingerprint(a.test_dir/f'test_source{i}.tsv') for i in (2,3)]
        if meta['sources']!=targets: raise ValueError('Target index does not match requested test files')
    if a.stage in ('infer','run'):
        infer(s1,a.index_dir,a.model,a.output_dir,a.threshold,a.batch_size,a.limit,a.threads,a.workers)
    if a.stage in ('validate','run'):
        validate_outputs(s1,a.output_dir,a.index_dir,a.limit or a.expected_rows,a.limit)


if __name__=='__main__': main()
