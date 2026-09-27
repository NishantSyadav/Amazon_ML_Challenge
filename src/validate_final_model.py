"""Training-only final-retriever diagnostic; never imported by test inference."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.disk_candidates import CandidateIndex,source_rows,progress,fingerprint
from src.features import feature_matrix
from src.inference import load_model,score
from src.evaluate import entity_f05
from src.provenance import source_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir',type=Path,required=True)
    p.add_argument('--index-dir',type=Path,required=True)
    p.add_argument('--split-ids',type=Path,default=Path('artifacts/splits/val_source1_ids.tsv'))
    p.add_argument('--model',type=Path,default=Path('artifacts/experiments/baseline_v31/baseline_v1_catboost.cbm'))
    p.add_argument('--output-dir',type=Path,default=Path('artifacts/final_validation'))
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    # Reconstruct the exact entity split used by run_real_experiment.py.
    original=pd.read_csv(a.split_ids,sep='\t',dtype=str).sample(n=5000,random_state=42)['entity_id'].tolist()
    train_ids,val_ids=train_test_split(original,test_size=.20,random_state=123)
    assert not set(train_ids)&set(val_ids)
    wanted=set(val_ids)
    refs=[r for r in source_rows(a.train_dir/'train_source1.tsv') if r[0] in wanted]
    if {r[0] for r in refs}!=wanted or len(refs)!=1000: raise ValueError('Held-out reference coverage differs')
    truth={}
    with (a.train_dir/'train_ground_truth.tsv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            if row['source1_entity_id'] in wanted:
                truth[row['source1_entity_id']]=set(row['matched_entity_ids'].split(',')) if row['matched_entity_ids'] else set()
    if set(truth)!=wanted: raise ValueError('Ground truth coverage differs')
    code_identity=source_manifest()
    model=load_model(a.model); candidates={}; scored={}
    with CandidateIndex(a.index_dir) as idx, (a.output_dir/'scored_pairs.tsv').open('w',encoding='utf-8',newline='') as f:
        targets=[fingerprint(a.train_dir/f'train_source{i}.tsv') for i in (2,3)]
        if idx.meta['sources']!=targets: raise ValueError('Expected training-only target index')
        writer=csv.writer(f,delimiter='\t',lineterminator='\n'); writer.writerow(['source1_entity_id','candidate_entity_id','match_probability'])
        for offset in range(0,len(refs),100):
            groups=[]; features=[]
            for eid,n,ad,c in refs[offset:offset+100]:
                rows=idx.retrieve(n,ad,c); ids=[r[1] for r in rows]
                candidates[eid]=set(ids); groups.append((eid,ids))
                features.extend((n,ad,c,r[2],r[3],r[4]) for r in rows)
            probabilities=score(model,feature_matrix(features)); j=0
            for eid,ids in groups:
                values=probabilities[j:j+len(ids)]; j+=len(ids)
                scored[eid]=list(zip(ids,map(float,values)))
                writer.writerows((eid,cid,p) for cid,p in scored[eid])
            progress(f'Held-out validation: {offset+len(groups)}/1000 S1')
        overflows=idx.exact_overflows; index_version=idx.meta['version']
    curve=[]
    for threshold in np.round(np.arange(.30,.901,.01),3):
        macro=float(np.mean([entity_f05(truth[eid],{cid for cid,p in scored[eid] if p>=threshold}) for eid in val_ids]))
        curve.append({'threshold':float(threshold),'macro_f0.5':macro})
    best=max(curve,key=lambda x:(x['macro_f0.5'],x['threshold']))
    total_truth=sum(map(len,truth.values())); found=sum(len(truth[eid]&candidates[eid]) for eid in val_ids)
    positives=[eid for eid in val_ids if truth[eid]]
    report={'scope':'1000 original held-out S1; training-only targets; threshold tuning diagnostics, not leaderboard',
      'source_code':code_identity,'index_version':index_version,
      'validation_entities':1000,'training_entities':len(train_ids),'split_overlap':0,'candidate_pairs':sum(map(len,candidates.values())),
      'candidate_link_recall':found/max(total_truth,1),'any_match_entity_recall':sum(bool(truth[e]&candidates[e]) for e in positives)/len(positives),
      'complete_entity_recall':sum(truth[e]<=candidates[e] for e in positives)/len(positives),
      'candidate_oracle_macro_f0.5':float(np.mean([entity_f05(truth[e],truth[e]&candidates[e]) for e in val_ids])),
      'macro_f0.5_at_0.610':next(x['macro_f0.5'] for x in curve if x['threshold']==.61),
      'best':best,'threshold_curve':curve,'exact_key_overflows':overflows,'model':fingerprint(a.model)}
    if source_manifest()!=code_identity: raise ValueError('Production code changed during held-out validation')
    (a.output_dir/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='threshold_curve'},indent=2),flush=True)


if __name__=='__main__': main()
