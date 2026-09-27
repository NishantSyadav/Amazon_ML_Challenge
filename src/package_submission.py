"""Package only verified full-run outputs and tracked source, then inspect the ZIP."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from src.disk_candidates import fingerprint,progress
from src.provenance import verify_provenance


def archive_hash(archive,name):
    h=hashlib.sha256()
    with archive.open(name) as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def package(root,output,zip_path):
    root=Path(root).resolve(); output=Path(output).resolve(); zip_path=Path(zip_path).resolve()
    run=json.loads((output/'run.json').read_text())
    qa=json.loads((output/'strict_qa.json').read_text())
    official=json.loads((output/'official_validation.json').read_text())
    if not qa['passed'] or not official['passed'] or any(r['source1_rows']!=1732544 for r in (run,qa,official)) or run['limit'] is not None:
        raise ValueError('Full-run readiness gates have not passed')
    for key in ('candidate_pairs','predicted_links','zero_matches','countries'):
        if run[key]!=qa[key]: raise ValueError(f'Inference/QA disagreement: {key}')
    for name,key in [('matching_results.tsv','matching_file'),('candidate_pairs.tsv','candidate_file')]:
        actual=fingerprint(output/name)
        if actual['sha256']!=qa[key]['sha256'] or actual['sha256']!=official[key]['sha256']:
            raise ValueError(f'Output changed after validation: {name}')
    model=root/'artifacts/experiments/baseline_v31/baseline_v1_catboost.cbm'
    if fingerprint(model)['sha256']!=run['model']['sha256']: raise ValueError('Model changed since inference')
    def git(*args): return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
    if git('status','--porcelain'): raise ValueError('Commit source/docs before packaging: Git is not clean')
    commit=git('rev-parse','HEAD'); branch=git('branch','--show-current')
    validation_path=root/'artifacts/final_validation_v2/validation.json'
    validation=json.loads(validation_path.read_text()) if validation_path.exists() else {}
    verify_provenance(run,validation,root)
    readiness=f'''# Submission readiness — verified local artifacts

- Source commit used for submission: `{commit}`
- Branch: `{branch}`
- Model: `artifacts/experiments/baseline_v31/baseline_v1_catboost.cbm`
- Model SHA-256: `{run['model']['sha256']}`
- Threshold: **{run['threshold']:.3f}**
- Retrieval: bounded hashed short/long rare n-grams, exact lookups, fuzzy reranking,
  maximum 50 final candidates per source; per-source exact key scan cap 512.
- Source1/output rows in each file: **{qa['source1_rows']:,}**
- Final scored candidate pairs: **{qa['candidate_pairs']:,}**
- Average candidates per S1: **{qa['average_candidates']:.6f}**
- Predicted links: **{qa['predicted_links']:,}**
- Zero-match entities: **{qa['zero_matches']:,}**
- Zero-candidate entities: **{qa['zero_candidates']:,}**
- Country coverage: `{json.dumps(qa['countries'],sort_keys=True)}`
- Inference seconds: {run['seconds']:.2f}; workers: {run['workers']}.
- Sum-of-process peak RSS: {run['peak_batch_rss_bytes']/2**30:.3f} GiB;
  shared mapped pages may be counted multiple times.
- Minimum observed system available RAM: {run['minimum_system_available_bytes']/2**30:.3f} GiB.
- Exact-key overflow events: {run['exact_key_overflows']}.
- Strict QA: **PASS**, all S1 and target IDs checked, exact headers/readback,
  no duplicates, empty rows retained, all matches subset of scored candidates.
- Official validator: **PASS**, unchanged supplied validator executed over
  {official['partitions']} exhaustive disjoint partitions with global S1 uniqueness.
  Optional official in-memory ID check is off; strict disk-backed QA checks every ID.
- Supplied validator SHA-256: `{official['validator']['sha256']}`
- Matching results: `{output/'matching_results.tsv'}`
- Matching SHA-256: `{qa['matching_file']['sha256']}`
- Candidate output: `{output/'candidate_pairs.tsv'}`
- Candidate SHA-256: `{qa['candidate_file']['sha256']}`
- ZIP: `{zip_path}`

## Internal validation (not leaderboard results)

Historical sampled V3.1: link recall 83.8402%, any-match entity recall 95.5829%,
complete entity recall 67.9337%, oracle macro F0.5 0.919427, model macro F0.5
0.816031 at threshold 0.610. These metrics do not describe the final retriever.

Final retriever held-out results: `{json.dumps({k:v for k,v in validation.items() if k in ['candidate_link_recall','any_match_entity_recall','complete_entity_recall','candidate_oracle_macro_f0.5','macro_f0.5_at_0.610','best']},sort_keys=True)}`
The 1,000 S1 entities are disjoint from the saved model's 4,000 training entities;
threshold tuning reuses the original validation set and is not an unbiased test score.

## Limitations

Bounded postings, hash collisions, exact scan caps, and top-K can miss links.
France is preserved but absent from model training. No test truth or leaderboard
score is available. The ZIP and TSVs still require your upload to the official portal.
'''
    (root/'SUBMISSION_READINESS.md').write_text(readiness,encoding='utf-8')
    zip_path.parent.mkdir(parents=True,exist_ok=True)
    if zip_path.exists(): raise ValueError(f'ZIP already exists: {zip_path}')
    code_prefix='code/business_entity_resolution/'
    sources=git('ls-files','src').splitlines()
    if any('experimental_v2' in p for p in sources): raise ValueError('Failed experiment is tracked')
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for name in ('matching_results.tsv','candidate_pairs.tsv'):
            z.write(output/name,'output/'+name)
        for path in sources:
            if path.endswith('.py'): z.write(root/path,code_prefix+path)
        for name in ('README.md','requirements.txt','requirements-dev.txt'):
            z.write(root/name,code_prefix+name)
        for path in git('ls-files','tests').splitlines():
            if path.endswith('.py'): z.write(root/path,code_prefix+path)
        z.write(model,code_prefix+'artifacts/experiments/baseline_v31/baseline_v1_catboost.cbm')
        z.write(root/'Documentation_template.md','Documentation_template.md')
        z.write(root/'SUBMISSION_READINESS.md','SUBMISSION_READINESS.md')
        z.writestr('SUBMISSION_COMMIT.txt',commit+'\n')
        for name in ('run.json','strict_qa.json','official_validation.json','official_validator.log'):
            z.write(output/name,'verification/'+name)
        if validation_path.exists(): z.write(validation_path,'verification/validation.json')
    with zipfile.ZipFile(zip_path) as z:
        names=z.namelist()
        required={'output/matching_results.tsv','output/candidate_pairs.tsv','Documentation_template.md',code_prefix+'src/pipeline.py',code_prefix+'README.md',code_prefix+'requirements.txt',code_prefix+'artifacts/experiments/baseline_v31/baseline_v1_catboost.cbm'}
        if not required<=set(names) or len(names)!=len(set(names)):
            raise ValueError('ZIP hierarchy or duplicate entry failure')
        if any(n.startswith(('/','\\')) or '..' in Path(n).parts or 'experimental_v2' in n for n in names):
            raise ValueError('Unsafe/unwanted archive member')
        bad=z.testzip()
        if bad: raise ValueError(f'ZIP CRC failure: {bad}')
        for name,key in [('matching_results.tsv','matching_file'),('candidate_pairs.tsv','candidate_file')]:
            if archive_hash(z,'output/'+name)!=qa[key]['sha256']:
                raise ValueError(f'ZIP output mismatch: {name}')
        if archive_hash(z,code_prefix+'artifacts/experiments/baseline_v31/baseline_v1_catboost.cbm')!=run['model']['sha256']:
            raise ValueError('ZIP model mismatch')
        manifest={'passed':True,'commit':commit,'branch':branch,'zip':fingerprint(zip_path),'entries':names}
    zip_path.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    progress(f'ZIP inspection passed: {len(names)} entries')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--output-dir',type=Path,default=Path('output'))
    p.add_argument('--zip',type=Path,required=True)
    a=p.parse_args(); package(a.root,a.output_dir,a.zip)


if __name__=='__main__': main()
