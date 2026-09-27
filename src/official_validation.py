"""Run the unchanged official validator over exhaustive bounded partitions."""
import argparse
import contextlib
import csv
import itertools
import json
import sqlite3
import time
import tempfile
from pathlib import Path
from src import official_validate_submission as official
from src.disk_candidates import COLUMNS,fingerprint,source_rows,progress
from src.pipeline import MATCH_HEADER,CANDIDATE_HEADER


def validate_partitioned(test_dir,output,scratch,batch_size=20000,expected_rows=1732544):
    if batch_size<1: raise ValueError('batch_size must be positive')
    test_dir=Path(test_dir).resolve(); output=Path(output).resolve(); scratch=Path(scratch).resolve()
    if scratch in (test_dir,output):
        raise ValueError('scratch must differ from source and output directories')
    scratch.mkdir(parents=True,exist_ok=True)
    scratch=Path(tempfile.mkdtemp(prefix='official-',dir=scratch))
    report_path=output/'official_validation.json'
    report_path.unlink(missing_ok=True)
    started=time.perf_counter(); rows=partitions=0; warning_set=set()
    # Preserve a complete transcript; no modification or monkey-patching of official code.
    seen=sqlite3.connect(scratch/'seen.sqlite')
    seen.execute('DROP TABLE IF EXISTS seen')
    seen.execute('CREATE TABLE seen(eid TEXT PRIMARY KEY) WITHOUT ROWID')
    try:
        with (output/'matching_results.tsv').open(encoding='utf-8',newline='') as mf, (output/'candidate_pairs.tsv').open(encoding='utf-8',newline='') as cf, (output/'official_validator.log').open('w',encoding='utf-8') as log:
            mr=csv.reader(mf,delimiter='\t'); cr=csv.reader(cf,delimiter='\t')
            if next(mr,None)!=MATCH_HEADER or next(cr,None)!=CANDIDATE_HEADER:
                raise ValueError('Incorrect official headers')
            combined=itertools.zip_longest(source_rows(test_dir/'test_source1.tsv'),mr,cr)
            while True:
                batch=list(itertools.islice(combined,batch_size))
                if not batch: break
                with (scratch/'test_source1.tsv').open('w',encoding='utf-8',newline='') as sf, (scratch/'matching_results.tsv').open('w',encoding='utf-8',newline='') as om, (scratch/'candidate_pairs.tsv').open('w',encoding='utf-8',newline='') as oc:
                    sw=csv.writer(sf,delimiter='\t',lineterminator='\n'); sw.writerow(COLUMNS)
                    mw=csv.writer(om,delimiter='\t',lineterminator='\n'); mw.writerow(MATCH_HEADER)
                    cw=csv.writer(oc,delimiter='\t',lineterminator='\n'); cw.writerow(CANDIDATE_HEADER)
                    for s,m,c in batch:
                        if s is None or m is None or c is None or len(m)!=2 or len(c)!=2 or m[0]!=s[0] or c[0]!=s[0]:
                            raise ValueError('Partition coverage/order mismatch')
                        try: seen.execute('INSERT INTO seen VALUES(?)',(s[0],))
                        except sqlite3.IntegrityError as exc: raise ValueError('Duplicate Source1 across partitions') from exc
                        sw.writerow(s); mw.writerow(m); cw.writerow(c)
                with contextlib.redirect_stdout(log):
                    print(f'Partition {partitions+1}; original rows {rows+1}-{rows+len(batch)}')
                    errors,warnings=official.validate(str(scratch/'matching_results.tsv'),str(scratch/'candidate_pairs.tsv'),str(scratch),check_ids=False)
                    for warning in warnings:
                        print('WARNING:',warning)
                        if not warning.startswith('ID-existence check is OFF'):
                            errors.append(warning)  # Official subset warning becomes a hard gate.
                    if errors:
                        print('FAIL:',errors)
                        raise ValueError(f'Official validator failed partition {partitions+1}: {errors[:5]}')
                    print('PASS — no blocking issues found.')
                log.flush(); warning_set.update(warnings)
                rows+=len(batch); partitions+=1; seen.commit()
                if partitions%5==0: progress(f'Official validator: {rows:,} rows passed in {partitions} partitions')
            if rows!=expected_rows: raise ValueError(f'Expected {expected_rows} rows, found {rows}')
            print(f'PASS: all {rows} rows, {partitions} disjoint partitions; global uniqueness verified.',file=log)
        report={'passed':True,'source1_rows':rows,'partitions':partitions,'partition_size':batch_size,
                'mode':'unchanged official.validate on exhaustive disjoint partitions; global S1 uniqueness and order checked',
                'warnings':sorted(warning_set),'target_id_check':'separate strict_qa.json verifies every candidate ID against the indexed original target sources',
                'validator':fingerprint(official.__file__),'matching_file':fingerprint(output/'matching_results.tsv'),
                'candidate_file':fingerprint(output/'candidate_pairs.tsv'),'seconds':time.perf_counter()-started}
        report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        progress(f'Official validation PASS: {rows:,} rows in {partitions} partitions')
        return report
    finally: seen.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('output'))
    p.add_argument('--scratch',type=Path,default=Path('work/official_validation'))
    p.add_argument('--batch-size',type=int,default=20000)
    p.add_argument('--expected-rows',type=int,default=1732544)
    a=p.parse_args()
    validate_partitioned(a.test_dir,a.output_dir,a.scratch,a.batch_size,a.expected_rows)


if __name__=='__main__': main()
