import csv
import numpy as np
import pandas as pd
import pytest
from src import features, inference


def test_model_features_match_training_and_handle_accents():
    from src.baseline_features import add_features
    from src.run_real_experiment import FEATURE_COLUMNS
    rows = [('Café Étoile', '10 rue République', 'France', 'Cafe Etoile', '10 rue Republique', 'France'),
            ('', '', 'NewCountry', '', '', 'NewCountry'),
            ('Foo Ltd', '12 Road', 'US', 'Bar', '120 Road', 'India')]
    assert hasattr(features, 'feature_matrix'), 'production feature generator missing'
    got = features.feature_matrix(rows)
    frame = pd.DataFrame(rows, columns=['s1_business_name', 's1_business_address', 's1_country',
        'candidate_business_name', 'candidate_business_address', 'candidate_country'])
    expected = add_features(frame)[FEATURE_COLUMNS].to_numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)
    assert got[0,4] == 1 and got[0,10] == 1
    assert got[1,4] == 0 and got[1,12] == 1


def test_threshold_boundary_and_multiple_matches():
    assert hasattr(inference, 'select_matches'), 'thresholding missing'
    assert inference.select_matches(['S2-1','S3-2','S2-3'], [0.61,0.9,0.6099], 0.61) == ['S2-1','S3-2']
    assert inference.select_matches([], [], 0.61) == []


def test_wrong_model_schema_rejected(tmp_path):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=2, verbose=False, allow_writing_files=False)
    m.fit(pd.DataFrame({'wrong': [0,1,2,3]}), [0,0,1,1])
    p = tmp_path/'wrong.cbm'
    m.save_model(str(p))
    assert hasattr(inference, 'load_model'), 'model schema gate missing'
    with pytest.raises(ValueError, match='feature'):
        inference.load_model(p)


def write_source(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w=csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['entity_id','business_name','business_address','country'])
        w.writerows(rows)


def test_disk_index_country_exact_fuzzy_and_overflow(tmp_path):
    from src import pipeline
    assert hasattr(pipeline, 'build_index'), 'disk retrieval missing'
    s2=tmp_path/'s2.tsv'; s3=tmp_path/'s3.tsv'
    write_source(s2, [('S2-1','Café Étoile','10 République','France'),
        ('S2-2','Alpha Unique','34 Road','NewCountry'),
        ('S2-3','Café Étoile','10 République','US')])
    write_source(s3, [('S3-1','Cafe Etoile','10 Republique','France')])
    pipeline.build_index([s2,s3], tmp_path/'index', bits=12, posting_cap=4)
    with pipeline.CandidateIndex(tmp_path/'index') as idx:
        found=idx.retrieve('Cafe Etoile','10 Republique','France')
        assert {r[1] for r in found} == {'S2-1','S3-1'}
        found=idx.retrieve('Alpha Uniqeu','34 Rod','NewCountry')
        assert 'S2-2' in {r[1] for r in found}
        assert idx.retrieve('','','NeverSeen') == []


def test_stream_qa_detects_subset_and_preserves_zero_rows(tmp_path):
    from src import pipeline
    assert hasattr(pipeline, 'validate_outputs'), 'stream QA missing'
    s1=tmp_path/'s1.tsv'; s2=tmp_path/'s2.tsv'; s3=tmp_path/'s3.tsv'
    write_source(s1,[('S1-1','a','b','France'),('S1-2','','','Unknown')])
    write_source(s2,[('S2-1','a','b','France')]); write_source(s3,[])
    pipeline.build_index([s2,s3],tmp_path/'index',bits=10,posting_cap=4)
    out=tmp_path/'out'; out.mkdir()
    (out/'matching_results.tsv').write_text('source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-2\t\n')
    (out/'candidate_pairs.tsv').write_text('source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1\nS1-2\t\n')
    report=pipeline.validate_outputs(s1,out,tmp_path/'index',expected_rows=2)
    assert report['source1_rows']==2 and report['zero_matches']==1
    assert report['countries']=={'France':1,'Unknown':1}
    (out/'candidate_pairs.tsv').write_text('source1_entity_id\tcandidate_entity_ids\nS1-1\t\nS1-2\t\n')
    with pytest.raises(ValueError, match='subset'):
        pipeline.validate_outputs(s1,out,tmp_path/'index',expected_rows=2)

def test_inference_end_to_end_real_model(tmp_path):
    from catboost import CatBoostClassifier
    from src import pipeline
    from src.features import FEATURE_COLUMNS
    from src.inference import score,load_model
    assert hasattr(pipeline,'infer')
    model=CatBoostClassifier(iterations=8,depth=2,verbose=False,allow_writing_files=False)
    model.fit(pd.DataFrame(np.array([[0]*16,[1]*16,[0]*16,[1]*16]),columns=FEATURE_COLUMNS),[0,1,0,1])
    mp=tmp_path/'model.cbm'; model.save_model(str(mp))
    s1=tmp_path/'s1.tsv'; s2=tmp_path/'s2.tsv'; s3=tmp_path/'s3.tsv'
    write_source(s1,[('S1-1','Café Étoile','10 République','France'),('S1-2','','','Unknown')])
    write_source(s2,[('S2-1','Cafe Etoile','10 Republique','France')])
    write_source(s3,[('S3-1','Cafe Etoile','10 Republique','France')])
    pipeline.build_index([s2,s3],tmp_path/'index',bits=10,posting_cap=4)
    pipeline.infer(s1,tmp_path/'index',mp,tmp_path/'out',threshold=0.5,batch_size=1)
    report=pipeline.validate_outputs(s1,tmp_path/'out',tmp_path/'index',expected_rows=2)
    assert report['candidate_pairs']==2 and report['predicted_links']==2
    assert report['zero_matches']==1
    assert (tmp_path/'out'/'matching_results.tsv').read_text() == 'source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\t\n'


def test_overfull_anchor_keeps_bounded_exact_candidates(tmp_path):
    from src import pipeline
    s2=tmp_path/'s2.tsv'; s3=tmp_path/'s3.tsv'
    write_source(s2,[(f'S2-{i}','Shared Name','Shared Address','France') for i in range(12)])
    write_source(s3,[])
    pipeline.build_index([s2,s3],tmp_path/'index',bits=10,posting_cap=2)
    with pipeline.CandidateIndex(tmp_path/'index',top_k=3,exact_limit=4) as idx:
        got=idx.retrieve('Shared Name','Shared Address','France')
        assert len(got)==3 and idx.exact_overflows>0
        assert idx.retrieve('Shared Name','Shared Address','US')==[]


def test_stale_index_rejected(tmp_path):
    from src import pipeline
    s2=tmp_path/'s2.tsv'; s3=tmp_path/'s3.tsv'
    write_source(s2,[('S2-1','Alpha','Road','France')]); write_source(s3,[])
    pipeline.build_index([s2,s3],tmp_path/'index',bits=10,posting_cap=2)
    write_source(s2,[('S2-2','Beta','Road','France')])
    with pytest.raises(ValueError,match='identity'):
        pipeline.build_index([s2,s3],tmp_path/'index',bits=10,posting_cap=2)

def test_exact_candidates_do_not_starve_source3(tmp_path):
    from src import pipeline
    s2=tmp_path/'s2.tsv'; s3=tmp_path/'s3.tsv'
    write_source(s2,[(f'S2-{i}','Common Name','1 Road','France') for i in range(8)])
    write_source(s3,[('S3-1','Common Name','1 Road','France')])
    pipeline.build_index([s2,s3],tmp_path/'index',bits=10,posting_cap=2)
    with pipeline.CandidateIndex(tmp_path/'index',top_k=2,exact_limit=4) as idx:
        got=idx.retrieve('Common Name','1 Road','France')
        assert 'S3-1' in {r[1] for r in got}


def test_derived_features_equal_training_float32():
    from src.baseline_features import add_features
    rows=[('abcdefghij','abcdefghijk','US','abcdefg','abcdefgh','US')]
    frame=pd.DataFrame(rows,columns=['s1_business_name','s1_business_address','s1_country','candidate_business_name','candidate_business_address','candidate_country'])
    expected=add_features(frame)[features.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(features.feature_matrix(rows),expected)

def test_failed_qa_invalidates_previous_pass(tmp_path):
    from src import pipeline
    s1=tmp_path/'s1.tsv'; s2=tmp_path/'s2.tsv'; s3=tmp_path/'s3.tsv'
    write_source(s1,[('S1-1','a','b','France')]); write_source(s2,[]); write_source(s3,[])
    pipeline.build_index([s2,s3],tmp_path/'idx',bits=10,posting_cap=2)
    out=tmp_path/'out'; out.mkdir()
    (out/'matching_results.tsv').write_text('source1_entity_id\tmatched_entity_ids\nS1-1\t\n')
    (out/'candidate_pairs.tsv').write_text('source1_entity_id\tcandidate_entity_ids\nS1-1\t\n')
    pipeline.validate_outputs(s1,out,tmp_path/'idx',expected_rows=1)
    (out/'matching_results.tsv').write_text('source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\n')
    with pytest.raises(ValueError): pipeline.validate_outputs(s1,out,tmp_path/'idx',expected_rows=1)
    assert not (out/'strict_qa.json').exists()


def test_random_feature_parity_without_double_rounding():
    import random
    from src.baseline_features import add_features
    rng=random.Random(42)
    rows=[]
    for _ in range(1000):
        strings=[''.join(rng.choices('abcdefg ',k=rng.randrange(5,40))) for j in range(4)]
        rows.append((strings[0],strings[1],'France',strings[2],strings[3],'France'))
    frame=pd.DataFrame(rows,columns=['s1_business_name','s1_business_address','s1_country','candidate_business_name','candidate_business_address','candidate_country'])
    expected=add_features(frame)[features.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(features.feature_matrix(rows),expected)

def test_index_rejects_wrong_array_shape(tmp_path):
    from src import pipeline
    s2=tmp_path/'s2.tsv'; s3=tmp_path/'s3.tsv'
    write_source(s2,[]); write_source(s3,[])
    pipeline.build_index([s2,s3],tmp_path/'idx',bits=10,posting_cap=2)
    np.save(tmp_path/'idx'/'counts.npy',np.zeros(2,dtype=np.uint32))
    with pytest.raises(ValueError,match='shape'):
        with pipeline.CandidateIndex(tmp_path/'idx'): pass

def test_long_name_anchors_survive_common_short_grams(tmp_path):
    from src import pipeline
    s2=tmp_path/'s2.tsv'; s3=tmp_path/'s3.tsv'
    # All four-grams of the target are common, but six-gram combinations are rare.
    target='abcdefghijklmno'
    distractors=[]
    for i in range(len(target)-3):
        for j in range(3):
            distractors.append((f'S2-{i}-{j}',target[i:i+4]+'xxxx'+str(j),'','France'))
    write_source(s2,distractors+[('S2-target',target,'','France')]); write_source(s3,[])
    pipeline.build_index([s2,s3],tmp_path/'idx',bits=16,posting_cap=2)
    with pipeline.CandidateIndex(tmp_path/'idx') as idx:
        found=idx.retrieve('abcdefghijklmnp','','France')
        assert 'S2-target' in {r[1] for r in found}

def test_parallel_inference_matches_serial_outputs(tmp_path):
    from catboost import CatBoostClassifier
    from src import pipeline
    import inspect
    assert 'workers' in inspect.signature(pipeline.infer).parameters, 'bounded process inference missing'
    model=CatBoostClassifier(iterations=2,verbose=False,allow_writing_files=False)
    model.fit(pd.DataFrame([[0]*16,[1]*16],columns=features.FEATURE_COLUMNS),[0,1])
    mp=tmp_path/'model.cbm'; model.save_model(str(mp))
    s1=tmp_path/'s1.tsv'; s2=tmp_path/'s2.tsv'; s3=tmp_path/'s3.tsv'
    write_source(s1,[('S1-1','Cafe Etoile','1 rue','France'),('S1-2','','','Unknown'),('S1-3','Other','2 rue','France')])
    write_source(s2,[('S2-1','Cafe Etoile','1 rue','France')]); write_source(s3,[('S3-1','Other','2 rue','France')])
    pipeline.build_index([s2,s3],tmp_path/'idx',bits=10,posting_cap=4)
    for workers in (1,2):
        pipeline.infer(s1,tmp_path/'idx',mp,tmp_path/f'out{workers}',batch_size=1,workers=workers)
    for name in ('matching_results.tsv','candidate_pairs.tsv'):
        assert (tmp_path/'out1'/name).read_bytes()==(tmp_path/'out2'/name).read_bytes()

def test_partitioned_official_validator_covers_all_rows(tmp_path):
    from src import pipeline
    import importlib.util
    assert importlib.util.find_spec('src.official_validation') is not None, 'bounded official validator missing'
    from src.official_validation import validate_partitioned
    data=tmp_path/'data'; data.mkdir(); out=tmp_path/'out'; out.mkdir()
    write_source(data/'test_source1.tsv',[('S1-1','a','b','France'),('S1-2','','','US'),('S1-3','c','d','India')])
    (out/'matching_results.tsv').write_text('source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-2\t\nS1-3\tS3-1\n')
    (out/'candidate_pairs.tsv').write_text('source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1\nS1-2\t\nS1-3\tS3-1,S2-2\n')
    report=validate_partitioned(data,out,tmp_path/'scratch',batch_size=2,expected_rows=3)
    assert report['passed'] and report['partitions']==2 and report['source1_rows']==3
    (out/'candidate_pairs.tsv').write_text('source1_entity_id\tcandidate_entity_ids\nS1-1\t\nS1-2\t\nS1-3\tS3-1,S2-2\n')
    with pytest.raises(ValueError):
        validate_partitioned(data,out,tmp_path/'scratch2',batch_size=2,expected_rows=3)

def test_official_scratch_cannot_overwrite_inputs_or_outputs(tmp_path):
    from src.official_validation import validate_partitioned
    data=tmp_path/'data'; data.mkdir(); out=tmp_path/'out'; out.mkdir()
    write_source(data/'test_source1.tsv',[('S1-1','a','b','France')])
    (out/'matching_results.tsv').write_text('source1_entity_id\tmatched_entity_ids\nS1-1\t\n')
    (out/'candidate_pairs.tsv').write_text('source1_entity_id\tcandidate_entity_ids\nS1-1\t\n')
    before={p:p.read_bytes() for p in [data/'test_source1.tsv',out/'matching_results.tsv',out/'candidate_pairs.tsv']}
    for scratch in (data,out):
        with pytest.raises(ValueError,match='scratch'):
            validate_partitioned(data,out,scratch,batch_size=1,expected_rows=1)
        assert all(p.read_bytes()==content for p,content in before.items())


def test_exact_query_work_is_bounded_per_source(tmp_path):
    from src import pipeline
    s2=tmp_path/'s2.tsv'; s3=tmp_path/'s3.tsv'
    write_source(s2,[(f'S2-{i}','Shared Name','Shared Address','France') for i in range(5000)])
    write_source(s3,[])
    pipeline.build_index([s2,s3],tmp_path/'idx',bits=10,posting_cap=2)
    with pipeline.CandidateIndex(tmp_path/'idx',top_k=3,exact_limit=4) as idx:
        calls=[0]
        def budget():
            calls[0]+=1
            return int(calls[0]>20)
        idx.con.set_progress_handler(budget,1000)
        try: found=idx.retrieve('Shared Name','Shared Address','France')
        except Exception as exc: pytest.fail(f'Bounded retrieval scanned too many SQLite operations: {exc}')
        assert len(found)==3

def test_packaging_rejects_changed_inference_code_and_missing_validation(tmp_path):
    import importlib.util
    assert importlib.util.find_spec('src.provenance') is not None, 'source provenance gate missing'
    from src.provenance import PRODUCTION_FILES,source_manifest,verify_provenance
    for name in PRODUCTION_FILES:
        p=tmp_path/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('original')
    run={'source_code':source_manifest(tmp_path),'index_version':2,'model':{'sha256':'abc'}}
    validation={'source_code':run['source_code'],'index_version':2,'model':{'sha256':'abc'},'validation_entities':1000,'training_entities':4000,'split_overlap':0}
    verify_provenance(run,validation,tmp_path)
    (tmp_path/'src/features.py').write_text('changed')
    with pytest.raises(ValueError,match='source'): verify_provenance(run,validation,tmp_path)
    (tmp_path/'src/features.py').write_text('original')
    with pytest.raises(ValueError,match='validation'): verify_provenance(run,{},tmp_path)
