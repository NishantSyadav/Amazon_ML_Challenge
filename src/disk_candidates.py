"""Disk-backed bounded retrieval. Does not load ground truth or training files."""
import csv
import hashlib
import heapq
import json
import sqlite3
import time
import zlib
from collections import Counter
from pathlib import Path
import numpy as np
import psutil
from rapidfuzz import fuzz
from src.baseline_features import normalize_text

COLUMNS=['entity_id','business_name','business_address','country']
VERSION=2


def progress(message):
    rss=psutil.Process().memory_info().rss/2**30
    print(f'{time.strftime("%H:%M:%S")} | {message} | RSS {rss:.2f} GiB',flush=True)


def source_rows(path):
    with open(path,encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f,delimiter='\t')
        if reader.fieldnames != COLUMNS:
            raise ValueError(f'{path}: expected {COLUMNS}, got {reader.fieldnames}')
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'{path}: malformed TSV row {reader.line_num}')
            eid=row['entity_id']
            if not eid or eid.strip()!=eid or any(x in eid for x in ',\t\n\r'):
                raise ValueError(f'{path}: invalid ID {eid!r}')
            yield tuple(row[k] for k in COLUMNS)


def fingerprint(path):
    p=Path(path)
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':h.hexdigest()}


def buckets(name,address,country,mask):
    seed=zlib.crc32(country.encode('utf-8'))
    result=[]
    for label,text,lengths in ((b'n',name,(4,6)),(b'a',address,(5,8))):
        compact=text.replace(' ','')
        prefix=zlib.crc32(label,seed)
        grams={compact[i:i+n] for n in lengths for i in range(max(0,len(compact)-n+1))}
        result.append({zlib.crc32(g.encode('ascii'),prefix)&mask for g in grams})
    return result


def connect(path):
    con=sqlite3.connect(str(path))
    con.execute('PRAGMA cache_size=-65536')
    con.execute('PRAGMA temp_store=FILE')
    return con


def build_index(paths,directory,bits=24,posting_cap=24):
    directory=Path(directory)
    identities=[fingerprint(p) for p in paths]
    meta_path=directory/'metadata.json'
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta['sources'] != identities or meta['version'] != VERSION or meta['bits']!=bits or meta['posting_cap']!=posting_cap:
            raise ValueError('Index identity/config mismatch; use a new index directory')
        progress('Reusing verified target index')
        return meta
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'records.sqlite').exists():
        raise ValueError('Incomplete index exists; choose a new directory (no silent reuse)')
    if not 8<=bits<=26 or not 1<=posting_cap<=128:
        raise ValueError('Invalid index size')
    size=1<<bits
    counts=np.lib.format.open_memmap(directory/'counts.npy',mode='w+',dtype=np.uint32,shape=(size,))
    postings=np.lib.format.open_memmap(directory/'postings.npy',mode='w+',dtype=np.uint32,shape=(size,posting_cap))
    counts[:]=0
    count_view=memoryview(counts)
    posting_view=memoryview(postings).cast('B').cast('I')
    con=connect(directory/'records.sqlite')
    con.execute('PRAGMA journal_mode=OFF')
    con.execute('CREATE TABLE records(rid INTEGER PRIMARY KEY,eid TEXT NOT NULL, name TEXT NOT NULL,address TEXT NOT NULL,country TEXT NOT NULL,source INTEGER NOT NULL)')
    started=time.monotonic(); rid=0; buffer=[]; country_counts=Counter()
    try:
        for source,path in enumerate(paths,start=2):
            for eid,name,address,country in source_rows(path):
                if not eid.startswith(f'S{source}-'):
                    raise ValueError(f'Wrong source prefix: {eid}')
                rid+=1
                name=normalize_text(name); address=normalize_text(address)
                buffer.append((rid,eid,name,address,country,source))
                country_counts[country]+=1
                ng,ag=buckets(name,address,country,size-1)
                for bucket in ng|ag:
                    count=count_view[bucket]
                    if count<posting_cap:
                        posting_view[bucket*posting_cap+count]=rid
                    if count<=posting_cap:
                        count_view[bucket]=count+1
                if len(buffer)>=20000:
                    con.executemany('INSERT INTO records VALUES(?,?,?,?,?,?)',buffer)
                    con.commit(); buffer.clear()
                    if rid%200000==0:
                        progress(f'Index: {rid:,} targets, {rid/(time.monotonic()-started):,.0f} rows/s')
        if buffer:
            con.executemany('INSERT INTO records VALUES(?,?,?,?,?,?)',buffer)
            con.commit()
        progress('Building exact-key and ID indexes')
        con.execute('CREATE UNIQUE INDEX by_id ON records(eid)')
        con.execute('CREATE INDEX by_name ON records(country,source,name,address)')
        con.execute('CREATE INDEX by_address ON records(country,source,address,name)')
        con.commit()
        counts.flush(); postings.flush()
        meta={'version':VERSION,'bits':bits,'posting_cap':posting_cap,'sources':identities,
              'target_rows':rid,'countries':dict(country_counts),'seconds':time.monotonic()-started}
        meta_path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
        progress(f'Index complete: {rid:,} targets in {meta["seconds"]:.1f}s')
        return meta
    finally:
        con.close()
        del count_view,posting_view,counts,postings


def rerank(name,address,row):
    def sim(a,b):
        return max(fuzz.ratio(a,b),fuzz.partial_ratio(a,b),fuzz.token_set_ratio(a,b))/100 if a and b else 0
    ns=sim(name,row[2]); ads=sim(address,row[3])
    both=bool(name and address and name==row[2] and address==row[3])
    return (int(both),0.6*ns+0.4*ads,ns,ads,row[1])


class CandidateIndex:
    def __init__(self,directory,top_k=50,pre_k=80,exact_limit=512):
        self.directory=Path(directory)
        self.meta=json.loads((self.directory/'metadata.json').read_text())
        if self.meta['version']!=VERSION:
            raise ValueError('Unsupported index version')
        self.counts=np.load(self.directory/'counts.npy',mmap_mode='r')
        self.postings=np.load(self.directory/'postings.npy',mmap_mode='r')
        size=1<<self.meta['bits']
        if self.counts.shape!=(size,) or self.postings.shape!=(size,self.meta['posting_cap']) or self.counts.dtype!=np.uint32 or self.postings.dtype!=np.uint32:
            raise ValueError('Index array shape or dtype mismatch')
        db=self.directory/'records.sqlite'
        if not db.is_file(): raise ValueError('Index record store missing')
        self.con=connect(db)
        count=self.con.execute('SELECT count(*) FROM records').fetchone()[0]
        if count!=self.meta['target_rows']:
            self.con.close()
            raise ValueError('Index record count mismatch')
        for key,columns in [('by_name',['country','source','name','address']),('by_address',['country','source','address','name'])]:
            actual=[r[2] for r in self.con.execute(f'PRAGMA index_info({key})')]
            if actual!=columns:
                self.con.close()
                raise ValueError('Exact index schema mismatch; rebuild source-aware indexes')
        self.top_k=top_k; self.pre_k=pre_k; self.exact_limit=exact_limit
        self.exact_overflows=0

    def __enter__(self): return self
    def __exit__(self,*args):
        self.con.close()
        del self.counts,self.postings

    def retrieve(self,name,address,country,normalized=False):
        if not normalized:
            name=normalize_text(name); address=normalize_text(address)
        ng,ag=buckets(name,address,country,(1<<self.meta['bits'])-1)
        hit=Counter()
        for grams,limit in ((ng,6),(ag,8)):
            selected=sorted((int(self.counts[g]),g) for g in grams if 0<self.counts[g]<=self.meta['posting_cap'])[:limit]
            for count,g in selected:
                for rid in self.postings[g,:count]:
                    hit[int(rid)]+=1/max(len(selected),1)
        # Single bounded IN query. Hash collisions cannot bypass exact country filter.
        rids=list(hit)
        found={}
        if rids:
            # Do not add country here: SQLite can choose a country-wide scan for large IN lists.
            q='SELECT * FROM records WHERE rid IN ('+','.join('?'*len(rids))+')'
            rows=[r for r in self.con.execute(q,rids) if r[4]==country]
            for source in (2,3):
                selected=heapq.nlargest(self.pre_k,(r for r in rows if r[5]==source),key=lambda r:(hit[r[0]],r[1]))
                found.update((r[0],r) for r in selected)
        queries=[]
        if name and address:
            queries.append(('country=? AND name=? AND address=?',(country,name,address)))
        if name: queries.append(('country=? AND name=?',(country,name)))
        if address: queries.append(('country=? AND address=?',(country,address)))
        for where,args in queries:
            for source in (2,3):
                rows=self.con.execute('SELECT * FROM records WHERE '+where+' AND source=? LIMIT ?',(*args,source,self.exact_limit+1)).fetchall()
                if len(rows)>self.exact_limit:
                    self.exact_overflows+=1
                    # Both-field exact keys are queried first; broad keys have bounded latency.
                    rows=rows[:self.exact_limit]
                found.update((row[0],row) for row in rows)
        ranked={2:[],3:[]}
        for row in found.values():
            ranked[row[5]].append((rerank(name,address,row),row))
        return [row for source in (2,3) for _,row in heapq.nlargest(self.top_k,ranked[source])]
