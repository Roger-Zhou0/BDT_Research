"""Shared transactional I/O, pinned modules, and execution identity. No GPU work on import."""
from __future__ import annotations
import contextlib, datetime, fcntl, hashlib, importlib.util, json, os, sys, types, uuid
from pathlib import Path
HERE = Path(__file__).resolve().parent
GCT2 = json.loads((HERE/'design/protocols/GCT2.json').read_text())
BCO1 = json.loads((HERE/'design/protocols/BCO1.json').read_text())
RUNS = {'gct2':'growth_correspondence_confirmation_v1', 'bco1':'bdt_cost_optimization_v1'}
BACKEND = {'native':'validation_v6_1.py','role':'role_adamw.py','temper':'tempering_reference.py','cms_protocol':'cms_protocol.py','geometry':'cms_geometry.py'}
CACHE_A = {'metadata.json':'9516fcab177c7e097b38a4e4ce2dc1f57bc54b83e2803f9d10ea76da3f0d7061','train.bin':'7ca3a6af7631c8f9566246ed5e1eaf6371b3d77221fd977820b8c46e01fb7b7b','val.bin':'74e9bbf7d1cdb114b84b7b74e613223c4afe60e7e902548ee951f841ad90cc6e'}
OCB_SOURCE='e3a4d782bb662ce6eec7674aca00ce16304aa29d87673f101dbbfe4392b80d99'

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def compact(x): return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def root(project,study='gct2'): return Path(project).resolve()/'runs'/RUNS[study]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()
def fsync_dir(path):
    fd=os.open(str(path),os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with tmp.open('x') as f:
            json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path);fsync_dir(path.parent)
    finally:
        if tmp.exists(): tmp.unlink()
def immutable(path,obj):
    path=Path(path)
    if path.exists():
        if read(path)!=obj: raise RuntimeError('Immutable record changed: '+str(path))
    else: atomic(path,obj)
@contextlib.contextmanager
def lock(path,blocking=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as e: raise RuntimeError('Already locked: '+str(path)) from e
        try: yield
        finally: fcntl.flock(f,fcntl.LOCK_UN)
def safe_file(base,rel):
    base=Path(base).resolve();p=base/rel
    if p.is_symlink() or not p.resolve().is_relative_to(base): raise RuntimeError('Unsafe path: '+str(p))
    return p
def verify(project=None):
    m=read(HERE/'SOURCE_MANIFEST.json')
    for rel,h in m.items():
        f=safe_file(HERE,rel)
        if not f.is_file() or sha(f)!=h: raise RuntimeError('Release source changed: '+rel)
    live={str(f.relative_to(HERE)) for f in HERE.rglob('*.py') if '__pycache__' not in f.parts}
    if live-set(m): raise RuntimeError('Unregistered Python source')
    return digest(m),{k:HERE/'backend'/n for k,n in BACKEND.items()}
def verify_source_only(): return verify()[0]
def load_module(name,path):
    path=Path(path).resolve()
    if name in sys.modules:
        mod=sys.modules[name]
        if Path(mod.__file__).resolve()!=path: raise RuntimeError('Module collision: '+name)
        return mod
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def backend(project=None):
    _,paths=verify(project)
    out={}
    for k,p in paths.items():
        name='cms_protocol' if k=='cms_protocol' else '_confirm_backend_'+k
        out[k]=load_module(name,p)
    return out

def engine(project,study='gct2'):
    """Load unchanged GCT1 arithmetic (metadata-only adapter) under a separate operational protocol/root."""
    spec=GCT2 if study=='gct2' else BCO1
    if 'gct_protocol' in sys.modules:
        if sys.modules['gct_protocol'].VERSION!=RUNS[study]: raise RuntimeError('One study per worker process')
    else:
        p=types.ModuleType('gct_protocol');p.__file__=str(HERE/'core.py')
        p.VERSION=RUNS[study];p.HERE=HERE;p.SEEDS=tuple(GCT2['candidate_seeds'])
        p.ARMS=tuple(GCT2['arms']);p.PARENT_STEPS=7630;p.POST_STEPS=3000;p.PREFIX=200
        p.GATE_SEED=127;p.GATE_PARENT=200;p.GATE_END=420;p.GATE_INTERRUPT=310
        p.PARAMETERS={'parent':81074688,'post':123551232}
        p.DEPENDENCIES={k:[str((HERE/'backend'/n).relative_to(HERE.parent)),sha(HERE/'backend'/n)] for k,n in BACKEND.items()}
        p.CACHE=GCT2['cache_sha256'] if study=='gct2' else CACHE_A;p.digest=digest;p.PROTOCOL_SHA=compact(spec)
        p.SPEC=dict(spec,scientific_optimizer_updates=97380);p.root=lambda project:root(project,study)
        sys.modules['gct_protocol']=p;sys.modules['gct_io']=sys.modules[__name__]
    rt=load_module('gct_runtime',HERE/'frozen/gct1/gct_runtime.py')
    load_module('gct_policy',HERE/'frozen/gct1/gct_policy.py')
    train=load_module('gct_train',HERE/'gct_train_adapter.py')
    return train,rt,backend(project)

def ocb_modules(project):
    """Only read/reuse OCB1 arithmetic and snapshots. Never writes OCB1 records."""
    engine(project,'bco1')
    if 'common' not in sys.modules:
        m=types.ModuleType('common');m.__file__=str(HERE/'core.py')
        for n in ('read','sha','atomic','lock','safe_file','digest','now','guard_cache','backend'):
            setattr(m,n,globals()[n])
        m.benchmark_design=lambda:read(HERE/'frozen/ocb1/OCB1.json')
        m.verify=lambda project:verify(project)
        m.verify_source_only=verify_source_only
        m.root=lambda project,study:Path(project)/'runs'/'optimizer_cost_benchmark_v1'
        sys.modules['common']=m
    model=load_module('modeling',HERE/'frozen/ocb1/modeling.py')
    ocb=load_module('_bco_original_ocb',HERE/'frozen/ocb1/ocb.py')
    return model,ocb

def stamp(path):
    st=Path(path).stat();return dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino)
def guard_cache(data):
    if {n:stamp(Path(data['path'])/n) for n in data['stat']}!=data['stat']: raise RuntimeError('Cache changed while executing')
def cache(project,study):
    d=Path(project)/'data_s1/fineweb_edu_3b';wanted=GCT2['cache_sha256'] if study=='gct2' else CACHE_A
    before={n:stamp(d/n) for n in wanted}
    if before['train.bin']['bytes']!=6000000000 or before['val.bin']['bytes']!=10000000: raise RuntimeError('Wrong cache sizes')
    for n,h in wanted.items():
        print('[CACHE SHA256]',n,flush=True)
        if sha(d/n)!=h: raise RuntimeError('Wrong token realization: '+n)
    result=dict(path=str(d.resolve()),hashes=wanted,stat=before,label='S1-B' if study=='gct2' else 'OWN-S1-R1')
    guard_cache(result);return result

def save_ck(directory,payload,torch,identity,keep=False):
    directory=Path(directory);store=directory/'checkpoints';store.mkdir(parents=True,exist_ok=True)
    old=read(directory/'latest.json') if (directory/'latest.json').exists() else None
    f=store/f"step_{payload['step']}_{uuid.uuid4().hex}.pt";tmp=f.with_suffix('.tmp')
    try:
        with tmp.open('xb') as out: torch.save(payload,out);out.flush();os.fsync(out.fileno())
        os.replace(tmp,f);fsync_dir(store)
        meta=dict(path=str(f.relative_to(directory)),sha256=sha(f),step=payload['step'],contract_sha=digest(payload['contract']),identity=identity,keep=keep)
        if keep: atomic(directory/'retained'/f"step_{payload['step']}.json",meta)
        atomic(directory/'latest.json',meta)
    finally:
        if tmp.exists(): tmp.unlink()
    if old and not old.get('keep'):
        prev=safe_file(directory,old['path'])
        if prev.parent!=store.resolve(): raise RuntimeError('Unsafe cleanup')
        if prev.exists(): prev.unlink()
    return meta

def load_ck(directory,torch):
    directory=Path(directory).resolve();meta=read(directory/'latest.json');f=safe_file(directory,meta['path'])
    if sha(f)!=meta['sha256']: raise RuntimeError('Checkpoint bytes changed: '+str(f))
    ck=torch.load(f,map_location='cpu',weights_only=False)
    if ck['step']!=meta['step'] or digest(ck['contract'])!=meta['contract_sha']: raise RuntimeError('Checkpoint contract changed')
    return ck,meta

def verify_completed(directory,expected=None,hash_payload=True):
    """No torch import; must pass before any SKIP_COMPLETE or cohort seal."""
    directory=Path(directory)
    if (directory/'fatal.json').exists(): raise RuntimeError('Fatal job needs investigation: '+str(directory))
    f=directory/'result.json'
    if not f.exists(): return None
    r=read(f);ct=r['contract']
    if r['state']!='complete' or digest(ct)!=r['contract_sha'] or ct!=read(directory/'contract.json'): raise RuntimeError('Invalid completed contract')
    if ct['source']!=verify()[0]: raise RuntimeError('Completed source changed')
    if expected:
        for k,v in expected.items():
            if ct.get(k)!=v: raise RuntimeError('Completed job identity changed: '+k)
    m=read(directory/'latest.json')
    if m!=r['checkpoint'] or m['identity']!=r['final'] or r['step']!=m['step'] or m['step']!=ct['end']: raise RuntimeError('Final checkpoint metadata mismatch')
    fck=safe_file(directory,m['path'])
    if not fck.is_file() or (hash_payload and sha(fck)!=m['sha256']): raise RuntimeError('Missing/changed final checkpoint')
    return r
