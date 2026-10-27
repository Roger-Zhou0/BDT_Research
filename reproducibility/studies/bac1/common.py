"""BAA1 transactional I/O and frozen arithmetic loading. No CUDA on import."""
from __future__ import annotations
import contextlib, datetime, fcntl, hashlib, importlib.util, json, os, sys, types, uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent
SPEC=json.loads((HERE/'PROTOCOL.json').read_text())
BACKEND={'native':'validation_v6_1.py','role':'role_adamw.py','temper':'tempering_reference.py','cms_protocol':'cms_protocol.py','geometry':'cms_geometry.py'}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def root(project):return Path(project).resolve()/'runs'/SPEC['version']
def safe_file(base,rel):
 base=Path(base).resolve();p=base/rel
 if p.is_symlink() or not p.resolve().is_relative_to(base):raise RuntimeError('Unsafe path: '+str(p))
 return p
@contextlib.contextmanager
def lock(path,blocking=False):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB))
  except BlockingIOError as e:raise RuntimeError('Already locked: '+str(p)) from e
  try:yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)
def fsync_dir(p):
 fd=os.open(str(p),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def atomic(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.'+uuid.uuid4().hex+'.tmp')
 try:
  with tmp.open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,p);fsync_dir(p.parent)
 finally:
  if tmp.exists():tmp.unlink()
def immutable(path,obj):
 p=Path(path)
 if p.exists():
  if read(p)!=obj:raise RuntimeError('Immutable record differs: '+str(p))
 else:atomic(p,obj)
def verify(project=None):
 m=read(HERE/'SOURCE_MANIFEST.json')
 for rel,h in m.items():
  p=safe_file(HERE,rel)
  if not p.is_file() or sha(p)!=h:raise RuntimeError('BAA1 source mismatch: '+rel)
 live={str(f.relative_to(HERE)) for f in HERE.rglob('*.py') if '__pycache__' not in f.parts}
 if live-set(m):raise RuntimeError('Unregistered Python source')
 return digest(m),{k:HERE/'backend'/n for k,n in BACKEND.items()}
def module(name,path):
 path=Path(path).resolve()
 if name in sys.modules:
  m=sys.modules[name]
  if Path(m.__file__).resolve()!=path:raise RuntimeError('Module collision: '+name)
  return m
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def backend(project=None):
 _,paths=verify(project)
 return {k:module('cms_protocol' if k=='cms_protocol' else '_asf_backend_'+k,v) for k,v in paths.items()}
def engine(project,reference=False):
 if 'gct_protocol' not in sys.modules:
  p=types.ModuleType('gct_protocol');p.__file__=str(HERE/'common.py');p.VERSION=SPEC['version'];p.HERE=HERE
  p.SEEDS=tuple(SPEC['seeds']);p.ARMS=tuple(SPEC['arms']);p.PARENT_STEPS=7630;p.POST_STEPS=SPEC['post_updates'];p.PREFIX=200;p.GATE_SEED=127;p.GATE_PARENT=200;p.GATE_END=420;p.GATE_INTERRUPT=310
  p.PARAMETERS={'parent':81074688,'post':123551232};p.CACHE=SPEC['cache_sha256'];p.PROTOCOL_SHA=digest(SPEC);p.digest=digest;p.root=root;p.SPEC=SPEC
  p.DEPENDENCIES={k:[str((HERE/'backend'/n).relative_to(HERE.parent)),sha(HERE/'backend'/n)] for k,n in BACKEND.items()}
  sys.modules['gct_protocol']=p;sys.modules['gct_io']=sys.modules[__name__]
 rt=module('gct_runtime',HERE/'frozen/gct_runtime.py');module('gct_policy',HERE/'frozen/gct_policy.py')
 tr=module('_asf_reference_train' if reference else 'asf_train',HERE/'frozen/gct_train.py' if reference else HERE/'train.py')
 return tr,rt,backend(project)
def stamp(p):
 s=Path(p).stat();return {'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino}
def guard_cache(d):
 if {n:stamp(Path(d['path'])/n) for n in d['stat']}!=d['stat']:raise RuntimeError('Cache changed during execution')
def cache(project):
 p=Path(project)/'data_s1/fineweb_edu_3b';before={n:stamp(p/n) for n in SPEC['cache_sha256']}
 if before['train.bin']['bytes']!=6000000000 or before['val.bin']['bytes']!=10000000:raise RuntimeError('Wrong S1-B token-file sizes')
 for n,h in SPEC['cache_sha256'].items():
  print('[BAA1 CACHE]',n,flush=True)
  if sha(p/n)!=h:raise RuntimeError('Wrong S1-B bytes: '+n)
 d={'path':str(p.resolve()),'hashes':SPEC['cache_sha256'],'stat':before};guard_cache(d);return d
def save_ck(directory,payload,torch,identity,keep=False):
 d=Path(directory);store=d/'checkpoints';store.mkdir(parents=True,exist_ok=True);old=read(d/'latest.json') if (d/'latest.json').exists() else None
 f=store/f"step_{payload['step']}_{uuid.uuid4().hex}.pt";tmp=f.with_suffix('.tmp')
 try:
  with tmp.open('xb') as o:torch.save(payload,o);o.flush();os.fsync(o.fileno())
  os.replace(tmp,f);fsync_dir(store)
  meta=dict(path=str(f.relative_to(d)),sha256=sha(f),step=payload['step'],contract_sha=digest(payload['contract']),identity=identity,keep=keep)
  if keep:atomic(d/'retained'/f"step_{payload['step']}.json",meta)
  atomic(d/'latest.json',meta)
 finally:
  if tmp.exists():tmp.unlink()
 if old and not old.get('keep'):
  prev=safe_file(d,old['path'])
  if prev.parent!=store.resolve():raise RuntimeError('Unsafe rolling-checkpoint cleanup')
  if prev.exists():prev.unlink()
 return meta
def load_ck(directory,torch):
 d=Path(directory).resolve();m=read(d/'latest.json');f=safe_file(d,m['path'])
 if sha(f)!=m['sha256']:raise RuntimeError('Checkpoint bytes differ: '+str(f))
 ck=torch.load(f,map_location='cpu',weights_only=False)
 if ck['step']!=m['step'] or digest(ck['contract'])!=m['contract_sha']:raise RuntimeError('Checkpoint contract mismatch')
 return ck,m
def completed(directory,expected=None,hash_payload=True):
 d=Path(directory)
 if (d/'fatal.json').exists():raise RuntimeError('Fatal job requires investigation: '+str(d))
 if not (d/'result.json').exists():return None
 r=read(d/'result.json');ct=r['contract'];m=read(d/'latest.json')
 if r['state']!='complete' or ct!=read(d/'contract.json') or digest(ct)!=r['contract_sha']:raise RuntimeError('Completed contract invalid')
 if ct['source']!=verify()[0]:raise RuntimeError('Completed source differs')
 if expected and any(ct.get(k)!=v for k,v in expected.items()):raise RuntimeError('Completed job identity differs')
 if r['checkpoint']!=m or r['final']!=m['identity'] or r['step']!=m['step'] or m['step']!=ct['end']:raise RuntimeError('Completed checkpoint link invalid')
 f=safe_file(d,m['path'])
 if not f.is_file() or hash_payload and sha(f)!=m['sha256']:raise RuntimeError('Completed checkpoint absent/changed')
 return r
