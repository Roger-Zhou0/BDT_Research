"""Hashed dependencies and transactional checkpoints scoped to GCT1 only."""
import contextlib,datetime,fcntl,hashlib,importlib.util,json,os,sys,uuid
from pathlib import Path
import gct_protocol as p

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def atomic(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
 try:
  with tmp.open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if tmp.exists():tmp.unlink()
@contextlib.contextmanager
def lock(path,blocking=False):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB))
  except BlockingIOError as e:raise RuntimeError('Already locked: '+str(path)) from e
  try:yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)
def verify(project):
 m=read(p.HERE/'SOURCE_MANIFEST.json')
 for rel,want in m.items():
  q=p.HERE/rel
  if q.is_symlink() or sha(q)!=want:raise RuntimeError('GCT1 source changed: '+rel)
 if {f.name for f in p.HERE.glob('*.py')}-set(m):raise RuntimeError('Unregistered source file')
 paths={}
 for key,(rel,want) in p.DEPENDENCIES.items():
  q=Path(project)/rel
  if not q.is_file() or q.is_symlink() or sha(q)!=want:raise RuntimeError('Missing/mismatched frozen dependency: '+str(q))
  paths[key]=q.resolve()
 return p.digest(m),paths
def backend(project):
 _,paths=verify(project);out={}
 for key,path in paths.items():
  name='cms_protocol' if key=='cms_protocol' else '_gct1_frozen_'+key
  if name in sys.modules:
   mod=sys.modules[name]
   if Path(mod.__file__).resolve()!=path:raise RuntimeError('Mixed module location: '+name)
  else:
   spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
  out[key]=mod
 return out

def stamp(path):
 st=Path(path).stat();return dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino)
def cache(project):
 d=Path(project)/'data_s1/fineweb_edu_3b';before={n:stamp(d/n) for n in p.CACHE}
 if before['train.bin']['bytes']!=6000000000 or before['val.bin']['bytes']!=10000000:raise RuntimeError('Wrong S1-B cache sizes')
 for n,h in p.CACHE.items():
  print('[cache] read-only SHA256',n,flush=True)
  if sha(d/n)!=h:raise RuntimeError('Wrong token realization: '+n)
 out=dict(path=str(d.resolve()),label='S1-B',hashes=p.CACHE,stat=before)
 guard_cache(out);return out
def guard_cache(d):
 if {n:stamp(Path(d['path'])/n) for n in d['stat']}!=d['stat']:raise RuntimeError('Cache changed during campaign')

def save_ck(directory,payload,torch,identity,keep=False):
 directory=Path(directory);store=directory/'checkpoints';store.mkdir(parents=True,exist_ok=True)
 old=read(directory/'latest.json') if (directory/'latest.json').exists() else None
 f=store/f"step_{payload['step']}_{uuid.uuid4().hex}.pt";tmp=f.with_suffix('.tmp')
 try:
  with tmp.open('xb') as out:torch.save(payload,out);out.flush();os.fsync(out.fileno())
  os.replace(tmp,f)
  info=dict(path=str(f.relative_to(directory)),sha256=sha(f),step=payload['step'],contract_sha=p.digest(payload['contract']),identity=identity,keep=keep)
  if keep:atomic(directory/'retained'/f"step_{payload['step']}.json",info)
  atomic(directory/'latest.json',info)
 finally:
  if tmp.exists():tmp.unlink()
 if old and not old.get('keep'):
  prev=(directory/old['path'])
  if prev.parent.resolve()!=store.resolve() or prev.is_symlink():raise RuntimeError('Unsafe retention path')
  if prev.exists():prev.unlink()
 return info
def load_ck(directory,torch):
 directory=Path(directory).resolve();meta=read(directory/'latest.json');f=(directory/meta['path']);f.resolve().relative_to(directory)
 if f.is_symlink() or sha(f)!=meta['sha256']:raise RuntimeError('Checkpoint checksum mismatch: '+str(f))
 ck=torch.load(f,map_location='cpu',weights_only=False)
 if ck['step']!=meta['step'] or p.digest(ck['contract'])!=meta['contract_sha']:raise RuntimeError('Checkpoint contract mismatch')
 return ck,meta
