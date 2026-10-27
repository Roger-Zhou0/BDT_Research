#!/usr/bin/env python3
"""GCT1 detached scheduler. Guards before CUDA; no historical study edits."""
from __future__ import annotations
import argparse,contextlib,getpass,json,os,re,shutil,signal,socket,subprocess,sys,time,zipfile
from pathlib import Path
import gct_protocol as p
import gct_io as io
STOP=False
CHILD=None

def say(*x):print(io.now(),*x,flush=True)
def session_guard():
 prefix=os.environ.get('JUPYTERHUB_SERVICE_PREFIX','')
 if getpass.getuser()!='account_b' or not (socket.gethostname().endswith('--good') or prefix.rstrip('/').endswith('/account_b/good')):
  raise RuntimeError('GCT1 is assigned only to account_b/good, the former CMS1 session. Do not use the other queues.')
def gpu():
 r=subprocess.run(['nvidia-smi','-i','0','--query-gpu=name,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
 if r.returncode or len(r.stdout.strip().splitlines())!=1:raise RuntimeError('GPU query failed '+r.stdout+r.stderr)
 a=[z.strip() for z in r.stdout.strip().split(',')]
 if len(a)!=4 or not a[1].startswith('GPU-') or 'H200' not in a[0]:raise RuntimeError('Expected the assigned H200 at index0')
 return dict(name=a[0],uuid=a[1],free_mib=int(a[2]),utilization=int(a[3]))
def live_work():
 found=[]
 for d in Path('/proc').glob('[0-9]*'):
  try:
   if int(d.name)==os.getpid() or d.stat().st_uid!=os.getuid():continue
   words=[x.decode(errors='replace') for x in (d/'cmdline').read_bytes().split(b'\0') if x]
   for s in [x for x in words[1:4] if x.endswith('.py')]:
    if Path(s).name in ('scheduler.py','coordinator.py','execution_control_probe_v1.py','replay.py') or (Path(s).name in ('study.py','overnight.py') and any(x in words for x in ('_worker','_coordinate'))) or (Path(s).name=='run.py' and 'train' in words):
     found.append(dict(pid=int(d.name),command=words));break
  except (OSError,ValueError):continue
 return found

def seeds_guard(project):
 pattern=re.compile(r'(?:^|[^0-9])seed[_-]?('+ '|'.join(map(str,p.SEEDS))+r')(?:[^0-9]|$)',re.I);found=[]
 for dp,dirs,names in os.walk(Path(project)/'runs',followlinks=False):
  d=Path(dp);dirs[:]=[n for n in dirs if (d/n).resolve()!=p.root(project) and not (d/n).is_symlink()]
  for n in dirs+names:
   if pattern.search(n):found.append(str(d/n))
  for n in names:
   f=d/n
   if n not in ('result.json','contract.json','diverged.json') or f.is_symlink() or f.stat().st_size>5<<20:continue
   try:
    v=io.read(f);s=v.get('seed',(v.get('contract') or {}).get('seed'))
    if s in p.SEEDS:found.append(str(f))
   except (ValueError,OSError,AttributeError):continue
 if found:raise RuntimeError('Registered seed conflict; no silent replacement: '+repr(found[:10]))
 return dict(seeds=list(p.SEEDS),conflicts=[],time=io.now())

def cms_done(project):
 root=Path(project)/'runs/correspondence_matched_s1_v1'
 if io.read(root/'scheduler_status.json').get('state')!='finished' or io.read(root/'analysis.json').get('state')!='complete':raise RuntimeError('CMS1 must be complete before GCT1 uses this session')
 for s in range(147,153):
  for m in p.ARMS:
   if not (root/f'seed_{s}'/'arms'/m/'report.json').is_file():raise RuntimeError('CMS1 final REPORT records missing')
 return dict(state='complete',analysis_sha=io.sha(root/'analysis.json'))

def admit(target):
 start=time.monotonic();good=0
 while good<2:
  if STOP:raise InterruptedError('Stopped')
  g=gpu()
  if g['uuid']!=target:raise RuntimeError('GPU changed within coordinator; inspect before resume')
  good=good+1 if g['free_mib']>=32768 else 0
  if good<2:say('[GPU admission]',g['free_mib'],'MiB free; need 32768');time.sleep(15)
  if time.monotonic()-start>86400:raise RuntimeError('24h admission wait exhausted')

def run_child(project,label,command,target,on_gpu=True):
 global CHILD
 logdir=Path(project)/'logs/gct1';logdir.mkdir(parents=True,exist_ok=True);countfile=p.root(project)/'oom_counts'/f'{label}.json'
 while True:
  if STOP:raise InterruptedError('Stopped')
  io.verify(project)
  if on_gpu:admit(target)
  if shutil.disk_usage(project).free<10*1024**3:raise RuntimeError('Less than 10 GiB disk headroom; no historical files removed')
  count=io.read(countfile)['failures'] if countfile.exists() else 0
  if count>=3:raise RuntimeError('Three same-recipe OOM failures exhausted for '+label)
  cmd=list(command)
  if count and '--resume' not in cmd:cmd.append('--resume')
  path=logdir/f'{label}_{time.time_ns()}.log';env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';env['CUDA_VISIBLE_DEVICES']='0' if on_gpu else ''
  with path.open('xb',buffering=0) as f:
   CHILD=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=project,env=env)
   io.atomic(p.root(project)/'scheduler_status.json',dict(state='running',label=label,pid=os.getpid(),child_pid=CHILD.pid,child_log=str(path),gpu_uuid=target,time=io.now()))
   (Path(project)/'logs/gct1_current_child_log.txt').write_text(str(path)+'\n');say('[start]',label,CHILD.pid,path)
   code=CHILD.wait()
  CHILD=None
  if code==0:say('[done]',label);return
  if code==76 or STOP:raise InterruptedError('Child interrupted; committed state preserved')
  if code==75:
   count+=1;io.atomic(countfile,dict(failures=count,log=str(path),time=io.now()));say('[OOM] same-recipe retry from committed state');time.sleep(45);continue
  raise RuntimeError('Child failed; inspect '+str(path)+f' (exit {code})')

def job(project,name,values,target,resume=False):
 jp=p.root(project)/'jobs'/f'{name}.json';v=dict(values,source=io.verify(project)[0],protocol=p.PROTOCOL_SHA)
 if jp.exists() and io.read(jp)!=v:raise RuntimeError('Registered job changed: '+name)
 if not jp.exists():io.atomic(jp,v)
 cmd=[sys.executable,'-u',str(p.HERE/'study.py'),'_worker','--project',str(project),'--job',str(jp)]
 if resume:cmd.append('--resume')
 run_child(project,name,cmd,target)

def gate(project,target,resume=False):
 root=p.root(project)/'gates'/target;done=root/'gate.json';source=io.verify(project)[0]
 if done.exists():
  r=io.read(done)
  if r['state']!='passed' or r['source']!=source or r['protocol']!=p.PROTOCOL_SHA:raise RuntimeError('Stale gate')
  for rel,h in r['evidence'].items():
   if io.sha(root/rel)!=h:raise RuntimeError('Changed gate evidence '+rel)
  say('[gate already passed]',target);return r
 from gct_analysis import compare
 def tr(name,stage,mode,end,parent=None,interrupt=None,rs=None):
  return job(project,'gate_'+target+'_'+name,dict(directory=str(root/name),seed=127,stage=stage,mode=mode,end=end,parent=str(root/parent) if parent else None,gate=True,action='train',interrupt=interrupt),target,resume if rs is None else rs)
 tr('parent','parent','adamw',200)
 job(project,'gate_'+target+'_grow',dict(directory=str(root/'grown'),seed=127,parent=str(root/'parent'),gate=True,action='grow'),target,resume)
 tr('prefix','prefix','adamw',200,'grown');tr('reference_prefix','prefix','reference_adamw',200,'grown')
 checks={'prefix_vs_reference':compare(root/'prefix',root/'reference_prefix')}
 for m in p.ARMS:
  tr(m+'_0','arm',m,420,'prefix')
  if m=='scrambled_matched':
   dest=root/(m+'_1')
   if not (dest/'latest.json').exists():
    # Different job records, identical trajectory contract, explicit test interruption.
    job(project,'gate_'+target+'_scrambled_interrupt',dict(directory=str(dest),seed=127,stage='arm',mode=m,end=420,parent=str(root/'prefix'),gate=True,action='train',interrupt=310),target,resume)
   tr(m+'_1','arm',m,420,'prefix',rs=True)
  else:tr(m+'_1','arm',m,420,'prefix')
  checks[m+'_repeat_or_resume']=compare(root/(m+'_0'),root/(m+'_1'))
 tr('reference_adamw','arm','reference_adamw',420,'prefix');tr('reference_bdt','arm','reference_bdt',420,'prefix')
 checks['adamw_vs_frozen']=compare(root/'adamw_0',root/'reference_adamw');checks['bdt_vs_frozen']=compare(root/'bdt_0',root/'reference_bdt')
 first=[]
 for m in p.ARMS:
  r=io.read(root/(m+'_0')/'result.json');z=next(z for z in r['proof'] if z['step']==201)
  first.append({k:z[k] for k in ('gradient','raw','batch','rng_before','rng_after')})
 if any(z!=first[0] for z in first):raise RuntimeError('First active gradients/raw proposals disagree')
 checks['first_active_inputs_equal']=True
 growth=io.read(root/'grown/result.json')['checks']
 evidence={str(f.relative_to(root)):io.sha(f) for f in root.rglob('*.json') if f.name in ('result.json','contract.json','initialization.json')}
 r=dict(state='passed',time=io.now(),source=source,protocol=p.PROTOCOL_SHA,checks=checks,growth_checks=growth,evidence=evidence,
        scope='Full-size gate seed127; six-to-twelve growth. One tested runtime/device; not an efficacy result.')
 io.atomic(done,r);say('GCT1 FULL-SIZE GROWTH + ACTIVE-CONTROL GATE: PASS')
 # Engineering payloads are ours only, and the pass plus complete record evidence is already durable.
 removed=[]
 for d in root.iterdir():
  store=d/'checkpoints'
  if not store.is_dir() or store.is_symlink():continue
  for f in store.glob('*.pt'):
   if f.is_symlink():raise RuntimeError('Refuse symlink checkpoint cleanup')
   removed.append(dict(path=str(f.relative_to(root)),sha256=io.sha(f)));f.unlink()
 io.atomic(root/'engineering_payload_cleanup.json',dict(time=io.now(),removed=removed,scope='Only this passed GCT1 engineering gate; all scientific checkpoints and historical files untouched'))
 return r

def coordinate(project,resume=False):
 global CHILD,STOP
 session_guard();project=Path(project).resolve();source,_=io.verify(project);root=p.root(project)
 def stopped(*_):
  global STOP
  STOP=True
  if CHILD is not None and CHILD.poll() is None:CHILD.terminate()
 signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
 with contextlib.ExitStack() as stack:
  stack.enter_context(io.lock(Path(project)/'logs/.gct1.lock'));g=gpu();stack.enter_context(io.lock(Path(project)/'logs/.practical_gpu_locks'/(g['uuid']+'.lock')))
  if live_work():raise RuntimeError('Other active workload found in this session; none stopped: '+repr(live_work()))
  try:
   completed=cms_done(project);reg=root/'registration.json';fixed=dict(source=source,protocol=p.PROTOCOL_SHA,dependencies=p.DEPENDENCIES,seeds=list(p.SEEDS),cache=p.CACHE)
   if reg.exists():
    if io.read(reg)['fixed']!=fixed:raise RuntimeError('Registration changed')
    if not resume:raise RuntimeError('Already registered; inspect status and use explicit --resume')
   else:
    if shutil.disk_usage(project).free<64*1024**3:raise RuntimeError('Need 64 GiB disk headroom at first registration; nothing deleted')
    io.atomic(reg,dict(fixed=fixed,seed_audit=seeds_guard(project),cms1_completion=completed,gpu=g,time=io.now()));io.atomic(root/'PROTOCOL.json',p.SPEC)
   with (root/'execution_sessions.jsonl').open('a') as f:f.write(json.dumps(dict(time=io.now(),gpu=g,resume=resume))+'\n')
   data=io.cache(project);cp=root/'cache_identity.json'
   if cp.exists() and io.read(cp)!=data:raise RuntimeError('Cache path/stats/hash changed; audit before resuming')
   if not cp.exists():io.atomic(cp,data)
   ct=root/'cpu_tests.json'
   if not ct.exists():run_child(project,'cpu_tests',[sys.executable,'-u',str(p.HERE/'tests.py'),'--project',str(project),'--output',str(ct)],g['uuid'],on_gpu=False)
   if not io.read(ct).get('passed') or io.read(ct).get('source')!=source:raise RuntimeError('CPU gate missing or wrong release')
   gate(project,g['uuid'],resume)
   for seed in p.SEEDS:
    d=root/f'seed_{seed}'
    def run(name,stage,mode,end,parent=None):
     job(project,f'seed_{seed}_'+name.replace('/','_'),dict(directory=str(d/name),seed=seed,stage=stage,mode=mode,end=end,parent=str(d/parent) if parent else None,gate=False,action='train',interrupt=None),g['uuid'],resume)
    run('parent','parent','adamw',7630)
    job(project,f'seed_{seed}_grow',dict(directory=str(d/'grown'),seed=seed,parent=str(d/'parent'),gate=False,action='grow'),g['uuid'],resume)
    run('prefix','prefix','adamw',200,'grown')
    for m in p.SPEC['order'][str(seed)]:run('arms/'+m,'arm',m,3000,'prefix')
   from gct_analysis import analyze
   result=analyze(project)
   if result['state']!='complete':raise RuntimeError('Complete queue lacks complete endpoints')
   io.atomic(root/'scheduler_status.json',dict(state='finished',time=io.now(),pid=os.getpid(),gpu_uuid=g['uuid'],analysis=str(root/'analysis.json')))
   say('GCT1 STUDY COMPLETE; no further experiment launched')
  except BaseException as ex:
   io.atomic(root/'scheduler_status.json',dict(state='interrupted' if isinstance(ex,(KeyboardInterrupt,InterruptedError)) else 'failed',time=io.now(),pid=os.getpid(),error=str(ex),gpu_uuid=g['uuid']));raise

def launch(project,resume=False):
 session_guard();io.verify(project);logs=Path(project)/'logs';logs.mkdir(exist_ok=True)
 with io.lock(logs/'.gct1_launcher.lock'):
  with io.lock(logs/'.gct1.lock'):pass
  cmd=[sys.executable,'-u',str(p.HERE/'study.py'),'_coordinate','--project',str(project)]
  if resume:cmd.append('--resume')
  log=logs/f'gct1_{time.time_ns()}.log';env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';env['CUDA_VISIBLE_DEVICES']='0'
  with log.open('xb',buffering=0) as f:proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,cwd=project,start_new_session=True)
  (logs/'gct1_latest_log.txt').write_text(str(log)+'\n');time.sleep(.7)
  if proc.poll() not in (None,0):raise RuntimeError(log.read_text())
  print(f'COORDINATOR_PID={proc.pid}\nLOG={log}\nDetached. Full-size gate must pass before seed159 scientific training.')

def status(project):
 root=p.root(project);q=root/'scheduler_status.json';r=io.read(q) if q.exists() else {'state':'not_started'};print(json.dumps(r,indent=2))
 if r.get('state')=='running' and Path('/proc').exists() and not (Path('/proc')/str(r.get('pid'))/'cmdline').exists():print('WARNING: recorded coordinator absent; status may be stale.')
 for label,path in [('Small parents','parent'),('Grown snapshots','grown'),('Post-growth prefixes','prefix')]:
  print(label,sum((root/f'seed_{s}'/path/'result.json').exists() for s in p.SEEDS),'/2')
 print('Continuations',sum((root/f'seed_{s}'/'arms'/m/'result.json').exists() for s in p.SEEDS for m in p.ARMS),'/6')
 if r.get('child_log') and Path(r['child_log']).is_file():
  with Path(r['child_log']).open('rb') as f:f.seek(max(0,Path(r['child_log']).stat().st_size-10000));print('\n'.join(f.read().decode(errors='replace').splitlines()[-8:]))

def export(project):
 out=Path(project)/'exports';out.mkdir(exist_ok=True);dest=out/f'GCT1_{time.time_ns()}.zip';manifest={}
 with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
  for root,label in [(p.root(project),'results'),(p.HERE,'source'),(Path(project)/'logs/gct1','logs')]:
   for dp,dirs,files in os.walk(root,followlinks=False):
    d=Path(dp);dirs[:]=[n for n in dirs if n not in ('checkpoints','__pycache__') and not (d/n).is_symlink()]
    for n in files:
     f=d/n
     if f.is_symlink() or f.suffix not in ('.json','.md','.csv','.py','.sh','.jsonl','.log') or f.stat().st_size>20<<20:continue
     with f.open('rb') as h:
      if f.suffix in ('.log','.jsonl'):h.seek(max(0,f.stat().st_size-150000))
      data=h.read()
     if f.suffix in ('.log','.jsonl'):data=b'\n'.join(data.splitlines()[-500:])+b'\n'
     arc=label+'/'+str(f.relative_to(root));z.writestr(arc,data);manifest[arc]=__import__('hashlib').sha256(data).hexdigest()
  for key,q in io.verify(project)[1].items():arc='frozen/'+key+'.py';z.writestr(arc,q.read_bytes());manifest[arc]=io.sha(q)
  z.writestr('MANIFEST.json',json.dumps(manifest,indent=2));z.writestr('SNAPSHOT.txt','Per-file live snapshot; logs bounded. No checkpoint payloads or token arrays. Pilot only.\n')
 print('EXPORT_ZIP='+str(dest)+'\nSHA256='+io.sha(dest))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['launch','status','integrity','export','analyze','_coordinate','_worker']);ap.add_argument('--project',default=str(p.HERE.parent));ap.add_argument('--resume',action='store_true');ap.add_argument('--job');a=ap.parse_args();project=Path(a.project).resolve()
 if a.action=='integrity':print('GCT1 SOURCE + FROZEN DEPENDENCIES: PASS',io.verify(project)[0])
 elif a.action=='launch':launch(project,a.resume)
 elif a.action=='status':status(project)
 elif a.action=='export':export(project)
 elif a.action=='analyze':
  from gct_analysis import analyze
  print(json.dumps(analyze(project),indent=2))
 elif a.action=='_coordinate':coordinate(project,a.resume)
 else:
  session_guard();q=Path(a.job).resolve();q.relative_to(p.root(project));j=io.read(q)
  if j['source']!=io.verify(project)[0] or j['protocol']!=p.PROTOCOL_SHA:raise RuntimeError('Worker job provenance changed')
  if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('Workspace must be set before torch import')
  import gct_train as tr
  from gct_runtime import configure
  configure();data=io.read(p.root(project)/'cache_identity.json');kwargs={k:v for k,v in j.items() if k not in ('source','protocol','action')}
  try:
   fn=tr.grow if j['action']=='grow' else tr.train;r=fn(project,**kwargs,data=data,resume=a.resume)
   if r.get('state')=='interrupted':raise SystemExit(76)
  except tr.torch.OutOfMemoryError:__import__('traceback').print_exc();raise SystemExit(75)
if __name__=='__main__':
 try:main()
 except Exception as ex:print('STOPPED:',type(ex).__name__,str(ex),file=sys.stderr);__import__('traceback').print_exc();raise SystemExit(1)
