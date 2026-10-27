#!/usr/bin/env python3
"""Detached FOLLOWUP queue; waits without holding a GPU or blocking FEV1 admission."""
from __future__ import annotations
import argparse,contextlib,getpass,json,os,shutil,signal,socket,subprocess,sys,time,zipfile
from pathlib import Path
import common as io
STOP=False
CHILD=None

def say(*a):print(io.now(),*a,flush=True)
def session():
 user=getpass.getuser();host=socket.gethostname()
 if user not in io.SPEC['accounts'] or host not in ('jupyter-'+user,'jupyter-'+user+'--good'):raise RuntimeError('Unregistered account/session')
 name=('A' if user=='account_a' else 'B')+('-good' if host.endswith('--good') else '-default')
 if name not in io.SPEC['lane_seeds']:raise RuntimeError('Unassigned study lane')
 return dict(account=user,hostname=host,lane=name)
def lane():return session()['lane']
def lane_dir(project):return io.root(project)/'lanes'/lane()
def process_identity(pid):
 p=Path('/proc')/str(pid)
 try:return dict(pid=pid,start_ticks=(p/'stat').read_text().split(') ',1)[1].split()[19],cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))
 except (OSError,IndexError):return None

def prerequisites(project,lane_override=None):
 for study in io.SPEC['depends']:
  d=Path(project)/'runs'/study;f=d/'status.json'
  if not f.is_file() or io.read(f).get('state')!='finished':return False,'Waiting for complete '+study+' cohort (all parents)'
  expected=io.SPEC['prerequisite_sources'][study]
  if io.read(d/'source_registration.json')!=expected:raise RuntimeError('Predecessor source/protocol differs')
  analysis=io.read(d/'analysis.json')
  if analysis.get('state')!='complete' or any(analysis.get(k)!=v for k,v in expected.items()):raise RuntimeError('Predecessor analysis not verified')
  protocol=io.read(d/'PROTOCOL.json')
  if io.digest(protocol)!=expected['protocol']:raise RuntimeError('Predecessor protocol changed')
  for seed in protocol['seeds']:
   for mode in protocol['arms']:
    r=io.read(d/f'seed_{seed}'/'arms'/mode/'result.json')
    if r.get('state')!='complete' or r.get('step')!=protocol['post_updates'] or any(r['contract'].get(k)!=v for k,v in expected.items()):raise RuntimeError('Predecessor endpoint incomplete/mixed')
 return True,'All PCT1 parents complete; effect sign not used for admission'

def local_prerequisites(project,lane_override=None):
 ln=lane_override or lane();study='predictive_copy_correspondence_v1';d=Path(project)/'runs'/study
 f=d/'lanes'/ln/'status.json'
 if not f.is_file() or io.read(f).get('state') not in ('training_complete','finished'):return False,'Waiting for this lane to finish PCT1 before engineering preparation'
 expected=io.SPEC['prerequisite_sources'][study]
 if io.read(d/'source_registration.json')!=expected:raise RuntimeError('Local predecessor source/protocol differs')
 protocol=io.read(d/'PROTOCOL.json')
 if io.digest(protocol)!=expected['protocol']:raise RuntimeError('Local predecessor protocol changed')
 for seed in io.SPEC['lane_seeds'][ln]:
  for mode in protocol['arms']:
   r=io.read(d/f'seed_{seed}'/'arms'/mode/'result.json')
   if r.get('state')!='complete' or r.get('step')!=protocol['post_updates'] or any(r['contract'].get(k)!=v for k,v in expected.items()):raise RuntimeError('This lane still has incomplete/mixed PCT1 work')
 return True,'This lane released by PCT1; engineering-only preparation permitted, no scientific advancement'

def other_work():
 found=[]
 for p in Path('/proc').glob('[0-9]*'):
  try:
   pid=int(p.name)
   if pid==os.getpid() or p.stat().st_uid!=os.getuid():continue
   ws=[s.decode(errors='replace') for s in (p/'cmdline').read_bytes().split(b'\0') if s]
   scripts=[Path(s).name for s in ws[1:5] if s.endswith('.py')]
   if any(s in ('scheduler.py','coordinator.py','replay.py') for s in scripts) or (any(s in ('study.py','overnight.py','run.py','rgs_queue.py','asf_queue.py','followup_queue.py') for s in scripts) and any(s in ws for s in ('_worker','_evaluate','worker','train','_run'))):found.append(dict(pid=pid,command=ws))
  except (OSError,ValueError):continue
 return found

def gpu():
 r=subprocess.run(['nvidia-smi','-i','0','--query-gpu=name,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
 vals=[x.strip() for x in r.stdout.strip().split(',')]
 if r.returncode or len(vals)!=4 or '\n' in r.stdout.strip() or 'H200' not in vals[0] or not vals[1].startswith('GPU-'):raise RuntimeError('Cannot identify one H200 GPU')
 return dict(name=vals[0],uuid=vals[1],free_mib=int(vals[2]),util=int(vals[3]))
def memory_wait(uuid):
 good=0;start=time.monotonic()
 while good<2:
  if STOP:raise InterruptedError('Stop requested')
  g=gpu()
  if g['uuid']!=uuid:raise RuntimeError('GPU changed within coordinator process')
  good=good+1 if g['free_mib']>=io.SPEC['gpu_free_mib'] else 0
  if good<2:time.sleep(10)
  if time.monotonic()-start>3600:raise RuntimeError('GPU memory admission exceeded one hour')

def runtime_probe(project):
 r=subprocess.run([sys.executable,str(io.HERE/'followup_queue.py'),'runtime','--project',str(project)],capture_output=True,text=True,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='0'),timeout=120)
 if r.returncode:raise RuntimeError('Runtime probe failed: '+r.stdout+r.stderr)
 return json.loads(r.stdout.strip().splitlines()[-1])

def expected(j,rt):return dict(seed=j['seed'],stage='grown' if j['stage']=='import' else j['stage'],mode=j['mode'],end=j['end'],gate=j['gate'],tiny=j.get('tiny',False),runtime=rt,protocol=io.digest(io.SPEC))

def run_job(project,j,data_path,rt,gid,jobkey,resume):
 global CHILD
 root=io.root(project);jb=root/'jobs'/jobkey;reg=jb/'job.json';ledger=jb/'attempts.json'
 stable=dict(j);stable.pop('interrupt',None)
 io.immutable(reg,dict(job=stable,source=io.verify()[0],protocol=io.digest(io.SPEC),runtime=rt))
 r=io.completed(j['directory'],expected(j,rt))
 if r:say('SKIP_COMPLETE',jobkey,io.sha(Path(j['directory'])/'result.json'));return r
 state=io.read(ledger) if ledger.exists() else {'oom_failures':0,'attempts':[]}
 if state.get('fatal'):raise RuntimeError('Prior fatal attempt: '+str(ledger))
 if state['attempts'] and not resume and not j.get('interrupt'):raise RuntimeError('Existing incomplete attempt requires explicit resume')
 while True:
  if STOP:raise InterruptedError('Stop requested')
  if state['oom_failures']>=3:raise RuntimeError('Durable three-OOM limit reached')
  memory_wait(gid);io.verify()
  if shutil.disk_usage(project).free<12*1024**3:raise RuntimeError('Less than12GiB free disk')
  label=Path(j['directory']).name;logs=Path(project)/'logs'/io.SPEC['short']/lane();logs.mkdir(parents=True,exist_ok=True)
  logfile=logs/(jobkey.replace('/','_')+'_'+str(time.time_ns())+'.log');req=jb/('attempt_'+str(time.time_ns())+'.json')
  io.atomic(req,dict(job=j,data_path=str(data_path),runtime=rt,gpu_uuid=gid,source=io.verify()[0],resume=resume or bool(state['attempts'])))
  a={'start':io.now(),'request':str(req),'request_sha256':io.sha(req),'log':str(logfile),'gpu_uuid':gid};state['attempts'].append(a);io.atomic(ledger,state)
  with logfile.open('xb',buffering=0) as f:
   CHILD=subprocess.Popen([sys.executable,'-u',str(io.HERE/'followup_queue.py'),'worker','--project',str(project),'--request',str(req)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=project,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4'))
   a['pid']=CHILD.pid;io.atomic(ledger,state)
   io.atomic(lane_dir(project)/'status.json',dict(state='gate' if j['gate'] else 'training',label=jobkey,pid=os.getpid(),child_pid=CHILD.pid,child_log=str(logfile),gpu_uuid=gid,time=io.now()))
   (Path(project)/f"logs/{io.SPEC['short']}_{lane()}_child_latest.txt").write_text(str(logfile)+'\n');say('START',jobkey,CHILD.pid,logfile)
   code=CHILD.wait();CHILD=None
  a.update(exit_code=code,end=io.now())
  if code==75:state['oom_failures']+=1
  elif code>0 and code!=76:state['fatal']=True
  io.atomic(ledger,state)
  if code==0:
   if j.get('interrupt'):return {'state':'test_paused'}
   r=io.completed(j['directory'],expected(j,rt))
   if r is None:raise RuntimeError('Child exited without a complete result')
   say('DONE',jobkey);return r
  if code==75 and state['oom_failures']<3:resume=True;say('OOM same-recipe retry',jobkey);time.sleep(30);continue
  if code<0 or code==76 or STOP:raise InterruptedError('Infrastructure interruption; explicit --resume required')
  raise RuntimeError('Worker failure: '+str(logfile))

def gate(project,rt,data_path,gid,resume):
 import experiment
 root=io.root(project);key=lane()+'_'+gid+'_'+io.digest(rt)[:16]+'_'+io.verify()[0][:16];d=root/'gates'/key;done=d/'gate.json'
 if done.exists():
  r=io.read(done)
  if r.get('state')!='passed' or r.get('source')!=io.verify()[0] or r.get('runtime')!=rt or r.get('gpu_uuid')!=gid:raise RuntimeError('Gate identity changed')
  for rel,h in r['records'].items():
   if io.sha(io.safe_file(root,rel))!=h:raise RuntimeError('Gate record changed')
  say('SKIP_PASSED_GPU_GATE',key);return
 for j in experiment.gate_plan(d):
  label=Path(j['directory']).name;run_job(project,j,data_path,rt,gid,'gate/'+key+'/'+label,True)
 # Compare without creating a GPU context in the waiting/coordinating process.
 outfile=d/'comparison.json'
 r=subprocess.run([sys.executable,str(io.HERE/'followup_queue.py'),'compare','--project',str(project),'--directory',str(d),'--output',str(outfile)],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUBLAS_WORKSPACE_CONFIG=':4096:8'),timeout=300)
 if r.returncode:raise RuntimeError('Full-size gate comparison failed')
 checks=io.read(outfile)
 records={str(f.relative_to(root)):io.sha(f) for f in d.rglob('*.json') if f.name in ('result.json','contract.json','comparison.json')}
 io.atomic(done,dict(state='passed',source=io.verify()[0],runtime=rt,gpu_uuid=gid,checks=checks,records=records,time=io.now()))
 say('FOLLOWUP FULL-SIZE REFERENCE + SCOPE + RESUME GATE: PASS')
 # Only this gate's own disposable checkpoint payloads; preserve checks/hashes/traces.
 cleanup=[]
 for f in d.rglob('checkpoints/*.pt'):
  if f.is_symlink() or not f.resolve().is_relative_to(d.resolve()):raise RuntimeError('Unsafe gate cleanup')
  cleanup.append({'path':str(f.relative_to(d)),'sha256':io.sha(f)});f.unlink()
 io.atomic(d/'checkpoint_cleanup.json',dict(removed=cleanup,time=io.now()))


def coordinate(project,resume,science=False):
 from operations import coordinate as execute
 return execute(project,resume,science)

def detached(project,resume=False,science=False):
 identity=session();root=io.root(project);logdir=Path(project)/'logs';logdir.mkdir(exist_ok=True)
 with io.lock(logdir/('.'+io.SPEC['short']+'_launch.lock')):
  p=lane_dir(project)/'coordinator.json'
  if p.exists():
   old=io.read(p);live=process_identity(old['pid'])
   if live and live.get('start_ticks')==old.get('start_ticks') and 'followup_queue.py' in live['cmdline']:raise RuntimeError('FOLLOWUP coordinator already live; do not relaunch')
   if not resume:raise RuntimeError('Previous queue record exists; inspect and use --resume')
  # Enforce no simultaneous GUI/automation installation of another scientific recipe.
  io.immutable(root/'PROTOCOL.json',io.SPEC);io.immutable(root/'source_registration.json',dict(source=io.verify()[0],protocol=io.digest(io.SPEC)))
  out=logdir/((io.SPEC['short']+'_'+lane()+'_')+str(time.time_ns())+'.log')
  cmd=[sys.executable,'-u',str(io.HERE/'followup_queue.py'),'coordinate','--project',str(project)]
  if resume:cmd.append('--resume')
  if science:cmd.append('--science')
  with out.open('xb',buffering=0) as f:proc=subprocess.Popen(cmd,cwd=project,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4'))
  pi=process_identity(proc.pid) or {'pid':proc.pid};io.atomic(p,dict(pi,time=io.now(),session=identity,source=io.verify()[0]));(logdir/f"{io.SPEC['short']}_{lane()}_latest.txt").write_text(str(out)+'\n')
  print('FOLLOWUP_COORDINATOR_PID='+str(proc.pid)+'\nLOG='+str(out)+'\nQueued process must show WAIT, gate progress, or completed state; PID alone is not training.',flush=True)

def main():
 a=argparse.ArgumentParser();a.add_argument('action',choices=['launch','coordinate','worker','runtime','compare','status','integrity','tests','export']);a.add_argument('--project',type=Path,default=io.HERE.parent);a.add_argument('--request',type=Path);a.add_argument('--directory',type=Path);a.add_argument('--output',type=Path);a.add_argument('--resume',action='store_true');a.add_argument('--science',action='store_true');args=a.parse_args();project=args.project.resolve();io.verify()
 if args.action=='integrity':print('FOLLOWUP SOURCE INTEGRITY: PASS',io.verify()[0]);return
 if args.action=='launch':detached(project,args.resume,args.science);return
 if args.action=='coordinate':coordinate(project,args.resume,args.science);return
 if args.action=='status':
  whole=io.root(project)/'status.json'
  print('cohort',io.read(whole) if whole.exists() else 'NOT COMPLETE')
  for name in ('status.json','coordinator.json'):
   p=lane_dir(project)/name;print(name,json.dumps(io.read(p),indent=2) if p.exists() else 'NOT PRESENT')
  ptr=project/f"logs/{io.SPEC['short']}_{lane()}_latest.txt"
  if ptr.exists():
   f=Path(ptr.read_text().strip())
   if f.is_file():
    with f.open('rb') as h:h.seek(max(0,f.stat().st_size-4000));print(h.read().decode(errors='replace'))
  return
 if args.action=='tests':
  r=subprocess.run([sys.executable,str(io.HERE/'tests/test_suite.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUBLAS_WORKSPACE_CONFIG=':4096:8'));raise SystemExit(r.returncode)
 if args.action=='export':
  dest=project/'exports'/('baa1_'+str(time.time_ns())+'.zip');dest.parent.mkdir(exist_ok=True)
  with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
   for base,prefix in [(io.HERE,'source'),(io.root(project),'records')]:
    for f in base.rglob('*'):
     if f.is_file() and not f.is_symlink() and f.suffix in ('.json','.md','.csv','.py','.sh') and '__pycache__' not in f.parts:z.write(f,prefix+'/'+str(f.relative_to(base)))
   logs=project/'logs'/io.SPEC['short']/lane()
   for f in logs.glob('*.log'):
    if f.is_symlink():continue
    with f.open('rb') as h:h.seek(max(0,f.stat().st_size-8000));z.writestr('log_tails/'+f.name,h.read())
  print('EXPORT_ZIP='+str(dest));return
 if args.action=='runtime':
  import torch
  _,rt,_=io.engine(project);rt.configure(4);print(json.dumps(rt.runtime(torch.device('cuda:0'))));return
 if args.action=='compare':
  import experiment
  io.engine(project);io.atomic(args.output,experiment.compare_gate(args.directory));return
 if args.action=='worker':
  import torch,experiment
  req=io.read(args.request)
  if req['source']!=io.verify()[0] or gpu()['uuid']!=req['gpu_uuid']:raise RuntimeError('Worker source/GPU changed')
  tr,rt,_=io.engine(project,req['job'].get('reference',False));rt.configure(4)
  if rt.runtime(torch.device('cuda:0'))!=req['runtime']:raise RuntimeError('Worker runtime changed')
  data=io.read(req['data_path']);io.guard_cache(data)
  try:
   r=experiment.execute(project,req['job'],data,req['resume'])
   if r.get('state')=='interrupted':raise SystemExit(76)
  except torch.OutOfMemoryError:raise SystemExit(75)
  return
if __name__=='__main__':
 try:main()
 except Exception as e:
  import traceback
  print('STOPPED:',type(e).__name__,str(e),flush=True);traceback.print_exc();raise SystemExit(1)
