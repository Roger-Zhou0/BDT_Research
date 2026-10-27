#!/usr/bin/env python3
from pathlib import Path
import argparse,fcntl,json,re
import protocol as p
from parent_evidence import verify_parent
from select_recipes import choose,verify_locked

def main():
 a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','train']);a.add_argument('--project',required=True)
 a.add_argument('--stage',choices=['development','report']);a.add_argument('--seed',type=int)
 a.add_argument('--method',choices=p.METHODS);a.add_argument('--lr',type=float);x=a.parse_args()
 project=Path(x.project).resolve();root=p.campaign(project);p.verify_sources();verify_parent(project)
 import torch
 import model_support as ms
 torch.set_num_threads(4);root.mkdir(parents=True,exist_ok=True)
 if x.action=='prepare':
  with (root/'.prepare.lock').open('a') as f:
   fcntl.flock(f,fcntl.LOCK_EX)
   proto=root/'PROTOCOL.json'
   if proto.exists() and json.loads(proto.read_text())!=p.SPEC:raise RuntimeError('Amendment protocol drift')
   p.atomic_json(proto,p.SPEC)
   for d in (project/'runs').iterdir():
    if d.name in (p.VERSION,p.PARENT_VERSION):continue
    m=re.search(r'(?:seed_?)(\d+)$',d.name)
    if m and int(m[1]) in p.ALL_SEEDS:raise RuntimeError('Reserved seed reused outside campaign: '+str(d))
   # Hash cache at preparation, rather than trusting names/stat alone.
   data=ms.verify_cache(project/'data_s1/fineweb_edu_3b',root)
   if data['hashes']!=p.PARENT_EVIDENCE['data_hashes']:raise RuntimeError('New/old data mismatch')
   p.atomic_json(root/'inherited_evidence_snapshot.json',p.PARENT_EVIDENCE)
  print('[bracketing prepared; old campaign untouched]',root,flush=True);return
 if None in (x.stage,x.seed,x.method,x.lr):a.error('Missing trial arguments')
 if x.stage=='development':
  if x.seed not in p.DEV_SEEDS or x.lr not in p.NEW_LRS:raise RuntimeError('Unregistered amendment trial')
 else:
  if x.seed not in p.REPORT_SEEDS:raise RuntimeError('Unregistered reporting seed')
  sel=choose(project)
  if sel['state']!='locked' or sel['selection']['chosen_lr'][x.method]!=x.lr:raise RuntimeError('Reporting recipe not locked')
 d=p.trial_dir(project,x.stage,x.seed,x.method,x.lr);d.mkdir(parents=True,exist_ok=True)
 with (d/'.writer.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  data=json.loads((root/'cache_identity.json').read_text());ms.quick_cache_guard(data)
  if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('Production requires CUDA BF16')
  device=torch.device('cuda:0');cfg=ms.make_config('124m',data['path'],lr=x.lr)
  from dataclasses import asdict
  cc=asdict(cfg);cc.pop('learning_rate')
  if cc!=p.PARENT_EVIDENCE['config_template']:raise RuntimeError('Non-LR configuration differs from v1')
  # Own-account/new-cache execution would be a separate experiment, not this amendment.
  expected=p.PARENT_EVIDENCE['runtime_by_seed'][str(x.seed)] if x.stage=='development' else p.PARENT_EVIDENCE['runtime_by_seed']['127']
  if ms.runtime_identity(device)!=expected:raise RuntimeError('Runtime differs from inherited campaign; audit before continuation')
  import engine
  result=engine.train(project,x.stage,x.seed,x.method,x.lr,data,device)
  if x.stage=='development':
   initial=json.loads((d/'initialization.json').read_text())
   if initial!=p.PARENT_EVIDENCE['initializations'][str(x.seed)]:raise RuntimeError('New vs old initial model/RNG mismatch')
if __name__=='__main__':
 try:main()
 except Exception as e:print('[STOPPED]',type(e).__name__+':',e,flush=True);raise
