#!/usr/bin/env python3
"""CPU tests use only temporary synthetic files/models; no experiment data."""
from pathlib import Path
import argparse,copy,json,tempfile
import protocol as p

def selection_tests():
 import select_recipes as s
 original_parent,original_new=s.verify_parent,s.read_new_grid
 def build(root,targets):
  rows=[]
  for seed in p.DEV_SEEDS:
   for method in p.METHODS:
    for lr in p.SEARCH_LRS:
     value=4.+((lr-targets[method])*1000)**2+(seed-127)*.002
     f=Path(root)/'fixture'/str(seed)/method/p.lr_tag(lr)/'result.json'
     p.atomic_json(f,dict(test_fixture=True,dev_loss=value))
     rows.append(dict(seed=seed,method=method,lr=lr,dev_loss=value,result=str(f),sha256=p.file_sha(f)))
  return rows
 try:
  with tempfile.TemporaryDirectory(prefix='bracket_selector_') as tmp:
   rows=build(tmp,dict(adamw=.0012,temper_cap3=.0010,gs_window=.0016))
   prior=[r for r in rows if r['lr'] in p.INHERITED_LRS];new=[r for r in rows if r['lr'] in p.NEW_LRS]
   s.verify_parent=lambda project:prior
   s.read_new_grid=lambda project:dict(ready=False,missing=['pending'],rows=new[:-1])
   assert s.choose(tmp)['state']=='waiting_bracketing'
   assert not (p.campaign(tmp)/'selection.lock.json').exists()
   s.read_new_grid=lambda project:dict(ready=True,missing=[],rows=new)
   state=s.choose(tmp);assert state['state']=='locked'
   assert state['selection']['chosen_lr']==dict(adamw=.0012,temper_cap3=.0010,gs_window=.0016)
   assert len(state['selection']['development_results'])==60
   assert s.choose(tmp)==state
   Path(rows[0]['result']).write_text('mutated')
   try:s.choose(tmp)
   except RuntimeError as e:assert 'evidence changed' in str(e)
   else:raise AssertionError('Changed selection evidence accepted')
  with tempfile.TemporaryDirectory(prefix='bracket_edge_') as tmp:
   rows=build(tmp,{m:.004 for m in p.METHODS})
   s.verify_parent=lambda project:[r for r in rows if r['lr'] in p.INHERITED_LRS]
   s.read_new_grid=lambda project:dict(ready=True,missing=[],rows=[r for r in rows if r['lr'] in p.NEW_LRS])
   assert s.choose(tmp)['state']=='unbracketed'
   assert not (p.campaign(tmp)/'selection.lock.json').exists()
   # Even an interior winner must not select after unauthorized report access.
   p.atomic_json(p.campaign(tmp)/'report/stray.json',{'test_fixture':True})
   try:s.choose(tmp)
   except RuntimeError as e:assert 'Reporting files' in str(e)
   else:raise AssertionError('Open report partition accepted')
  with tempfile.TemporaryDirectory(prefix='bracket_divergence_') as tmp:
   rows=build(tmp,{m:.0016 for m in p.METHODS})
   for r in rows:
    if r['lr']==.0024:r['dev_loss']=None
   assert all(v==.0016 for v in s.summarize(rows)['chosen_lr'].values())
  # Grid allocation is identical across all methods, no method-specific extension.
  assert len(p.NEW_LRS)==3 and len(p.SEARCH_LRS)==10 and not p.EDGE_LRS
  print('[PASS] pending barrier; independent recipe selection; immutable lock; edge stop; report-access refusal; divergent recipe invalidation')
 finally:s.verify_parent,s.read_new_grid=original_parent,original_new

def evidence_parser_tests():
 import select_recipes as s
 with tempfile.TemporaryDirectory(prefix='bracket_new_evidence_') as tmp:
  source={q.relative_to(p.HERE).as_posix():p.file_sha(q) for q in sorted(p.HERE.rglob('*.py')) if '__pycache__' not in q.parts}
  for seed in p.DEV_SEEDS:
   for method in p.METHODS:
    for lr in p.NEW_LRS:
     d=p.trial_dir(tmp,'development',seed,method,lr)
     cfg=copy.deepcopy(p.PARENT_EVIDENCE['config_template']);cfg['learning_rate']=lr
     ct=dict(version=p.VERSION,protocol_sha=p.PROTOCOL_SHA,stage='development',seed=seed,method=method,lr=lr,
         source_sha256=source,data_hashes=p.PARENT_EVIDENCE['data_hashes'],runtime=p.PARENT_EVIDENCE['runtime_by_seed'][str(seed)],config=cfg)
     ini=p.PARENT_EVIDENCE['initializations'][str(seed)]
     p.atomic_json(d/'contract.json',ct);p.atomic_json(d/'initialization.json',ini)
     p.atomic_json(d/'result.json',dict(contract=ct,contract_sha256=p.digest(ct),state='complete',step=3000,report=None,initialization=ini,
         dev=dict(loss=4.,partition_start=0,partition_stop=2048*513,tokens=2048*512)))
  assert s.read_new_grid(tmp)['ready']
  q=p.trial_dir(tmp,'development',127,'adamw',.001)/'result.json';r=json.loads(q.read_text());r['report']={'loss':3.9};p.atomic_json(q,r)
  try:s.read_new_grid(tmp)
  except RuntimeError as e:assert 'reporting-contaminated' in str(e)
  else:raise AssertionError('Development report leakage not caught')
 print('[PASS] actual amendment result parser rejects reporting contamination')

def main():
 a=argparse.ArgumentParser();a.add_argument('--cuda',action='store_true');x=a.parse_args()
 p.verify_sources()
 import queue
 assert hasattr(queue,'Queue')
 import torch
 import backend_tests as b
 torch.set_num_threads(4 if x.cuda else 2)
 if x.cuda and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):raise RuntimeError('CUDA BF16 gate requires target GPU')
 device=torch.device('cuda:0' if x.cuda else 'cpu')
 b.math_gate(device);b.identity_gate(device);b.workflow_gate(device)
 selection_tests();evidence_parser_tests()
 print('LR BRACKETING CUDA GATE: PASS' if x.cuda else 'LR BRACKETING CPU TESTS: PASS')
if __name__=='__main__':main()
