#!/usr/bin/env python3
"""CPU fixture tests + actual frozen-backend integration. No efficacy outputs."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,copy,json,shutil,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
import gct_protocol as p
import gct_io as io
import gct_train as tr
from gct_policy import Policy
from gct_runtime import configure,cpu,rng,restore_rng,treehash
from gct_analysis import compare,analyze
import study
PROJECT=None
MODS=None

class Tests(unittest.TestCase):
 def models(self):
  cfg=tr.config(MODS['native'],'unused',False,True);m=tr.build(MODS['native'],cfg,127);opt=Policy(m,cfg,MODS,'adamw',torch.device('cpu'),127)
  # A few deterministic nonzero gradients populate all named Adam states.
  for step in range(1,4):
   for k,t in enumerate(m.parameters()):t.grad=torch.full_like(t,.003*(k+1))
   opt.step(.0001,step)
  return cfg,m,opt
 def test_protocol_and_budget(self):
  self.assertEqual(p.SPEC['scientific_optimizer_updates'],32460)
  self.assertEqual(p.SEEDS,(159,160));self.assertFalse(p.SPEC['report_access']);self.assertFalse(p.SPEC['automatic_extensions'])
  self.assertEqual(7630*32768,250019840)
 def test_growth_age_mapping_nonaliasing(self):
  cfg,m,opt=self.models();st=cpu(opt.named_adam());st['age']=7630
  for r in st['states'].values():
   if r['kind']=='rest':r['state']['step'].fill_(7630)
  opt.load_adam(st);opt.phase=7630
  cfg2=tr.config(MODS['native'],'unused',True,True);grown=tr.build(MODS['native'],cfg2,127)
  go=Policy(grown,cfg2,MODS,'adamw',torch.device('cpu'),127,True,7630)
  checks=tr.copy_growth(m,opt,grown,go)
  self.assertEqual(checks['grown_age'],7630);self.assertEqual(checks['next_global_data_step'],7631)
  self.assertEqual(checks['covariance_depth_dimension'],3)
  self.assertTrue(checks['independent_moment_storage']);self.assertTrue(checks['both_covariance_histories_zero'])
  for k,t in enumerate(grown.parameters()):t.grad=torch.full_like(t,.01)
  go.step(.000003,1);self.assertEqual(go.base.step_num,7631);self.assertEqual(go.canonical['gate'].last_step,1)
  for n in go.rest_names:self.assertEqual(int(go.rest.state[go.byname[n]]['step']),7631)
 def test_missing_and_bad_state_refusal(self):
  cfg,m,opt=self.models();good=cpu(opt.named_adam())
  n=opt.rest_names[0];bad=copy.deepcopy(good);bad['states'][n]['state']={}
  with self.assertRaises(RuntimeError):opt.load_adam(bad)
  bad=copy.deepcopy(good);q=next(n for n,r in bad['states'].items() if r['kind']=='projection');bad['states'][q]['m'].flatten()[0]=float('nan')
  with self.assertRaises(RuntimeError):opt.load_adam(bad)
  bad=copy.deepcopy(good);bad['states'][q]['v'].flatten()[0]=-1
  with self.assertRaises(RuntimeError):opt.load_adam(bad)
 def test_phase_reset_refusal(self):
  cfg,m,opt=self.models()
  with self.assertRaises(RuntimeError):opt.step(.01,1)
 def test_geometry_rng_transport_and_degeneracy(self):
  geo=MODS['geometry'];before=treehash(rng());per=geo.make_permutations(4,7,127);self.assertEqual(before,treehash(rng()))
  u=torch.randn(4,7,5);self.assertTrue(torch.equal(u,geo.transport(geo.transport(u,per,'gate'),per,'gate',inverse=True)))
  for q in (u,torch.zeros_like(u)):
   v,info=geo.match_direction(q,q,q);self.assertTrue(torch.equal(v,q))
  a=u-u.mean(0,keepdim=True);d=torch.randn_like(a);d-=d.mean(0,keepdim=True);d-=(d*a).sum()/a.square().sum()*a
  d*=torch.linalg.vector_norm(a)/torch.linalg.vector_norm(d);v=u.mean(0,keepdim=True)+a*.99995+d*.009999875
  with self.assertRaises(geo.DegenerateDirection):geo.match_direction(u,v,u)
 def test_pinned_bdt_candidate(self):
  cfg=tr.config(MODS['native'],'unused',True,True);m=tr.build(MODS['native'],cfg,127);opt=Policy(m,cfg,MODS,'bdt',torch.device('cpu'),127,True,0)
  ref={q:MODS['temper'].OnlineDepthTemper(4,'cpu',torch.float32,MODS['temper'].TemperConfig(200,200,10,.9,3.)) for q in ('gate','up','down')}
  for step in range(1,421):
   for q in ref:
    u=torch.randn(4,64,48) if q!='down' else torch.randn(4,48,64)
    a,_=opt.canonical[q].transform(u,step);b,_=ref[q].transform(u,step);self.assertTrue(torch.equal(a,b))
 def test_seed_conflict_and_incomplete_analysis(self):
  with tempfile.TemporaryDirectory() as d:
   z=Path(d)/'runs/other/seed_159';z.mkdir(parents=True)
   with self.assertRaises(RuntimeError):study.seeds_guard(d)
  # An entirely empty pilot must never be labeled complete.
  self.assertEqual(analyze(PROJECT,write=False)['state'],'incomplete')
 def test_scientific_job_refusals(self):
  with self.assertRaises(RuntimeError):tr.validate_scientific_job(159,'parent','bdt',7630,False,False,None,None)
  with self.assertRaises(RuntimeError):tr.validate_scientific_job(159,'arm','adamw',6000,False,False,'x',None)
  with self.assertRaises(RuntimeError):tr.validate_scientific_job(161,'parent','adamw',7630,False,False,None,None)
  with self.assertRaises(RuntimeError):tr.validate_scientific_job(159,'arm','adamw',3000,False,True,'x',None)
 def test_checkpoint_corruption_refusal(self):
  with tempfile.TemporaryDirectory() as d:
   meta=io.save_ck(d,dict(step=3,contract={'fixture':True},x=torch.arange(5)),torch,{'fixture':True})
   ck,_=io.load_ck(d,torch);self.assertEqual(ck['step'],3)
   with (Path(d)/meta['path']).open('ab') as f:f.write(b'corruption')
   with self.assertRaises(RuntimeError):io.load_ck(d,torch)
 def test_checkpoint_retention_and_identity_guard(self):
  with tempfile.TemporaryDirectory() as d:
   a=io.save_ck(d,dict(step=1,contract={}),torch,{},keep=True)
   b=io.save_ck(d,dict(step=2,contract={}),torch,{})
   io.save_ck(d,dict(step=3,contract={}),torch,{})
   self.assertTrue((Path(d)/a['path']).exists());self.assertFalse((Path(d)/b['path']).exists())
 def test_growth_mapping_names(self):
  self.assertEqual(tr.source_name('blocks.11.ffn.up.weight',6),'blocks.5.ffn.up.weight')
  self.assertEqual(tr.source_name('token_embedding.weight',6),'token_embedding.weight')

 def test_gate_orchestration_and_evidence_guard(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as td:
   td=Path(td);_,deps=io.verify(PROJECT)
   for key,path in deps.items():
    dest=td/p.DEPENDENCIES[key][0];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
   calls=[]
   def fake_job(project,name,values,target,resume=False):
    calls.append((name,dict(values),resume));d=Path(values['directory']);d.mkdir(parents=True,exist_ok=True)
    io.atomic(d/'latest.json',{'mock':True})
    if values.get('interrupt'):return
    row=dict(state='complete',initial={},proof=[dict(step=201,gradient='g',raw='u',batch='b',rng_before='r',rng_after='r')],final=dict(model='m',adam='a'),dev={'loss':1.},checks={'mock_orchestration_only':True})
    io.atomic(d/'result.json',row)
   with patch.object(study,'job',fake_job):
    got=study.gate(td,'GPU-CPU-FIXTURE')
    self.assertEqual(got['state'],'passed');self.assertEqual(len(calls),13)
    interrupt=next(x for x in calls if x[1].get('interrupt'));self.assertEqual(interrupt[1]['interrupt'],310)
    resume=next(x for x in calls if x[0].endswith('_scrambled_matched_1'));self.assertTrue(resume[2])
    study.gate(td,'GPU-CPU-FIXTURE');self.assertEqual(len(calls),13)
    f=p.root(td)/'gates/GPU-CPU-FIXTURE/bdt_0/result.json';bad=io.read(f);bad['dev']['loss']=2.;io.atomic(f,bad)
    with self.assertRaises(RuntimeError):study.gate(td,'GPU-CPU-FIXTURE')
 def test_complete_pilot_analysis_and_invalid_records(self):
  with tempfile.TemporaryDirectory() as td:
   td=Path(td);source,deps=io.verify(PROJECT)
   for key,path in deps.items():
    dest=td/p.DEPENDENCIES[key][0];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
   for seed in p.SEEDS:
    d=p.root(td)/f'seed_{seed}';ctx=dict(seed=seed,source=source,protocol=p.PROTOCOL_SHA,config={'fixture':True},runtime={'fixture':True},cache=p.CACHE,offset=7630,gate=False,tiny=False)
    def base(stage,step):
     c=dict(ctx,stage=stage,mode='adamw');return dict(state='complete',contract=c,contract_sha=p.digest(c),step=step,checkpoint={'sha256':stage},initial={'phase':0},final={'phase':step,'offset':7630},timing={},dev={'loss':4.})
    parent=base('parent',7630);grown=base('grown',0);grown['contract']['parent_sha']='parent';grown['contract_sha']=p.digest(grown['contract']);grown.update(checks={'fixture':True},grown_dev={'loss':4.2},growth_loss_jump=.2)
    prefix=base('prefix',200);prefix['contract']['parent_sha']='grown';prefix['contract_sha']=p.digest(prefix['contract']);prefix['initial']=grown['final']
    for name,row in [('parent',parent),('grown',grown),('prefix',prefix)]:io.atomic(d/name/'result.json',row)
    history=[dict(step=i,global_step=7630+i,batch='b',rng_before='r',rng_after='r') for i in range(201,3001)]
    for mode,loss in [('adamw',4.),('bdt',3.99),('scrambled_matched',4.01)]:
     row=base('arm',3000);row['contract'].update(mode=mode,parent_sha='prefix');row['contract_sha']=p.digest(row['contract']);row['initial']=prefix['final'];row.update(history=history,curve=[dict(step=201,gradient='g',raw='u',batch='b',rng_before='r',rng_after='r')],matching={'active_groups':8400},memory={});row['dev']=dict(loss=loss,offset=0,tokens=1048576,stop=1050624)
     io.atomic(d/'arms'/mode/'result.json',row);io.atomic(d/'arms'/mode/'latest.json',row['checkpoint'])
   result=analyze(td,write=False);self.assertEqual(result['state'],'complete');self.assertEqual(len(result['rows']),2)
   f=p.root(td)/'seed_159/arms/bdt/result.json';row=io.read(f);row['contract']['cache']={'wrong':'cache'};row['contract_sha']=p.digest(row['contract']);io.atomic(f,row)
   with self.assertRaises(RuntimeError):analyze(td,write=False)


def integration(project):
 global PROJECT,MODS
 with tempfile.TemporaryDirectory(prefix='gct1_cpu_integration_') as tmp:
  project=Path(tmp);_,dep=io.verify(PROJECT)
  for key,path in dep.items():dest=project/p.DEPENDENCIES[key][0];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
  # The dynamically loaded dependencies retain the same verified original paths.
  # Separate subprocess isolation is used in production; here re-use that exact backend.
  for name in ['cms_protocol']+['_gct1_frozen_'+k for k in p.DEPENDENCIES if k!='cms_protocol']:
   __import__('sys').modules.pop(name,None)
  d=project/'data_s1/fineweb_edu_3b';d.mkdir(parents=True);gen=np.random.default_rng(7001)
  for n,size in [('train.bin',50000),('val.bin',5000)]:gen.integers(0,97,size=size,dtype=np.uint16).tofile(d/n)
  (d/'metadata.json').write_text('{"fixture":true}\n')
  data=dict(path=str(d),label='SYNTHETIC-CPU-ONLY',hashes={n:io.sha(d/n) for n in p.CACHE},stat={n:io.stamp(d/n) for n in p.CACHE})
  root=p.root(project)/'gates/cpu_fixture'
  def run(name,stage,mode,end,parent=None,**kw):return tr.train(project,root/name,127,stage,mode,end,root/parent if parent else None,gate=True,tiny=True,data=data,**kw)
  run('parent','parent','adamw',200)
  gr=tr.grow(project,root/'grown',127,root/'parent',gate=True,tiny=True,data=data)
  assert gr['checks']['rng_preserved'] and gr['checks']['grown_age']==200
  run('prefix','prefix','adamw',200,'grown');run('reference_prefix','prefix','reference_adamw',200,'grown')
  checks={'prefix_vs_reference':compare(root/'prefix',root/'reference_prefix'),'growth':gr['checks']}
  for mode in p.ARMS:
   run(mode+'_0','arm',mode,420,'prefix')
   if mode=='scrambled_matched':
    paused=run(mode+'_1','arm',mode,420,'prefix',interrupt=310);assert paused['state']=='test_paused'
    run(mode+'_1','arm',mode,420,'prefix',resume=True)
   else:run(mode+'_1','arm',mode,420,'prefix')
   checks[mode+'_repeat_resume']=compare(root/(mode+'_0'),root/(mode+'_1'))
  run('reference_adamw','arm','reference_adamw',420,'prefix');run('reference_bdt','arm','reference_bdt',420,'prefix')
  checks['adamw_vs_frozen']=compare(root/'adamw_0',root/'reference_adamw');checks['bdt_vs_frozen']=compare(root/'bdt_0',root/'reference_bdt')
  first=[]
  for mode in p.ARMS:
   r=io.read(root/(mode+'_0')/'result.json');row=next(v for v in r['proof'] if v['step']==201);first.append({k:row[k] for k in ('gradient','raw','batch','rng_before','rng_after')})
   assert r['matching']['active_groups']==660 and r['final']['offset']==200 and r['final']['phase']==420
   assert r['history'][0]['global_step']==401 and r['history'][-1]['global_step']==620
  assert all(x==first[0] for x in first);checks['first_active_inputs_equal']=True
  # This negative check must not be satisfied by merely equal endpoints.
  f=root/'bdt_1/result.json';r=io.read(f);r['proof'][0]['loss']+=.01;io.atomic(f,r)
  try:compare(root/'bdt_0',root/'bdt_1')
  except RuntimeError:checks['changed_trace_refused']=True
  else:raise AssertionError('Changed trace accepted')
  return checks

def main():
 global PROJECT,MODS
 ap=argparse.ArgumentParser();ap.add_argument('--project',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();PROJECT=Path(a.project).resolve();configure(1)
 source,_=io.verify(PROJECT);MODS=io.backend(PROJECT)
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);r=unittest.TextTestRunner(verbosity=2).run(suite)
 if not r.wasSuccessful():raise SystemExit(1)
 checks=integration(PROJECT)
 result=dict(passed=True,unit_tests=r.testsRun,integration=checks,source=source,protocol=p.PROTOCOL_SHA,torch=torch.__version__,time=io.now(),
             scope='CPU tiny geometry with exact frozen dependencies, real phase200/420, synthetic large-age7630 check. NOT a full-size CUDA pass.')
 io.atomic(a.output,result);print('GCT1 CPU TESTS: PASS',json.dumps(result,sort_keys=True),flush=True)
if __name__=='__main__':main()
