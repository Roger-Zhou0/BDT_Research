"""Prospective source, design, refusal, and actual-backend repeat/resume tests."""
import os,sys,json,copy,tempfile,unittest,math
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common as io
import torch,numpy as np
import experiment
tr,rt,mods=io.engine(io.HERE.parent);rt.configure(1)

class Tests(unittest.TestCase):
 def test_integrity(self):self.assertEqual(len(io.verify()[0]),64)
 def test_jobgraph(self):
  jobs=experiment.science_jobs('/tmp/p');n=len(io.SPEC['seeds']);fresh=io.SPEC['kind']=='ancestry_confirmation'
  self.assertEqual(len(jobs),n*(7 if fresh else 6))
  self.assertEqual(sum(j['stage']=='parent' for j in jobs),n if fresh else 0)
  self.assertEqual(sum(j['stage']=='arm' for j in jobs),4*n)
 def test_budget(self):
  self.assertEqual(len(io.SPEC['seeds'])*(200+4*(io.SPEC['post_updates']-200)),io.SPEC['work_updates_12layer'])
 def test_gate_separate(self):self.assertNotIn(127,io.SPEC['seeds'])
 def test_orders(self):
  for s in io.SPEC['seeds']:self.assertEqual(sorted(io.SPEC['arm_order'][str(s)]),sorted(io.SPEC['arms']))
 def test_lanes(self):self.assertEqual(sorted(s for row in io.SPEC['lane_seeds'].values() for s in row),sorted(io.SPEC['seeds']))
 def test_explicit_release_default(self):
  import inspect,operations
  self.assertFalse(inspect.signature(operations.coordinate).parameters['science'].default)
 def test_immutable(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'x';io.immutable(f,{'a':1});io.immutable(f,{'a':1})
   with self.assertRaises(RuntimeError):io.immutable(f,{'a':2})
 def test_lock(self):
  with tempfile.TemporaryDirectory() as d:
   with io.lock(Path(d)/'x'):
    with self.assertRaises(RuntimeError):
     with io.lock(Path(d)/'x'):pass
 def test_path(self):
  with self.assertRaises(RuntimeError):io.safe_file('/tmp/a','../x')
 def test_nan_write(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):io.atomic(Path(d)/'x',{'x':float('nan')})
 def test_corrupt_checkpoint(self):
  with tempfile.TemporaryDirectory() as d:
   m=io.save_ck(d,{'step':0,'contract':{},'x':torch.ones(2)},torch,{'x':1})
   (Path(d)/m['path']).write_bytes(b'not-a-checkpoint')
   with self.assertRaises(RuntimeError):io.load_ck(d,torch)
 def test_predecessor_absent(self):
  from operations import prerequisites
  with tempfile.TemporaryDirectory() as d:self.assertFalse(prerequisites(d)[0])
 def test_primary_statistics(self):
  from paired_analysis import six_parent_contrast
  rows=[dict(seed=i,a=1.+i*.001,b=1.) for i in range(1,7)]
  v=six_parent_contrast(rows,list(range(1,7)),'a','b');self.assertAlmostEqual(v['mean'],.0035)
  with self.assertRaises(ValueError):six_parent_contrast(rows[:-1],list(range(1,7)),'a','b')
 def test_unreached_quality(self):
  from paired_analysis import first_crossing_bracket
  v=first_crossing_bracket([{'step':500,'loss':4.,'training_seconds':1.}],3.)
  self.assertFalse(v['reached'])
 def test_partial_report_refusal(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises((RuntimeError,FileNotFoundError)):experiment.evaluate_report(d,175,'adamw')
 def test_fresh_labels_not_training_results(self):
  if io.SPEC['kind']=='ancestry_confirmation':self.assertEqual(io.SPEC['seeds'],list(range(175,181)))
  else:self.assertEqual(io.SPEC['fresh_seeds'],[])
 def test_own_training_rng_unchanged_by_charts(self):
  from ancestry import make_family
  before=torch.random.get_rng_state().clone();x=make_family(6,16,175,0)
  self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
  self.assertEqual(x['preserved']['audit']['chart_multiset_sha256'],x['reassigned']['audit']['chart_multiset_sha256'])
 def test_schedule_boundaries(self):
  if io.SPEC['kind']!='window_pilot':return
  from window_schedule import strength,collect_history,last_history_step
  self.assertEqual(strength('early_bdt',3000),1.);self.assertEqual(strength('early_bdt',3001),0.)
  self.assertEqual(strength('late_bdt',6200),0.);self.assertEqual(strength('late_bdt',6201),.005)
  self.assertEqual(strength('late_bdt',6400),1.);self.assertEqual(strength('late_bdt',9001),0.)
  self.assertFalse(collect_history('early_bdt',3001));self.assertTrue(collect_history('late_bdt',6200))
  self.assertEqual(last_history_step('adamw',15000),200)
 def test_equal_active_windows(self):
  if io.SPEC['kind']!='window_pilot':return
  from window_schedule import strength
  for m in ('early_bdt','late_bdt'):self.assertEqual(sum(strength(m,s)>0 for s in range(1,15001)),2800)
 def test_synthetic_boundary_reference(self):
  if io.SPEC['kind']!='window_pilot':return
  from boundary_tests import run
  self.assertEqual(run('cpu')['state'],'passed')


 def test_engineering_coordinator_without_scientific_release(self):
  import contextlib,types
  from unittest.mock import patch
  import operations,followup_queue as q
  with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
   project=Path(tmp);lane=next(iter(io.SPEC['lane_seeds']));root=io.root(project)
   io.atomic(root/'lanes'/lane/'cpu_validation.json',{'state':'passed','source':io.verify()[0]})
   def boundary(cmd,**kwargs):
    if '--output' in cmd:io.atomic(Path(cmd[cmd.index('--output')+1]),{'state':'passed','source':io.verify()[0]})
    return types.SimpleNamespace(returncode=0)
   mocks={}
   for obj,name,value in [(operations,'prerequisites',(True,'test')),(operations,'register',{}),(q,'session',{}),(q,'lane',lane),(q,'other_work',[]),(q,'gpu',{'uuid':'GPU-test'}),(q,'memory_wait',None),(q,'runtime_probe',{'device':'test'}),(q,'gate',None),(io,'cache',{'path':'test','hashes':{},'stat':{}}),(operations.shutil,'disk_usage',types.SimpleNamespace(free=10**15))]:
    mocks[name]=stack.enter_context(patch.object(obj,name,return_value=value))
   stack.enter_context(patch.object(operations.subprocess,'run',side_effect=boundary))
   operations.coordinate(project,False,science=False)
   self.assertEqual(io.read(root/'lanes'/lane/'status.json')['state'],'engineering_complete_not_released')
   mocks['cache'].assert_called_once_with(project);mocks['gate'].assert_called_once()
   self.assertFalse(list(root.glob('seed_*')))

 def test_report_slice_matches_reference_at_offset_zero(self):
  from report_slice import evaluate_slice
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'val.bin';np.random.default_rng(12).integers(0,96,2000,dtype=np.uint16).tofile(path)
   cfg=tr.config(mods['native'],temp,True,True);model=tr.build(mods['native'],cfg,127)
   expected=rt.evaluate(model,path,cfg,mods['native'],torch.device('cpu'),True)
   actual=evaluate_slice(model,path,cfg,mods['native'],torch.device('cpu'),0,4)
   self.assertEqual(actual,expected)

def integration():
 with tempfile.TemporaryDirectory(prefix=io.SPEC['short']+'_integration_') as temp:
  project=Path(temp);dat=project/'data';dat.mkdir();gen=np.random.default_rng(127)
  for f,n in [('train.bin',80000),('val.bin',2000)]:gen.integers(0,96,n,dtype=np.uint16).tofile(dat/f)
  data={'path':str(dat),'hashes':{f:io.sha(dat/f) for f in ('train.bin','val.bin')},'stat':{f:io.stamp(dat/f) for f in ('train.bin','val.bin')}}
  d=io.root(project)/'gates/cpu'
  for j in experiment.gate_plan(d,True):
   print('ACTUAL_CPU_JOB',Path(j['directory']).name,'interrupt',j.get('interrupt'),flush=True)
   experiment.execute(project,j,data,True)
  checks=experiment.compare_gate(d)
  out=dict(state='passed',source=io.verify()[0],checks=checks,device='cpu',torch=torch.__version__,numpy=np.__version__,scope='Actual frozen-backend tiny-model reference/repeat/interruption/resume; not full-size CUDA or language-model efficacy',time=io.now())
  io.atomic(Path(os.environ.get('FOLLOWUP_TEST_OUTPUT',str(io.HERE.parent/(io.SPEC['short']+'_cpu_validation.json')))),out)
  print('ACTUAL_BACKEND_CPU_INTEGRATION_PASS',json.dumps(out),flush=True)
if __name__=='__main__':
 res=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 if not res.wasSuccessful():raise SystemExit(1)
 if '--unit-only' not in sys.argv:integration()
