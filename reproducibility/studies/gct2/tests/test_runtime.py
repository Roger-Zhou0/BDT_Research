"""Operational fixtures: no user directories, real training or GPU timing."""
import ast,contextlib,copy,hashlib,io,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core,study,gct2
class RuntimeTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.p=Path(self.temp.name);(self.p/'runs').mkdir();(self.p/'logs').mkdir()
  study.STOP=False;study.TARGET_UUID=None
 def tearDown(self):self.temp.cleanup()
 def fixture(self,end=3000):
  d=core.root(self.p)/'fixture';d.mkdir(parents=True)
  f=d/'checkpoints/f.pt';f.parent.mkdir();f.write_bytes(b'fixture')
  ct=dict(source=core.verify()[0],protocol=core.compact(core.GCT2),seed=161,stage='arm',mode='bdt',end=end,gate=False,tiny=False,runtime={})
  m=dict(path='checkpoints/f.pt',sha256=core.sha(f),step=end,contract_sha=core.digest(ct),identity={'fixture':1})
  r=dict(state='complete',contract=ct,contract_sha=core.digest(ct),step=end,final=m['identity'],checkpoint=m)
  core.atomic(d/'contract.json',ct);core.atomic(d/'latest.json',m);core.atomic(d/'result.json',r)
  return d,ct,r
 def test_bundled_sources(self):self.assertGreater(len(core.verify()[1]),3)
 def test_backend_exact_native(self):self.assertEqual(core.sha(core.HERE/'backend/validation_v6_1.py'),'ee16f65a5b948b0e645131a01ddc437edb5806e671c44c80792b40efecc9dd8b')
 def test_backend_exact_temper(self):self.assertEqual(core.sha(core.HERE/'backend/tempering_reference.py'),'d1bd72d9ccd18019aea83551b838d5915f727a0a7b6a0a2fd1006fb1c47969f1')
 def test_metadata_adapter_only(self):
  def strip(t):
   for n in ast.walk(t):
    if isinstance(n,ast.Constant) and isinstance(n.value,str):n.value='<str>'
   return ast.dump(t,include_attributes=False)
  self.assertEqual(strip(ast.parse((core.HERE/'gct_train_adapter.py').read_text())),strip(ast.parse((core.HERE/'frozen/gct1/gct_train.py').read_text())))
 def test_instrumentation_no_expression_change(self):
  class Strip(ast.NodeTransformer):
   def visit_ImportFrom(self,n):return None if n.module=='torch.profiler' else n
   def visit_With(self,n):
    self.generic_visit(n)
    e=n.items[0].context_expr
    return n.body if isinstance(e,ast.Call) and isinstance(e.func,ast.Name) and e.func.id=='_bco_region' else n
  actual=Strip().visit(ast.parse((core.HERE/'candidate/profiled_reference.py').read_text()))
  self.assertEqual(ast.dump(actual,include_attributes=False),ast.dump(ast.parse((core.HERE/'backend/tempering_reference.py').read_text()),include_attributes=False))
 def test_gate_key_uuid(self):self.assertNotEqual(study.gate_key('B-default','GPU-a',{}),study.gate_key('B-default','GPU-b',{}))
 def test_gate_key_runtime(self):self.assertNotEqual(study.gate_key('B-good','GPU-a',{}),study.gate_key('B-good','GPU-a',{'version':'changed'}))
 def test_gate_independent_registration_names(self):
  for gid in ('GPU-a','GPU-b'):
   k=study.gate_key('B-default',gid,{})
   core.immutable(core.root(self.p)/'jobs'/'gate'/k/'parent.json',dict(path=gid))
  self.assertEqual(len(list(core.root(self.p).glob('jobs/gate/*/parent.json'))),2)
 def test_immutable_refusal(self):
  f=self.p/'x.json';core.immutable(f,{'a':1})
  with self.assertRaises(RuntimeError):core.immutable(f,{'a':2})
 def test_immutable_idempotent(self):
  f=self.p/'x.json';core.immutable(f,{'a':1});h=core.sha(f);core.immutable(f,{'a':1});self.assertEqual(core.sha(f),h)
 def test_completed_valid(self):d,ct,r=self.fixture();self.assertEqual(core.verify_completed(d,ct),r)
 def test_corrupt_result_contract(self):
  d,ct,r=self.fixture();r['contract']['end']=4000;core.atomic(d/'result.json',r)
  with self.assertRaises(RuntimeError):core.verify_completed(d)
 def test_corrupt_payload_refused(self):
  d,ct,r=self.fixture();(d/'checkpoints/f.pt').write_bytes(b'changed')
  with self.assertRaises(RuntimeError):core.verify_completed(d)
 def test_missing_payload_refused(self):
  d,ct,r=self.fixture();(d/'checkpoints/f.pt').unlink()
  with self.assertRaises(RuntimeError):core.verify_completed(d)
 def test_fatal_completed_refused(self):
  d,ct,r=self.fixture();core.atomic(d/'fatal.json',{})
  with self.assertRaises(RuntimeError):core.verify_completed(d)
 def test_wrong_seed_skip_refused(self):
  d,ct,r=self.fixture()
  with self.assertRaises(RuntimeError):core.verify_completed(d,{'seed':162})
 def test_completed_skip_no_child_no_log(self):
  d,ct,r=self.fixture();j=dict(directory=str(d),seed=161,stage='arm',mode='bdt',end=3000,gate=False)
  with patch.object(study.subprocess,'Popen',side_effect=AssertionError('No child permitted')):
   with contextlib.redirect_stdout(io.StringIO()):got=study.run_process(self.p,'gct2','B-good','fixture','gct_job',j,'fixture',{},self.p/'data.json',True)
  self.assertEqual(got,r);self.assertEqual(list((self.p/'logs').rglob('*.log')),[])
 def test_interruption_directive_not_job_identity(self):
  d=core.root(self.p)/'interrupted';core.atomic(d/'latest.json',{'step':310})
  j=dict(directory=str(d),seed=127,stage='arm',mode='bdt',end=420,gate=True,interrupt=310)
  with contextlib.redirect_stdout(io.StringIO()):study.run_process(self.p,'gct2','B-good','pause','gct_job',j,'gate/fake',{},self.p/'data.json',True)
  saved=core.read(core.root(self.p)/'jobs/gate/fake/job.json')
  self.assertNotIn('interrupt',saved['job']);self.assertEqual(list((self.p/'logs').rglob('*.log')),[])
 def test_three_oom_budget_persists(self):
  base=core.root(self.p)/'jobs'/'blocked';core.atomic(base/'attempts.json',{'oom_failures':3,'attempts':[]})
  with patch.object(study.subprocess,'Popen',side_effect=AssertionError('No child permitted')):
   with self.assertRaisesRegex(RuntimeError,'OOM limit'):study.run_process(self.p,'gct2','B-good','blocked','profile',{},'blocked',{},self.p/'data.json',True)
 def test_fatal_ledger_blocks(self):
  base=core.root(self.p)/'jobs'/'blocked';core.atomic(base/'attempts.json',{'oom_failures':0,'attempts':[],'fatal':True})
  with self.assertRaisesRegex(RuntimeError,'Fatal job'):study.run_process(self.p,'gct2','B-good','blocked','profile',{},'blocked',{},self.p/'data.json',True)
 def test_prior_profile_evidence_tamper(self):
  f=self.p/'profile.json';b=self.p/'summary.json';core.atomic(b,{});core.atomic(f,dict(source=core.verify()[0],runtime={},evidence={'summary.json':core.sha(b)}));core.atomic(b,{'bad':1})
  with self.assertRaises(RuntimeError):study.run_process(self.p,'gct2','B-good','profile','profile',{},'profile',{},self.p/'data.json',True,output_result=f)
 def test_lock_excludes_second_holder(self):
  with core.lock(self.p/'l'):
   with self.assertRaises(RuntimeError):
    with core.lock(self.p/'l'):pass
 def test_finalizer_lock_released(self):
  with core.lock(self.p/'final'):pass
  with core.lock(self.p/'final'):pass
 def test_no_report_before_seal(self):
  with self.assertRaises(FileNotFoundError):gct2.report_one(self.p,161,'bdt',{})
 def test_unsafe_path(self):
  with self.assertRaises(RuntimeError):core.safe_file(self.p,'../escape')
 def test_symlink_checkpoint(self):
  f=self.p/'inside';f.symlink_to(self.p.parent/'outside')
  with self.assertRaises(RuntimeError):core.safe_file(self.p,'inside')
 def test_right_sessions(self):
  with patch('getpass.getuser',return_value='account_b'),patch('socket.gethostname',return_value='jupyter-account_b--good'):self.assertEqual(study.target('gct2'),'B-good')
  with patch('getpass.getuser',return_value='account_b'),patch('socket.gethostname',return_value='jupyter-account_b'):self.assertEqual(study.target('gct2'),'B-default')
 def test_wrong_bco_session_refused(self):
  with patch('getpass.getuser',return_value='account_a'),patch('socket.gethostname',return_value='jupyter-account_a'):
   with self.assertRaises(RuntimeError):study.target('bco1')
 def test_gpu_drift_refused(self):
  with patch.object(study,'gpu',return_value={'uuid':'GPU-other','free_mib':99999}):
   with self.assertRaises(RuntimeError):study.memory_wait('GPU-target')
 def test_gate_proof_tamper(self):
  f=self.p/'proof.json';core.atomic(f,{});g=self.p/'gate.json';core.atomic(g,dict(state='passed',source=core.verify()[0],runtime={},evidence={'proof.json':core.sha(f)}));core.atomic(f,{'changed':1})
  with self.assertRaises(RuntimeError):study.verify_gate(self.p,g,{})
 def test_valid_gate_reuse(self):
  f=self.p/'proof.json';core.atomic(f,{});g=self.p/'gate.json';core.atomic(g,dict(state='passed',source=core.verify()[0],runtime={},evidence={'proof.json':core.sha(f)}));self.assertEqual(study.verify_gate(self.p,g,{})['state'],'passed')
 def test_cache_stat_change(self):
  f=self.p/'val.bin';f.write_bytes(b'123');d=dict(path=str(self.p),stat={'val.bin':core.stamp(f)});core.guard_cache(d);f.write_bytes(b'12345')
  with self.assertRaises(RuntimeError):core.guard_cache(d)
 def test_prerequisite_failure_no_stop_other(self):
  f=self.p/'runs/growth_scratch_reference_v1/scheduler_status.json';core.atomic(f,{'state':'failed'})
  with self.assertRaises(RuntimeError):study.prerequisite(self.p,'gct2','B-default')
 def test_prerequisite_complete(self):
  f=self.p/'runs/growth_scratch_reference_v1/scheduler_status.json';core.atomic(f,{'state':'finished'});study.prerequisite(self.p,'gct2','B-default')
 def test_manifest_protocol_exact(self):
  sys.path.insert(0,str(core.HERE/'design/tools'));from design import GCT2,BCO1,digest
  self.assertEqual(core.compact(core.GCT2),digest(GCT2));self.assertEqual(core.compact(core.BCO1),digest(BCO1))
class CoordinationTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.p=Path(self.temp.name);(self.p/'runs').mkdir();(self.p/'logs').mkdir()
 def tearDown(self):self.temp.cleanup()
 def gate_fixture(self,lane='B-good',name='first'):
  root=core.root(self.p);f=root/'gates'/name/'gate.json'
  core.atomic(f,dict(state='passed',source=core.verify()[0],runtime={},evidence={}))
  return f
 def test_first_lane_never_reports_early(self):
  f=self.gate_fixture()
  with patch.object(study,'gct_gate',return_value=f),patch.object(study,'run_process',return_value={}) as calls,patch.object(gct2,'seal_cohort',side_effect=AssertionError('Premature seal')):
   out=study.gct_coordinate(self.p,'B-good','GPU-one',{},self.p/'cache',False)
  self.assertEqual(out['state'],'training_complete_waiting_for_other_lane');self.assertEqual(calls.call_count,18)
  self.assertFalse((core.root(self.p)/'COHORT_LOCK.json').exists())
 def test_last_lane_seals_before_reporting(self):
  f=self.gate_fixture();root=core.root(self.p);core.atomic(root/'lanes/B-default/training_complete.json',{})
  events=[]
  def work(*args,**kwargs):events.append(args[4]);return {}
  def seal(*args):events.append('sealed');return dict(state='sealed')
  with patch.object(study,'gct_gate',return_value=f),patch.object(study,'run_process',side_effect=work),patch.object(gct2,'seal_cohort',side_effect=seal),patch.object(gct2,'analyze_reports',return_value={'joint_confirmation':True}):
   out=study.gct_coordinate(self.p,'B-good','GPU-one',{},self.p/'cache',False)
  self.assertEqual(events[:18],['gct_job']*18);self.assertEqual(events[18],'sealed');self.assertEqual(events[19:],['report']*18)
  self.assertEqual(out['state'],'finished')
 def test_completed_lane_migration_keeps_old_training_record(self):
  old=self.gate_fixture(name='old');new=self.gate_fixture(name='new');root=core.root(self.p)
  row=dict(source=core.verify()[0],runtime={},gates=[str(old.relative_to(root))],seeds=[161,163,165])
  dest=root/'lanes/B-good/training_complete.json';core.atomic(dest,row)
  with patch.object(study,'gct_gate',return_value=new),patch.object(study,'run_process',return_value={}):study.gct_coordinate(self.p,'B-good','GPU-new',{},self.p/'cache',True)
  self.assertEqual(core.read(dest),row)
 def test_real_collector_normalized_schema_fixture(self):
  sys.path.insert(0,str(core.HERE/'design/tests'));from test_design import fixture
  f=fixture();root=core.root(self.p);root.mkdir()
  inventory=dict(complete=True,collisions=[]);core.atomic(root/'seed_inventory.json',inventory)
  core.atomic(root/'registration.json',dict(inventory_sha256=core.sha(root/'seed_inventory.json')))
  cohort=f['cohort_lock'];cohort.update(source=core.verify()[0],protocol=core.compact(core.GCT2),runtime={});core.atomic(root/'COHORT_LOCK.json',cohort)
  for lane in core.GCT2['lanes']:core.atomic(root/'lanes'/lane/'cache.json',dict(path=str(self.p),stat={}))
  for r in f['reports']:
   r.update(training_source_sha256=core.verify()[0],collector_source_sha256=core.sha(core.HERE/'gct2.py'),cohort_sha256=core.sha(root/'COHORT_LOCK.json'),stop=3550624,runtime={})
   core.atomic(root/'reports'/f"seed_{r['seed']}"/(r['arm']+'.json'),r)
  with patch.object(gct2,'seal_cohort',return_value=cohort):out=gct2.analyze_reports(self.p)
  self.assertTrue(out['joint_confirmation']);self.assertEqual(len(out['rows']),6)
  self.assertEqual(core.read(root/'normalized_reports.json')['schema'],'gct2_endpoint_bundle_v1')
 def test_science_identity_does_not_change_with_gpu_uuid(self):
  j=gct2.stages(self.p,161)[-1];before=study.expected(j,{'gpu':'NVIDIA H200 NVL'},'gct2')
  study.TARGET_UUID='GPU-new';after=study.expected(j,{'gpu':'NVIDIA H200 NVL'},'gct2')
  self.assertEqual(before,after);self.assertNotIn('gpu_uuid',before)
class RegistrationTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.p=Path(self.temp.name);(self.p/'runs').mkdir();(self.p/'logs').mkdir()
 def tearDown(self):self.temp.cleanup()
 def test_atomic_registration_and_reuse(self):
  with patch.object(study.shutil,'disk_usage',return_value=Mock(free=200*1024**3)):
   one=study.register(self.p,'gct2',{});two=study.register(self.p,'gct2',{})
  self.assertEqual(one,two);self.assertTrue((core.root(self.p)/'seed_inventory.json').is_file())
  self.assertFalse(list(self.p.glob('.gct2_registration_*')))
 def test_changed_runtime_refused(self):
  with patch.object(study.shutil,'disk_usage',return_value=Mock(free=200*1024**3)):
   study.register(self.p,'gct2',{})
   with self.assertRaises(RuntimeError):study.register(self.p,'gct2',{'changed':True})
 def test_inventory_collision_no_study_side_effect(self):
  (self.p/'runs/old_seed_161').mkdir()
  with self.assertRaises(RuntimeError):study.register(self.p,'gct2',{})
  self.assertFalse(core.root(self.p).exists())
 def test_failed_publication_leaves_no_partial_study(self):
  original=core.atomic
  def faulty(path,value):
   if Path(path).name=='registration.json':raise OSError('simulated interruption')
   return original(path,value)
  with patch.object(study.shutil,'disk_usage',return_value=Mock(free=200*1024**3)),patch.object(core,'atomic',side_effect=faulty):
   with self.assertRaises(OSError):study.register(self.p,'gct2',{})
  self.assertFalse(core.root(self.p).exists());self.assertFalse(list(self.p.glob('.gct2_registration_*')))
 def test_report_wrong_source_refused_before_access(self):
  root=core.root(self.p);core.atomic(root/'COHORT_LOCK.json',dict(state='sealed',endpoints=[{}]*18,source='wrong',protocol=core.compact(core.GCT2)))
  with self.assertRaisesRegex(RuntimeError,'valid complete cohort'):gct2.report_one(self.p,161,'bdt',{})
if __name__=='__main__':unittest.main(verbosity=2)
