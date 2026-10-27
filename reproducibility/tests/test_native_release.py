"""Portable release/source/plan tests. No CUDA, dataset or checkpoint downloads."""
import ast,hashlib,importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_study_entry',ROOT/'scripts/native_study.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
class NativeReleaseTests(unittest.TestCase):
 def test_exact_and_derivative_hashes(self):
  d=json.loads((ROOT/'studies/ORIGIN.json').read_text())
  for row in d['files']:
   self.assertEqual(hashlib.sha256((ROOT/row['public_member']).read_bytes()).hexdigest(),row['public_sha256'])
   if row['transformation']=='byte-identical':self.assertEqual(row['original_sha256'],row['public_sha256'])
 def test_provenance_labels_match_hash_identity(self):
  d=json.loads((ROOT/'studies/ORIGIN.json').read_text())
  for row in d['files']:
   same=row['original_sha256']==row['public_sha256']
   if row['transformation']=='byte-identical':self.assertTrue(same)
   self.assertFalse((not same) and row['transformation'].startswith('byte-identical'))
 def test_sources_match_public_identities(self):
  d=json.loads((ROOT/'studies/ORIGIN.json').read_text())
  for st in entry.SOURCES:self.assertEqual(entry.source_id(st),d['studies'][st.upper()]['public_source'])
 def test_plan_sizes(self):
  self.assertEqual([len(entry.plan(s)['jobs']) for s in entry.SOURCES],[24,36,42])
 def test_full_seeds_and_arms(self):
  for s,first,last,arms in [('cms1',147,152,3),('gct2',161,166,3),('bac1',175,180,4)]:
   p=entry.plan(s);self.assertEqual(sorted({j['seed'] for j in p['jobs']}),list(range(first,last+1)))
   self.assertEqual(sum(j['stage']=='arm' for j in p['jobs']),6*arms)
 def test_no_gate_seed_in_science_plan(self):
  for st in entry.SOURCES:self.assertNotIn(127,{j['seed'] for j in entry.plan(st)['jobs']})
 def test_parent_dependencies(self):
  for st in entry.SOURCES:
   seen=set()
   for j in entry.plan(st)['jobs']:
    if j['parent'] is not None:self.assertIn(j['parent'],seen)
    seen.add(j['rel'])
 def test_original_schedule_endpoints(self):
  for st in entry.SOURCES:
   for j in entry.plan(st)['jobs']:
    self.assertEqual(j['end'],{'parent':200 if st=='cms1' else 7630,'prefix':200,'grown':0,'arm':3000}[j['stage']])
 def test_plan_import_does_not_import_torch(self):
  code=f"import runpy,sys;sys.argv=['native_study.py','--study','cms1'];runpy.run_path({str(ROOT/'scripts/native_study.py')!r},run_name='__main__');assert 'torch' not in sys.modules"
  r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
 def test_gpu_requires_explicit_permission(self):
  r=subprocess.run([sys.executable,str(ROOT/'scripts/native_study.py'),'--study','cms1','--action','job'],capture_output=True,text=True)
  self.assertNotEqual(r.returncode,0);self.assertIn('--allow-gpu',r.stderr)
 def test_native_report_signatures(self):
  # Do not invent extra arguments for original entrypoints.
  for path,name,n in [('studies/cms1/correspondence_matched_s1_v1/cms_train.py','report_job',2),('studies/gct2/gct2.py','report_one',4),('studies/bac1/experiment.py','evaluate_report',3)]:
   tree=ast.parse((ROOT/path).read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name);self.assertEqual(len(node.args.args),n)
 def test_native_source_syntax(self):
  for f in (ROOT/'studies').rglob('*.py'):ast.parse(f.read_text())
 def test_native_cohort_barriers_retained(self):
  for path,text in [('studies/cms1/correspondence_matched_s1_v1/cms_analysis.py','freeze_cohort'),('studies/gct2/gct2.py','len(cohort[\'endpoints\'])!=18'),('studies/bac1/experiment.py',"len(lock['endpoints']) != 24")]:self.assertIn(text,(ROOT/path).read_text())
