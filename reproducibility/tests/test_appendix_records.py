import hashlib,json,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
from repro.appendix_records import allocation_record_check
ROOT=Path(__file__).resolve().parents[1]
class AppendixRecordTests(unittest.TestCase):
 def test_public_and_original_hashes_are_distinct(self):
  r=allocation_record_check();self.assertNotEqual(r['original_payload_sha256'],r['public_payload_sha256']);self.assertFalse(r['original_hashes_modified'])
 def test_retained_grid_and_mean_cells(self):
  r=allocation_record_check();self.assertEqual(r['records'],240);self.assertEqual(r['displayed_mean_cells_checked'],48)
 def test_no_new_tensor_or_model_claim(self):
  r=allocation_record_check();self.assertFalse(r['new_checkpoint_audit']);self.assertEqual(r['new_model_evaluations'],0)
 def changed(self,change,update_public_hash=False):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'manuscript';shutil.copytree(ROOT/'reference/manuscript_v20',p)
   f=p/'evidence/budget_interpretability/plot_data_compact.json';d=json.loads(f.read_text());change(d);f.write_text(json.dumps(d))
   if update_public_hash:
    op=p/'ORIGIN.json';o=json.loads(op.read_text())
    for row in o['inputs']:
     if row['original_source_member']=='evidence/budget_interpretability/plot_data_compact.json':row['public_sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
    op.write_text(json.dumps(o))
   with self.assertRaises(RuntimeError):allocation_record_check(p)
 def test_changed_public_bytes_refused(self):self.changed(lambda d:d['values'][0][0][0][0].__setitem__(0,.99))
 def test_partial_grid_refused_even_with_rehashed_fixture(self):self.changed(lambda d:d['values'][0][0].pop(),True)
 def test_promoted_tensor_audit_refused(self):self.changed(lambda d:d['checked'].__setitem__('new_checkpoint_payload_audit',True),True)
 def test_replay_refuses_existing_output(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);(p/'keep.txt').write_text('original')
   r=subprocess.run([sys.executable,str(ROOT/'scripts/reproduce_appendix.py'),'--out',td],capture_output=True,text=True)
   self.assertNotEqual(r.returncode,0);self.assertEqual((p/'keep.txt').read_text(),'original')
