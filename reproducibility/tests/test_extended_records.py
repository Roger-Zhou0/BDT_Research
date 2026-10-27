import copy,hashlib,importlib.util,json,shutil,tempfile,unittest
from pathlib import Path
from repro.extended_records import gct2_fev1,bac1_display,bac1_records,BAC1_ANALYSIS_SHA256,BAC1_LOCK_SHA256
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('all_records',ROOT/'scripts/reproduce_all.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class ExtendedRecordsTests(unittest.TestCase):
 def test_gct2_complete(self):
  d=gct2_fev1();self.assertEqual(d['training_parents'],6);self.assertEqual(d['model_endpoint_cells'],18);self.assertEqual(d['record_hash_links'],36)
 def test_gct2_no_new_evaluation(self):self.assertEqual(gct2_fev1()['model_evaluations_performed'],0)
 def test_gct2_report_mean(self):self.assertAlmostEqual(gct2_fev1()['results']['REPORT']['contrasts']['bdt - adamw']['paired_95']['mean'],-.0022038559739788375,14)
 def test_fev1_mean(self):self.assertAlmostEqual(gct2_fev1()['results']['FEV1']['contrasts']['bdt - scrambled_matched']['paired_95']['mean'],-.0032149826465399656,14)
 def test_intervals_nested(self):
  for part in gct2_fev1()['results'].values():
   for r in part['contrasts'].values():
    self.assertLess(r['paired_97_5']['lower'],r['paired_95']['lower']);self.assertGreater(r['paired_97_5']['upper'],r['paired_95']['upper'])
 def bad_gct(self,change,filename='VERIFIED_GCT2_FEV1.json'):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)/'records';shutil.copytree(ROOT/'reference/recovered_records/gct2',root)
   p=root/filename;d=json.loads(p.read_text());change(d);p.write_text(json.dumps(d))
   with self.assertRaises((RuntimeError,ValueError)):gct2_fev1(root)
 def test_missing_gct_row(self):self.bad_gct(lambda d:d['rows'].pop())
 def test_duplicate_gct_row(self):self.bad_gct(lambda d:d['rows'].__setitem__(0,copy.deepcopy(d['rows'][1])))
 def test_changed_gct_score(self):self.bad_gct(lambda d:d['rows'][0].__setitem__('REPORT',0.0))
 def test_changed_binding(self):self.bad_gct(lambda d:d['rows'][0].__setitem__('training_result_sha256','0'*64))
 def test_changed_seal(self):self.bad_gct(lambda d:d.__setitem__('state','not_sealed'),'GCT2_LOCK.json')
 def test_changed_recorded_ci(self):self.bad_gct(lambda d:d['summaries']['REPORT']['contrasts']['bdt_minus_adamw']['ci95'].__setitem__(0,0.0))
 def test_bac1_unrounded_records(self):
  d=bac1_records();self.assertEqual(d['model_endpoint_cells'],24);self.assertEqual(d['endpoint_result_hashes'],24);self.assertEqual(len(d['scores_unrounded']),6)
 def test_bac1_original_record_hashes(self):
  root=ROOT/'reference/recovered_records/bac1';self.assertEqual(hashlib.sha256((root/'BAC1_FINAL_ANALYSIS.json').read_bytes()).hexdigest(),BAC1_ANALYSIS_SHA256);self.assertEqual(hashlib.sha256((root/'BAC1_COHORT_LOCK.json').read_bytes()).hexdigest(),BAC1_LOCK_SHA256)
 def test_bac1_unrounded_primary(self):
  d=bac1_records();r=d['primary']['recomputed_paired_95'];self.assertAlmostEqual(r['mean'],.0033002886921167374,14);self.assertAlmostEqual(r['lower'],.002971574611868282,13);self.assertAlmostEqual(r['upper'],.0036290027723651927,13)
 def test_bac1_corruption_refused(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)/'bac1';shutil.copytree(ROOT/'reference/recovered_records/bac1',root);p=root/'BAC1_FINAL_ANALYSIS.json';d=json.loads(p.read_text());d['rows'][0]['adamw']=0.;p.write_text(json.dumps(d))
   with self.assertRaises(RuntimeError):bac1_records(root)
 def test_bac1_complete_display(self):
  d=bac1_display();self.assertEqual(len(d['scores_at_displayed_precision']),6);self.assertEqual(d['primary_only'],'draw0_reassigned - draw0_preserved')
 def test_bac1_display_precision_label(self):self.assertIn('nine-decimal',bac1_display()['evidence_level'])
 def test_bac1_separate_original_summary(self):
  d=bac1_display();self.assertEqual(d['primary_reported_unrounded_mean'],.003300288692);self.assertEqual(d['model_evaluations_performed'],0)
 def test_bac1_no_partial(self):
  d=json.loads((ROOT/'data/display_tables.json').read_text());d['tables']['bac1_all_endpoints'].pop()
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'data.json';p.write_text(json.dumps(d))
   with self.assertRaises(RuntimeError):bac1_display(p)
 def test_bac1_changed_values(self):
  d=json.loads((ROOT/'data/display_tables.json').read_text());d['tables']['bac1_all_endpoints'][0][4]='9.0'
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'data.json';p.write_text(json.dumps(d))
   with self.assertRaises(RuntimeError):bac1_display(p)
 def test_manuscript_inventory(self):self.assertEqual(len(module.manuscript_inventory()['table_fragments']),42)
 def test_origin_exact_tables(self):
  d=json.loads((ROOT/'reference/manuscript_v20/ORIGIN.json').read_text())
  for row in d['inputs']:
   if '/tables/' in row['member']:self.assertEqual(row['original_sha256'],row['public_sha256'])
 def test_no_merged_parent_units(self):
  d=module.recompute_all();self.assertEqual(d['allocation']['additional_independent_replications'],0);self.assertEqual(d['GCT2_FEV1']['training_parents'],6);self.assertEqual(d['AGT1_FRT1_ART1_BCT1']['parent_units'],4)
 def test_full_record_check(self):self.assertEqual(module.recompute_all()['new_model_evaluations'],0)
