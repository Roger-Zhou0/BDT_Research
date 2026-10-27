"""Synthetic fixtures only. These are not scientific results."""
import copy,json,math,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from design import *
from admission import scan,seed_values
from analyze_gct2 import analyze,paired
from analyze_bco1 import analyze as timing

def h(value):return digest(str(value))
def fixture():
    required=('source','cache','growth_mapping','optimizer_age','all_parents_distinct','lineage',
              'first_active_inputs','full_batch_rng_histories','matching','final_checkpoints','target_gates')
    b=dict(schema='gct2_endpoint_bundle_v1',protocol_digest=digest(GCT2),seeds=list(SEEDS),cache_sha256=CACHE,
           report_partition=GCT2['report'],parent_cost_counted_per_path=True,
           inventory=dict(complete=True,collisions=[],manifest_sha256=h('inventory')),
           checks={k:True for k in required},training_source_sha256=h('source'),collector_source_sha256=h('collector'),
           cohort_lock=dict(state='sealed',sealed_at='2026-09-20T00:00:00Z',endpoints=[]),reports=[])
    for i,s in enumerate(SEEDS):
        for arm in ARMS:
            row=dict(seed=s,arm=arm,step=3000,adam_age=10630,parent_steps=7630,prefix_steps=200,
                     completed_at='2026-09-19T23:00:00Z',training_result_sha256=h((s,arm,'train')),
                     checkpoint_sha256=h((s,arm,'ck')),prefix_sha256=h((s,'prefix')),small_parent_sha256=h((s,'parent')))
            b['cohort_lock']['endpoints'].append(row)
            # Deliberately synthetic, varying seed differences for testing only.
            loss=4+i*.01+{'adamw':.004+i*.0003,'bdt':0,'scrambled_matched':.008+i*.0007}[arm]
            b['reports'].append(dict(seed=s,arm=arm,loss=loss,partition='REPORT',tokens=1048576,offset=2500000,
               started_at='2026-09-20T00:01:00Z',completed_at='2026-09-20T00:01:10Z',
               training_result_sha256=row['training_result_sha256'],checkpoint_sha256=row['checkpoint_sha256'],
               training_source_sha256=h('source'),collector_source_sha256=h('collector')))
    return b

def time_fixture(family='full_step'):
    n=20 if family=='full_step' else 100;out=[]
    for i,order in enumerate(BCO1['timing']['orders']):
        def row(condition):
            return dict(condition=condition,updates=n,start=501,end=500+n,
                wall_ms=n*{'split_adam':100,'reference_bdt':110,'candidate_bdt':103}[condition],
                observations_per_projection=n//10 if condition!='split_adam' else 0,
                common_state_sha256=h('common'),restored_after_warmup=True,synchronized=True,
                profiler=False,shadow_matching=False,audit_hashes_inside_timer=False,
                final_model_sha256=h('final'),final_adam_sha256=h('adam'),final_covariance_sha256=h('cov'))
        out.append(dict(block=i,family=family,state='complete',protocol_digest=digest(BCO1),exact_equivalence_gate=True,
           gpu_uuid='GPU-fixture',runtime_sha256=h('rt'),source_sha256=h('src'),measurements=[row(c) for c in order],
           sentinel_before=row('split_adam'),sentinel_after=row('split_adam')))
    return out

class DesignTests(unittest.TestCase):
    def test_exact_science_budget(self):self.assertEqual(budget()['total_scientific_updates'],97380)
    def test_per_path_cost(self):self.assertEqual(budget()['recommended_path_token_exposures'],348323840)
    def test_job_count_and_updates(self):self.assertEqual((len(jobs()),sum(x['updates'] for x in jobs())),(36,97380))
    def test_eighteen_branches(self):self.assertEqual(sum(j['stage']=='arm' for j in jobs()),18)
    def test_parent_independence_count(self):self.assertEqual(sum(j['stage']=='parent' for j in jobs()),6)
    def test_arm_orders_balanced(self):self.assertEqual(set(ORDER.values()),set(permutations(ARMS)))
    def test_within_lane_balance(self):
        for seeds in ([161,163,165],[162,164,166]):
            for pos in range(3):self.assertEqual({ORDER[s][pos] for s in seeds},set(ARMS))
    def test_dependency_dag(self):
        seen=set()
        for j in jobs():self.assertTrue(set(j['depends_on'])<=seen);self.assertNotIn(j['id'],seen);seen.add(j['id'])
    def test_report_disjoint_from_dev(self):self.assertLessEqual(GCT2['dev']['stop'],GCT2['report']['offset'])
    def test_report_fits_val(self):self.assertLessEqual(GCT2['report']['stop']*2,10000000)
    def test_gate_names_follow_uuid(self):
        self.assertNotEqual(engineering_key('B-good','GPU-a','rt','src'),engineering_key('B-good','GPU-b','rt','src'))
    def test_unsafe_gate_component(self):
        with self.assertRaises(ValueError):engineering_key('../bad','GPU-b','rt','src')
    def test_manifest_export(self):
        with tempfile.TemporaryDirectory() as td:
            write_manifests(td);self.assertEqual(json.loads((Path(td)/'GCT2.json').read_text()),GCT2)
    def test_candidate_not_admitted_to_science(self):self.assertFalse(GCT2['release']['allow_BCO1_candidate_in_confirmation'])

class InferenceTests(unittest.TestCase):
    def test_valid_joint(self):self.assertTrue(analyze(fixture())['joint_confirmation'])
    def test_all_estimates_retained_when_unfavorable(self):
        b=fixture()
        for row in b['reports']:
            if row['arm']=='bdt':row['loss']+=.02
        r=analyze(b);self.assertFalse(r['joint_confirmation']);self.assertEqual(len(r['comparisons']),2)
    def test_missing_seed_refusal(self):
        b=fixture();b['reports'].pop()
        with self.assertRaises(ValueError):analyze(b)
    def test_duplicate_refusal(self):
        b=fixture();b['reports'][-1]=b['reports'][0]
        with self.assertRaises(ValueError):analyze(b)
    def test_pilot_pooling_refusal(self):
        b=fixture();b['seeds']=[159,160]+list(SEEDS)
        with self.assertRaises(ValueError):analyze(b)
    def test_early_report_refusal(self):
        b=fixture();b['reports'][0]['started_at']='2026-09-19T23:59:59Z'
        with self.assertRaises(ValueError):analyze(b)
    def test_late_training_refusal(self):
        b=fixture();b['cohort_lock']['endpoints'][0]['completed_at']='2026-09-20T00:00:01Z'
        with self.assertRaises(ValueError):analyze(b)
    def test_all_protocol_fields_locked(self):
        b=fixture();b['protocol_digest']=h('other')
        with self.assertRaises(ValueError):analyze(b)
    def test_bad_cache_refusal(self):
        b=fixture();b['cache_sha256']=dict(CACHE,**{'train.bin':h('other')})
        with self.assertRaises(ValueError):analyze(b)
    def test_unverified_lineage_refusal(self):
        b=fixture();b['checks']['lineage']=False
        with self.assertRaises(ValueError):analyze(b)
    def test_wrong_checkpoint_refusal(self):
        b=fixture();b['reports'][0]['checkpoint_sha256']=h('other')
        with self.assertRaises(ValueError):analyze(b)
    def test_parent_reuse_refusal(self):
        b=fixture()
        for row in b['cohort_lock']['endpoints']:row['small_parent_sha256']=h('same_parent')
        with self.assertRaises(ValueError):analyze(b)
    def test_unequal_prefix_refusal(self):
        b=fixture();b['cohort_lock']['endpoints'][0]['prefix_sha256']=h('other')
        with self.assertRaises(ValueError):analyze(b)
    def test_global_age_refusal(self):
        b=fixture();b['cohort_lock']['endpoints'][0]['adam_age']=3000
        with self.assertRaises(ValueError):analyze(b)
    def test_parent_cost_exclusion_refusal(self):
        b=fixture();b['parent_cost_counted_per_path']=False
        with self.assertRaises(ValueError):analyze(b)
    def test_nonfinite_refusal(self):
        b=fixture();b['reports'][0]['loss']=float('nan')
        with self.assertRaises(ValueError):analyze(b)
    def test_adjusted_interval_wider(self):
        r=analyze(fixture())['comparisons']['bdt_minus_adamw'];self.assertLess(r['ci975'][0],r['ci95'][0])
    def test_zero_variance_review(self):self.assertIsNone(paired([-.01]*6)['ci975'])

class InventoryTests(unittest.TestCase):
    def test_nested_metadata(self):self.assertIn((161,'contract.seed'),seed_values({'contract':{'seed':161}}))
    def test_clear_scan_no_write(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'runs/a').mkdir(parents=True);(root/'runs/a/result.json').write_text('{"seed":160}')
            before=set(root.rglob('*'));r=scan(root);self.assertEqual(r['state'],'clear');self.assertEqual(before,set(root.rglob('*')))
    def test_collision_stops(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'runs/a').mkdir(parents=True);(root/'runs/a/result.json').write_text('{"contract":{"seed":161}}')
            self.assertEqual(scan(root)['state'],'blocked')
    def test_folder_collision(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'runs/a_seed_166').mkdir(parents=True);self.assertEqual(scan(root)['state'],'blocked')
    def test_unreadable_metadata_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'runs/a').mkdir(parents=True);(root/'runs/a/result.json').write_text('{bad')
            self.assertFalse(scan(root)['complete'])

class TimingTests(unittest.TestCase):
    def test_paired_cost_success_fixture(self):self.assertTrue(timing(time_fixture())['meaningful_engineering_reduction'])
    def test_optimizer_only_not_full_step_claim(self):self.assertFalse(timing(time_fixture('optimizer_only'),'optimizer_only')['meaningful_engineering_reduction'])
    def test_incomplete_block_set(self):
        with self.assertRaises(ValueError):timing(time_fixture()[:-1])
    def test_wrong_order(self):
        b=time_fixture();b[0]['measurements'].reverse()
        with self.assertRaises(ValueError):timing(b)
    def test_different_uuid(self):
        b=time_fixture();b[0]['gpu_uuid']='GPU-other'
        with self.assertRaises(ValueError):timing(b)
    def test_different_runtime(self):
        b=time_fixture();b[0]['runtime_sha256']=h('other')
        with self.assertRaises(ValueError):timing(b)
    def test_mutated_warmup(self):
        b=time_fixture();b[0]['measurements'][0]['restored_after_warmup']=False
        with self.assertRaises(ValueError):timing(b)
    def test_profiler_contamination(self):
        b=time_fixture();b[0]['measurements'][0]['profiler']=True
        with self.assertRaises(ValueError):timing(b)
    def test_missing_covariance_refresh(self):
        b=time_fixture();r=next(r for r in b[0]['measurements'] if r['condition']=='candidate_bdt');r['observations_per_projection']=0
        with self.assertRaises(ValueError):timing(b)
    def test_changed_final_tensor_hash(self):
        b=time_fixture();r=next(r for r in b[0]['measurements'] if r['condition']=='candidate_bdt');r['final_model_sha256']=h('other')
        with self.assertRaises(ValueError):timing(b)
    def test_all_blocks_retained_when_unstable(self):
        b=time_fixture()
        for item in b[:5]:item['sentinel_after']['wall_ms']*=1.2
        r=timing(b);self.assertEqual(r['quality_control'],'timing_inconclusive');self.assertEqual(r['metrics']['candidate_over_reference']['all_blocks']['n'],12)
        self.assertFalse(r['meaningful_engineering_reduction'])
    def test_no_gain_not_relabelled(self):
        b=time_fixture()
        for item in b:
            row=next(r for r in item['measurements'] if r['condition']=='candidate_bdt');row['wall_ms']*=2
        self.assertFalse(timing(b)['meaningful_engineering_reduction'])

if __name__=='__main__':unittest.main(verbosity=2)
