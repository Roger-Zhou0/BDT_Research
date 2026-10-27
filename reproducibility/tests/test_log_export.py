"""Synthetic log fixtures only. No experiment records are altered for these tests."""
from pathlib import Path
import hashlib,json,tempfile,unittest
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repro.log_export import Limits,plan,export,inspect_text

class LogViews(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.src=self.root/'input.txt';self.out=self.root/'view'
        self.limits=Limits(max_file_bytes=10000,max_input_bytes=20000,max_output_bytes=30000,minimum_free_bytes=0)
    def tearDown(self):self.tmp.cleanup()
    def spec(self,fmt='jsonl',state='unknown',scope='unknown'):
        return {'inputs':[{'source_id':'source_01','run_id':'synthetic_run','attempt_id':'attempt_01',
                'path':str(self.src),'format':fmt,'run_state':state,'capture_scope':scope}]}
    def run_export(self,**kwargs):return export(self.spec(),self.out,limits=self.limits,**kwargs)
    def entry(self):return json.loads((self.out/'INVENTORY.json').read_text())['inputs'][0]
    def test_plan_reads_metadata_only(self):
        self.src.write_bytes(b'not-json');r=plan(self.spec(),self.limits)
        self.assertFalse(r['content_read']);self.assertEqual(r['files_written'],0);self.assertFalse(self.out.exists())
    def test_partial_tail_is_retained_and_flagged(self):
        raw=b'{"step":1,"loss":2.1}\n{"step":2,"loss":'
        self.src.write_bytes(raw);r=self.run_export();e=self.entry()
        self.assertEqual(self.src.read_bytes(),raw);self.assertEqual((self.out/e['view']).read_bytes(),raw)
        self.assertEqual(e['diagnostics']['parse_issues'][0]['kind'],'invalid_unterminated_final_line')
        self.assertEqual(e['source_sha256'],hashlib.sha256(raw).hexdigest());self.assertFalse(r['run_completion_verified'])
    def test_malformed_interior_line_not_deleted(self):
        text='{"step":1}\nTraceback: interrupted\n{"step":2}\n';self.src.write_text(text)
        self.run_export();e=self.entry();self.assertEqual(e['diagnostics']['valid_json_records'],2)
        self.assertEqual((self.out/e['view']).read_text(),text)
    def test_exact_duplicate_is_retained(self):
        self.src.write_text('{"step":1}\n{"step":1}\n');self.run_export();d=self.entry()['diagnostics']
        self.assertEqual(d['valid_json_records'],2);self.assertEqual(len(d['duplicate_records']),1)
    def test_different_records_same_step_not_collapsed(self):
        self.src.write_text('{"step":1,"loss":2}\n{"step":1,"loss":3}\n');self.run_export();d=self.entry()['diagnostics']
        self.assertEqual(len(d['duplicate_records']),0);self.assertEqual(len(d['repeated_steps']),1)
    def test_step_regression_is_not_new_seed(self):
        self.src.write_text('{"phase":310}\n{"phase":300}\n');self.run_export();e=self.entry()
        self.assertEqual(len(e['diagnostics']['step_regressions']),1);self.assertEqual(e['run_id'],'synthetic_run')
    def test_valid_unterminated_record_is_not_claimed_truncated(self):
        d=inspect_text('{"step":1}','jsonl');self.assertFalse(d['ends_with_newline']);self.assertEqual(d['parse_issues'],[])
    def test_declared_complete_is_not_verified_completion(self):
        self.src.write_text('{"step":1}\n');export(self.spec(state='complete',scope='excerpt'),self.out,limits=self.limits)
        e=self.entry();self.assertEqual(e['declared_run_state'],'complete');self.assertFalse(e['run_completion_verified'])
        self.assertEqual(e['declared_capture_scope'],'excerpt')
    def test_limits_refuse_before_output(self):
        self.src.write_bytes(b'x'*12)
        with self.assertRaises(ValueError):export(self.spec(),self.out,limits=Limits(10,20,10000,0))
        self.assertFalse(self.out.exists());self.assertEqual(self.src.stat().st_size,12)
    def test_total_limit_refuses(self):
        self.src.write_bytes(b'12345');s=self.spec();s['inputs'].append(dict(s['inputs'][0],source_id='source_02'))
        with self.assertRaises(ValueError):plan(s,Limits(10,9,10000,0))
    def test_low_disk_refuses(self):
        self.src.write_text('{}\n')
        class Disk:free=100
        with patch('repro.log_export.shutil.disk_usage',return_value=Disk()):
            with self.assertRaises(OSError):self.run_export()
        self.assertFalse(self.out.exists())
    def test_existing_output_refused(self):
        self.src.write_text('{}\n');self.out.mkdir();(self.out/'retain.txt').write_text('keep')
        with self.assertRaises(FileExistsError):self.run_export()
        self.assertEqual((self.out/'retain.txt').read_text(),'keep')
    def test_symlink_refused(self):
        p=self.root/'actual.txt';p.write_text('{}');self.src.symlink_to(p)
        with self.assertRaises(ValueError):self.run_export()
    def test_binary_checkpoint_refused(self):
        self.src=self.root/'weights.pt';self.src.write_bytes(b'fake')
        with self.assertRaises(ValueError):self.run_export()
    def test_utf8_error_is_not_silently_repaired(self):
        self.src.write_bytes(b'abc\xff')
        with self.assertRaises(UnicodeDecodeError):self.run_export()
        self.assertFalse(self.out.exists());self.assertEqual(self.src.read_bytes(),b'abc\xff')
    def test_redaction_has_public_lineage_without_private_path(self):
        private='/'+'home'+'/example_user/work';self.src.write_text(json.dumps({'path':private,'loss':2.3})+'\n')
        with self.assertRaises(ValueError):self.run_export()
        self.run_export(redactions={private:'<REDACTED_PATH>'});e=self.entry()
        public=''.join(p.read_text() for p in self.out.rglob('*') if p.is_file())
        self.assertNotIn(private,public);self.assertIn(private,self.src.read_text())
        self.assertEqual(e['redaction_counts_by_placeholder'],{'<REDACTED_PATH>':1})
    def test_ansi_removal_only_on_requested_view(self):
        self.src.write_bytes(b'\x1b[31mERROR\x1b[0m\n')
        export(self.spec(fmt='text'),self.out,strip_ansi=True,limits=self.limits)
        e=self.entry();self.assertEqual(e['ansi_sequences_removed'],2)
        self.assertEqual((self.out/e['view']).read_text(),'ERROR\n');self.assertIn(b'\x1b',self.src.read_bytes())
    def test_redaction_cannot_rename_a_scientific_field(self):
        self.src.write_text('{"loss":2.3}\n')
        with self.assertRaises(ValueError):self.run_export(redactions={'loss':'<REDACTED_NAME>'})
        self.assertFalse(self.out.exists())
    def test_redaction_cannot_change_numeric_payload(self):
        self.src.write_text('{"loss":2.3}\n')
        with self.assertRaises(ValueError):self.run_export(redactions={'"loss":2.3':'<REDACTED_VALUE>'})
        self.assertFalse(self.out.exists())
    def test_nan_is_flagged_not_erased(self):
        self.src.write_text('{"loss":NaN}\n');self.run_export();self.assertEqual(len(self.entry()['diagnostics']['parse_issues']),1)
        self.assertIn('NaN',(self.out/'views/source_01.txt').read_text())
    def test_json_document_failure_is_explicit(self):
        self.assertEqual(inspect_text('{','json')['parse_issues'][0]['kind'],'invalid_json_document')
    def test_neutral_id_path_traversal_refused(self):
        self.src.write_text('{}');s=self.spec();s['inputs'][0]['source_id']='../escape'
        with self.assertRaises(ValueError):plan(s,self.limits)
    def test_attempts_remain_separate(self):
        self.src.write_text('{}\n');s=self.spec();s['inputs'].append(dict(s['inputs'][0],source_id='source_02',attempt_id='attempt_02',resumed_from_attempt='attempt_01'))
        export(s,self.out,limits=self.limits);m=json.loads((self.out/'INVENTORY.json').read_text())
        self.assertEqual(len(m['inputs']),2);self.assertFalse(m['resume_chains_verified'])
        self.assertEqual(m['inputs'][1]['declared_resumed_from_attempt'],'attempt_01')

if __name__=='__main__':unittest.main()
