"""Offline tests of the recovered preparation source with simulated API inputs.

These tests exercise token packing, not the real dataset shuffle implementation
or byte reproduction of the scientific token cache. No network API is called.
"""
from pathlib import Path
import contextlib,hashlib,io,json,runpy,sys,tempfile,types,unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repro.data_audit import ROOT,inspect_cache,recorded_environment_comparison
SOURCE=ROOT/'reference/data/prepare_s1_data.py.txt'

class RecoveredPreparation(unittest.TestCase):
    def invoke(self,out,rows,encoding,train=6,val=5):
        calls={}
        class Stream:
            def shuffle(self,**kwargs):calls['shuffle']=kwargs;return self
            def __iter__(self):return iter(rows)
        def load_dataset(*args,**kwargs):calls['dataset']=(args,kwargs);return Stream()
        class Tokenizer:
            vocab_size=100;eos_token_id=99
            def __call__(self,texts,**kwargs):calls['encoding']=kwargs;return {'input_ids':[encoding[t] for t in texts]}
        class Auto:
            @staticmethod
            def from_pretrained(*args,**kwargs):calls['tokenizer']=(args,kwargs);return Tokenizer()
        modules={'datasets':types.SimpleNamespace(load_dataset=load_dataset),
                 'transformers':types.SimpleNamespace(AutoTokenizer=Auto)}
        args=['prepare_s1_data.py','--output-dir',str(out),'--train-tokens',str(train),
              '--val-tokens',str(val),'--text-batch','1']
        with patch.dict(sys.modules,modules),patch.object(sys,'argv',args),contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(SOURCE),run_name='__main__')
        return calls
    def test_original_source_hash(self):
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'ae0538cd75e94a48640eeefcb3ba420a64ac484fb95120ad011a69b8f604097c')
    def test_validation_boundary_document_tail_is_discarded(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);calls=self.invoke(p,[{'text':'a'},{'text':'b'},{'text':'c'}],
                {'a':[11,12,13,14,15,16],'b':[21,22],'c':[31,32,33]})
            self.assertEqual(np.fromfile(p/'val.bin',dtype=np.uint16).tolist(),[11,12,13,14,15])
            self.assertEqual(np.fromfile(p/'train.bin',dtype=np.uint16).tolist(),[21,22,99,31,32,33])
            self.assertEqual(calls['shuffle'],{'seed':20260828,'buffer_size':10000})
            self.assertEqual(calls['encoding'],{'add_special_tokens':False,'truncation':False})
            self.assertEqual(calls['dataset'][1]['revision'],'main')
            self.assertNotIn('revision',calls['tokenizer'][1])
    def test_empty_text_rows_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.invoke(p,[{'text':''},{},{'text':'a'},{'text':'b'}],{'a':[11,12],'b':[21,22]},train=3,val=3)
            self.assertEqual(json.loads((p/'metadata.json').read_text())['documents_consumed'],2)
            self.assertEqual(np.fromfile(p/'val.bin',dtype=np.uint16).tolist(),[11,12,99])
    def test_exhausted_stream_retains_partial_bytes_without_completion_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            with self.assertRaises(RuntimeError):self.invoke(p,[{'text':'a'}],{'a':[11]})
            self.assertTrue((p/'train.bin').exists());self.assertFalse((p/'metadata.json').exists())
    def test_hash_verifier_not_original_count_only_shortcut_is_authoritative(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.invoke(p,[{'text':'a'},{'text':'b'}],{'a':[11,12],'b':[21,22]},train=3,val=3)
            expected={n:{'bytes':(p/n).stat().st_size,'sha256':hashlib.sha256((p/n).read_bytes()).hexdigest()} for n in ('metadata.json','train.bin','val.bin')}
            (p/'train.bin').write_bytes(bytes([0])*6)
            # The original implementation's completion shortcut checks metadata fields only.
            self.invoke(p,[],{},train=3,val=3)
            self.assertEqual((p/'train.bin').read_bytes(),bytes([0])*6)
            self.assertFalse(inspect_cache(p,hash_bytes=True,expected=expected)['exact_cache_verified'])

class DataAudit(unittest.TestCase):
    def fixture(self,p):
        for name in ('metadata.json','train.bin','val.bin'):(p/name).write_bytes(b'1234')
        return {n:{'bytes':4,'sha256':hashlib.sha256(b'1234').hexdigest()} for n in ('metadata.json','train.bin','val.bin')}
    def test_size_preflight_reads_no_payload(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r=inspect_cache(p,expected=self.fixture(p));self.assertTrue(r['sizes_match'])
            self.assertFalse(r['exact_cache_verified']);self.assertEqual(r['payload_bytes_read'],0)
    def test_hashing_is_in_place(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);expected=self.fixture(p);before={n:(p/n).read_bytes() for n in expected}
            r=inspect_cache(p,hash_bytes=True,expected=expected)
            self.assertTrue(r['exact_cache_verified']);self.assertEqual(r['files_written'],0)
            self.assertEqual(r['payload_bytes_read'],12);self.assertEqual(before,{n:(p/n).read_bytes() for n in expected})
    def test_wrong_size_is_not_hashed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);expected=self.fixture(p);(p/'train.bin').write_bytes(b'12345')
            r=inspect_cache(p,hash_bytes=True,expected=expected)
            self.assertFalse(r['exact_cache_verified']);self.assertFalse(r['files']['train.bin']['sha256_checked'])
    def test_missing_is_not_zero_records(self):
        with tempfile.TemporaryDirectory() as d:
            r=inspect_cache(Path(d));self.assertFalse(r['exact_cache_verified']);self.assertFalse(r['sizes_match'])
    def test_partial_contract_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);expected=self.fixture(p);del expected['train.bin']
            with self.assertRaises(ValueError):inspect_cache(p,hash_bytes=True,expected=expected)
    def test_empty_contract_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):inspect_cache(Path(d),hash_bytes=True,expected={})
    def test_runtime_capture_is_not_gpu_verification(self):
        r=recorded_environment_comparison();self.assertEqual(r['contracts_checked_in_archive'],18)
        self.assertFalse(r['gpu_environment_verified']);self.assertFalse(r['environment_modified'])
        self.assertEqual(r['recorded_package_versions']['torch'],'2.11.0+cu128')

if __name__=='__main__':unittest.main()
