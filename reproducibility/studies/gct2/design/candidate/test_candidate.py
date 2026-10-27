"""CPU transformation tests, not full-model training or H200 timing evidence."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, tempfile, unittest
import torch
from prepare_candidate import build, transform_source, REFERENCE_SHA

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

REF=None
CAND=None
REFERENCE_PATH=None

class CandidateTests(unittest.TestCase):
    def pair(self,dtype=torch.float32,roles=12):
        a=REF.OnlineDepthTemper(roles,'cpu',dtype,REF.TemperConfig(200,200,10,.9,3.))
        b=CAND.OnlineDepthTemper(roles,'cpu',dtype,CAND.TemperConfig(200,200,10,.9,3.))
        return a,b
    def equal_state(self,a,b):
        aa,bb=a.state_dict(),b.state_dict()
        self.assertEqual(set(aa),set(bb))
        for key in aa:
            if torch.is_tensor(aa[key]):self.assertTrue(torch.equal(aa[key],bb[key]),key)
            else:self.assertEqual(aa[key],bb[key],key)
    def check(self,steps,dtype=torch.float32,diagnostics=False,active=True,zero=False):
        a,b=self.pair(dtype);gen=torch.Generator(device='cpu').manual_seed(8127)
        nonidentity=0
        before=torch.get_rng_state().clone()
        for step in range(1,steps+1):
            u=torch.randn(12,64,generator=gen,dtype=dtype)*.001
            if zero and step%7==0:u.zero_()
            if zero and step%11==0:u.fill_(.001)
            out,report=a.transform(u,step,active=active)
            other,got=b.transform(u,step,active=active,diagnostics=diagnostics)
            self.assertTrue(torch.equal(out,other),str(step))
            self.equal_state(a,b)
            for key,value in got.items(): self.assertEqual(value,report[key],key)
            self.assertIn('centered_norm_ratio',got);self.assertIn('operator_condition',got)
            if diagnostics:self.assertEqual(got,report)
            if step>200 and not torch.equal(u,out):nonidentity+=1
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        if active and steps>200:self.assertGreater(nonidentity,0)
    def test_fp32_1000_update_and_state_identity(self):self.check(1000)
    def test_fp64_420_update_and_state_identity(self):self.check(420,torch.float64)
    def test_full_diagnostic_output_unchanged(self):self.check(420,diagnostics=True)
    def test_inactive_observer_identity(self):self.check(420,active=False)
    def test_zero_and_common_only_proposals(self):self.check(420,zero=True)
    def test_interrupted_resume(self):
        a,b=self.pair();gen=torch.Generator().manual_seed(319)
        for step in range(1,421):
            u=torch.randn(12,96,generator=gen)*.001
            out,_=a.transform(u,step);other,_=b.transform(u,step,diagnostics=False)
            self.assertTrue(torch.equal(out,other))
            if step==310:
                saved=b.state_dict()
                b=CAND.OnlineDepthTemper(12,'cpu',torch.float32,CAND.TemperConfig(200,200,10,.9,3.))
                b.load_state_dict(saved)
            self.equal_state(a,b)
    def test_nonfinite_refusal(self):
        for v in (float('nan'),float('inf')):
            a,b=self.pair();u=torch.ones(12,8);u[0,0]=v
            for t in (a,b):
                with self.assertRaises(FloatingPointError):t.transform(u,1)
    def test_wrong_order_and_shape_refusal(self):
        a,b=self.pair()
        for t in (a,b):
            with self.assertRaises(ValueError):t.transform(torch.ones(12,8),2)
            with self.assertRaises(ValueError):t.transform(torch.ones(11,8),1)
    def test_source_and_overwrite_refusal(self):
        with self.assertRaises(ValueError):transform_source(REFERENCE_PATH.read_bytes()+b'\n')
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'candidate.py';p.write_text('different')
            with self.assertRaises(ValueError):build(REFERENCE_PATH,p)
        with self.assertRaises(ValueError):build(REFERENCE_PATH,REFERENCE_PATH)
    def test_reference_unmodified(self):
        self.assertEqual(hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest(),REFERENCE_SHA)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--output',type=Path)
    args=ap.parse_args();REFERENCE_PATH=args.reference.resolve()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    with tempfile.TemporaryDirectory(prefix='bco1_cpu_') as td:
        candidate=Path(td)/'candidate.py';meta=build(REFERENCE_PATH,candidate)
        REF=load('bco1_test_reference',REFERENCE_PATH);CAND=load('bco1_test_candidate',candidate)
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CandidateTests))
        record=dict(meta,tests_run=result.testsRun,passed=result.wasSuccessful(),torch=torch.__version__,device='cpu',
                    test_scope='Exact tensor/covariance state checks on small proposal stacks. Not full model/CUDA/performance evidence.')
        if args.output:args.output.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record,indent=2))
        raise SystemExit(0 if result.wasSuccessful() else 1)
