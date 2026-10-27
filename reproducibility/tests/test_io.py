"""New wrapper integrity tests, separate from numerical-policy tests."""
from pathlib import Path
import json,subprocess,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repro.training import save,load,FORMAT,code_identity
from repro.bootstrap import MODS,ROOT
import torch

class PortableIO(unittest.TestCase):
    def test_restricted_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.pt';v={'format':FORMAT,'value':torch.arange(5)}
            save(p,v);got=load(p);self.assertTrue(torch.equal(v['value'],got['value']))

    def test_checkpoint_corruption_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.pt';save(p,{'format':FORMAT,'value':torch.ones(2)})
            p.write_bytes(p.read_bytes()+b'changed')
            with self.assertRaises(RuntimeError):load(p)

    def test_existing_checkpoint_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.pt';save(p,{'format':FORMAT})
            with self.assertRaises(FileExistsError):save(p,{'format':FORMAT})

    def test_gpu_training_requires_explicit_flag(self):
        proc=subprocess.run([sys.executable,str(ROOT/'scripts/train_bct1.py'),'--data','not_present',
            '--out','not_created','--seed','946201'],capture_output=True,text=True)
        self.assertNotEqual(proc.returncode,0)
        self.assertIn('explicit --allow-training is required',proc.stderr)
        self.assertFalse((ROOT/'not_created').exists())

    def test_transform_does_not_rotate_moments(self):
        p=torch.nn.Parameter(torch.zeros(3,4));p.grad=torch.arange(12.).reshape(3,4)
        opt=MODS['role'].RoleAdamW({'gate':[p]})
        u=opt.propose(.001);m=opt.state[id(p)]['m'].clone();v=opt.state[id(p)]['v'].clone()
        opt.apply(.001,{name:-value for name,value in u.items()})
        self.assertTrue(torch.equal(m,opt.state[id(p)]['m']))
        self.assertTrue(torch.equal(v,opt.state[id(p)]['v']))

if __name__=='__main__':unittest.main()
