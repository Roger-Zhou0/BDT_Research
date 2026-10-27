"""Independent invariant assertions around unchanged archived kernels (CPU only)."""
from pathlib import Path
import copy
import hashlib
import json
import math
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repro.bootstrap import MODS, ROOT, configure_cpu
from repro.geometry import (OnlineDepthTemper, TemperConfig, helmert_basis, RoleAdamW, transport,
    match_direction, make_permutations, DegenerateDirection, sectors, exchange, smooth_window,
    sector_pair, UndefinedSector, make_family, DifferenceBank)
import torch
configure_cpu(1)


def raw(seed=17):
    g=torch.Generator().manual_seed(seed)
    return torch.randn(12,8,6,generator=g) * torch.linspace(.1,2.,12)[:,None,None]


def warm():
    t=OnlineDepthTemper(12,config=TemperConfig(max_operator_condition=3.))
    for step in range(1,201):t.transform(raw(step),step)
    return t


def norm(x):return float(torch.linalg.vector_norm(x.double()))
def centered(x):return x.double()-x.double().mean(0,keepdim=True)


class Kernels(unittest.TestCase):
    def test_native_source_hashes(self):
        m=json.loads((ROOT/'native/NATIVE_KERNEL_MANIFEST.json').read_text())
        self.assertEqual(len(m),19)
        for rel,h in m.items():self.assertEqual(hashlib.sha256((ROOT/'native'/rel).read_bytes()).hexdigest(),h,rel)

    def test_helmert(self):
        b=helmert_basis(12,device=torch.device('cpu'),dtype=torch.float64)
        self.assertLess(norm(b.T@b-torch.eye(11)),1e-14)
        self.assertLess(norm(b.sum(0)),1e-14)

    def test_prefix_identity_and_observations(self):
        t=OnlineDepthTemper(12)
        for step in range(1,201):
            u=raw(step);v,r=t.transform(u,step);self.assertTrue(torch.equal(u,v))
        self.assertEqual(t.observations,20)

    def test_bdt_norm_common_angle_cap(self):
        t=warm()
        for step in range(201,421):
            u=raw(step);v,r=t.transform(u,step)
            self.assertLess(abs(norm(centered(v))/norm(centered(u))-1),5e-6)
            self.assertLess(norm(v.double().mean(0)-u.double().mean(0))/norm(u),1e-6)
            a=centered(u);b=centered(v)
            self.assertGreaterEqual(float((a*b).sum())/(norm(a)*norm(b)),2*math.sqrt(3)/4-1e-6)
            self.assertLessEqual(r['operator_condition'],3.00001)

    def test_release_exact_identity(self):
        t=warm();u=raw();v,_=t.transform(u,201,active=False);self.assertTrue(torch.equal(v,u))
        self.assertEqual(t.last_step,201)

    def test_temper_resume(self):
        a=warm();b=OnlineDepthTemper(12,config=a.config);b.load_state_dict(a.state_dict())
        for step in range(201,211):
            u=raw(step);va,_=a.transform(u,step);vb,_=b.transform(u,step)
            self.assertTrue(torch.equal(va,vb))

    def test_temper_refuses_age_change(self):
        t=warm()
        with self.assertRaises(ValueError):t.transform(raw(),202)

    def test_temper_refuses_config_change(self):
        t=warm();other=OnlineDepthTemper(12)
        with self.assertRaises(ValueError):other.load_state_dict(t.state_dict())

    def test_temper_nonfinite_refusal(self):
        t=OnlineDepthTemper(12);u=raw();u[0,0,0]=float('nan')
        with self.assertRaises(FloatingPointError):t.transform(u,1)

    def test_zero_centered_identity(self):
        t=warm();u=torch.ones(12,8,6);v,_=t.transform(u,201)
        self.assertTrue(torch.equal(u,v))

    def test_transport_roundtrip_all_swiglu_projections(self):
        p=make_permutations(12,8,127)
        for name,u in [('gate',raw()),('up',raw()),('down',raw().transpose(1,2).contiguous())]:
            self.assertTrue(torch.equal(transport(transport(u,p,name),p,name,inverse=True),u))

    def test_matched_direction_invariants(self):
        t=warm();u=raw();v,_=t.transform(u,201)
        candidate=raw(123)
        w,r=match_direction(u,v,candidate)
        a=centered(u);b=centered(v);c=centered(w)
        self.assertLess(abs(norm(c)/norm(a)-1),1e-5)
        self.assertLess(norm(w.double().mean(0)-u.double().mean(0))/norm(u),1e-5)
        self.assertLess(abs(float((a*b).sum())/(norm(a)*norm(b))-float((a*c).sum())/(norm(a)*norm(c))),1e-5)
        self.assertLess(abs(norm(c-a)/norm(a)-norm(b-a)/norm(a)),1e-5)
        self.assertEqual(r['kind'],'matched')

    def test_undefined_matched_direction_refuses(self):
        t=warm();u=raw();v,_=t.transform(u,201)
        with self.assertRaises(DegenerateDirection):match_direction(u,v,u)

    def test_balanced_charts_preserve_multiset(self):
        d=make_family(6,8,127,0)
        self.assertEqual(d['preserved']['audit']['preserved_ancestry_pairs'],6)
        self.assertEqual(d['reassigned']['audit']['preserved_ancestry_pairs'],0)
        for row in d.values():self.assertEqual(row['audit']['chart_multiplicities'],[2]*6)
        self.assertEqual(d['preserved']['audit']['chart_multiset_sha256'],d['reassigned']['audit']['chart_multiset_sha256'])

    def test_ancestry_sectors_orthogonal_recompose(self):
        u=raw();a,d=sectors(centered(u));c=centered(u)
        self.assertLess(norm(a+d-c),1e-12)
        self.assertLess(abs(float((a*d).sum())),1e-12)
        self.assertLess(abs(norm(a)**2+norm(d)**2-norm(c)**2),1e-10)

    def test_sector_matched_rotation(self):
        t=warm();u=raw();v,_=t.transform(u,201)
        plus,minus,r=sector_pair(u,v,[a for a in range(6) for _ in range(2)])
        for y in (plus,minus):self.assertLess(abs(norm(centered(y))/norm(centered(u))-1),1e-5)
        self.assertEqual(r['kind'],'matched')

    def test_window_boundary_weights(self):
        u=torch.arange(12.).reshape(12,1)
        actual=smooth_window(u,400)
        expected=.9*u+.05*(torch.cat([u[:1],u[:-1]])+torch.cat([u[1:],u[-1:]]))
        self.assertTrue(torch.equal(actual,expected));self.assertAlmostEqual(float(actual[0]),.05,6)
        self.assertTrue(torch.equal(smooth_window(u,200),u))

    def test_window_nonfinite_refusal(self):
        u=raw();u[0,0,0]=float('inf')
        with self.assertRaises(FloatingPointError):smooth_window(u,400)

    def test_budget_exchange_preserves_each_budget(self):
        t=warm();u=raw();b,_=t.transform(u,201);g=smooth_window(u,201)
        v,r=exchange(u,b,g,True)
        for x,y in zip(sectors(centered(b)),sectors(centered(v))):self.assertLess(abs(norm(x)-norm(y))/norm(centered(u)),5e-5)
        self.assertLess(abs(norm(centered(v))/norm(centered(u))-1),5e-5)
        self.assertEqual(r['fallback_sectors'],[])

    def test_budget_exchange_undefined_carrier_refuses(self):
        u=raw();carrier=u.mean(0,keepdim=True).expand_as(u).clone()
        # Exact zero direction, nonzero positive budgets.
        with self.assertRaises(RuntimeError):exchange(u,u,carrier)

    def test_candidate_order_no_state_cross_contamination(self):
        full=warm();d=DifferenceBank(full);full_copy=copy.deepcopy(full);d_copy=copy.deepcopy(d)
        for step in range(201,221):
            u=raw(step);b,_=full.transform(u,step);s,_=d.transform(u,step)
            s2,_=d_copy.transform(u,step);b2,_=full_copy.transform(u,step)
            self.assertTrue(torch.equal(b,b2));self.assertTrue(torch.equal(s,s2))

    def test_native_adam_moments_and_separate_decay(self):
        initial=torch.randn(3,4,generator=torch.Generator().manual_seed(32))
        p=torch.nn.Parameter(initial.clone());q=torch.nn.Parameter(initial.clone())
        role=RoleAdamW({'gate':[p]},betas=(.9,.95),eps=1e-8,weight_decay=.1)
        ref=torch.optim.AdamW([q],lr=.001,betas=(.9,.95),eps=1e-8,weight_decay=.1,foreach=False)
        for step in range(4):
            grad=torch.full_like(p,.3+step);p.grad=grad.clone();q.grad=grad.clone()
            proposed=role.propose(.001);role.apply(.001,proposed);ref.step()
            self.assertTrue(torch.allclose(p,q,atol=2e-7,rtol=1e-6))
            self.assertTrue(torch.equal(role.state[id(p)]['m'],ref.state[q]['exp_avg']))
            self.assertTrue(torch.equal(role.state[id(p)]['v'],ref.state[q]['exp_avg_sq']))

if __name__=='__main__':unittest.main()
