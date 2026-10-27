#!/usr/bin/env python3
"""Unit and actual-frozen-backend CPU integration tests. No GPU allocation."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import cms_protocol as p
import cms_io as io
import cms_runtime as rt
from cms_geometry import make_permutations,transport,match_direction,validate_permutations,DegenerateDirection


def centered(x):return x-x.mean(0,keepdim=True)


def target_with_angle(raw,theta):
    u=raw.double();m=u.mean(0,keepdim=True);c=u-m;n=c.norm()
    g=torch.Generator(device='cpu').manual_seed(824)
    d=centered(torch.randn(c.shape,generator=g,dtype=torch.float64))
    d=d-(d*c).sum()/n.square()*c;d=d/d.norm()
    return (m+np.cos(theta)*c+np.sin(theta)*n*d).to(raw.dtype)


class GeometryTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)
    def test_norm_angle_and_displacement(self):
        for dtype in (torch.float32,torch.float64):
            for roles in (4,12):
                for angle in (.0001,.03,.3):
                    u=torch.randn(roles,7,5,dtype=dtype)
                    target=target_with_angle(u,angle)
                    candidate=u+.005*torch.randn_like(u)
                    v,report=match_direction(u,target,candidate)
                    self.assertEqual(report['kind'],'matched')
                    self.assertLess(report['angle_error'],p.GEOMETRY['angle_absolute_tolerance'])
                    torch.testing.assert_close(v.double().mean(0),u.double().mean(0),atol=2e-7,rtol=1e-6)
                    self.assertAlmostEqual(float(centered(v.double()).norm()/centered(u.double()).norm()),1.,places=6)
    def test_zero_is_exact(self):
        u=torch.randn(12,7,5)
        out,report=match_direction(u,u,u)
        self.assertTrue(torch.equal(out,u));self.assertEqual(report['kind'],'exact_identity')
        z=torch.zeros_like(u);self.assertTrue(torch.equal(match_direction(z,z,z)[0],z))
    def test_degenerate_refusal(self):
        u=torch.randn(12,7,5);t=target_with_angle(u,.1)
        with self.assertRaises(DegenerateDirection):match_direction(u,t,u)
    def test_nonfinite_refusal(self):
        u=torch.randn(4,3,5);v=u.clone();v[0,0,0]=float('nan')
        with self.assertRaises(FloatingPointError):match_direction(u,u,v)
    def test_permutation_transport_and_rng(self):
        before=torch.get_rng_state().clone();pm=make_permutations(12,17,147)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(torch.equal(pm,make_permutations(12,17,147)))
        validate_permutations(pm)
        for q,sh in [('gate',(12,17,7)),('up',(12,17,7)),('down',(12,7,17))]:
            u=torch.randn(sh)
            self.assertTrue(torch.equal(transport(transport(u,pm,q),pm,q,inverse=True),u))
    def test_hidden_unit_function(self):
        pm=make_permutations(4,17,147)
        g,u,d=[torch.randn(sh,dtype=torch.float64) for sh in ((4,17,7),(4,17,7),(4,7,17))]
        x=torch.randn(3,7,dtype=torch.float64)
        pg,pu,pd=[transport(t,pm,q) for t,q in [(g,'gate'),(u,'up'),(d,'down')]]
        for role in range(4):
            y=torch.nn.functional.silu(x@g[role].T)*(x@u[role].T);y=y@d[role].T
            yy=torch.nn.functional.silu(x@pg[role].T)*(x@pu[role].T);yy=yy@pd[role].T
            torch.testing.assert_close(y,yy,atol=1e-12,rtol=1e-12)
    def test_large_integer_sample_index(self):
        for n in (1,50257*768,2**24+1,2**27+3):
            k=min(32,n)
            idx=torch.zeros(1,dtype=torch.int64) if k==1 else torch.arange(k,dtype=torch.int64)*(n-1)//(k-1)
            self.assertGreaterEqual(int(idx.min()),0);self.assertLess(int(idx.max()),n)
            self.assertEqual(int(idx[-1]),n-1)
    def test_statistics(self):
        from cms_analysis import paired
        z=paired([-.001,-.002,-.003,-.004,-.005,-.006])
        self.assertAlmostEqual(z['mean'],-.0035);self.assertEqual(z['favorable'],6)
        self.assertLess(z['ci975_bonferroni_two'][0],z['ci95'][0])
    def test_integrity_and_transaction(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);f=root/'hello.py';f.write_text('pass\n')
            io.verify_manifest(root,{'hello.py':io.sha(f)})
            f.write_text('fail\n')
            with self.assertRaises(RuntimeError):io.verify_manifest(root,{'hello.py':'0'*64})
            payload=dict(step=1,contract={'fixed':1},value=torch.tensor([1.]))
            first=io.commit_checkpoint(root/'trial',payload,torch,{'test':'first'})
            obj,meta=io.load_checkpoint(root/'trial',torch);self.assertEqual(meta,first)
            second=io.commit_checkpoint(root/'trial',dict(payload,step=2),torch,{'test':'second'})
            self.assertFalse((root/'trial'/first['path']).exists())
            self.assertTrue((root/'trial'/second['path']).exists())


def integration(project):
    """Run the real archived model at tiny geometry, with the real 200-step onset."""
    from cms_train import train_job
    from cms_analysis import compare_jobs
    rt.configure(1)
    with tempfile.TemporaryDirectory(prefix='cms1_cpu_',dir=project) as td:
        td=Path(td);data=td/'synthetic';data.mkdir();g=np.random.default_rng(813)
        for name in ('train.bin','val.bin'):g.integers(0,97,50000,dtype=np.uint16).tofile(data/name)
        di=dict(path=str(data),hashes={n:io.sha(data/n) for n in ('train.bin','val.bin')},
                stat={n:io.stamp(data/n) for n in ('train.bin','val.bin')},label='SYNTHETIC CPU TEST ONLY')
        common=td/'common';reference=td/'reference_parent'
        train_job(project,common,127,'adamw',200,tiny=True,gate=True,data_identity=di)
        train_job(project,reference,127,'reference_adamw',200,tiny=True,gate=True,data_identity=di)
        checks={'prefix_vs_frozen_adam':compare_jobs(common,reference,reference=True)}
        for arm in p.ARMS:
            a,b=td/(arm+'_a'),td/(arm+'_b')
            train_job(project,a,127,arm,214,parent=common,tiny=True,gate=True,data_identity=di)
            if arm=='scrambled_matched':
                first=train_job(project,b,127,arm,214,parent=common,tiny=True,gate=True,data_identity=di,interrupt_at=207)
                assert first['state']=='test_interrupted'
                train_job(project,b,127,arm,214,parent=common,tiny=True,gate=True,data_identity=di,resume=True)
            else:train_job(project,b,127,arm,214,parent=common,tiny=True,gate=True,data_identity=di)
            checks[arm+'_repeat_resume']=compare_jobs(a,b)
        for arm,reference_mode in [('adamw','reference_adamw'),('bdt','reference_bdt')]:
            d=td/reference_mode
            train_job(project,d,127,reference_mode,214,parent=common,tiny=True,gate=True,data_identity=di)
            checks[arm+'_vs_frozen']=compare_jobs(td/(arm+'_a'),d,reference=True)
        # A mismatched comparison must fail rather than issue a false PASS.
        try:compare_jobs(td/'adamw_a',td/'bdt_a')
        except AssertionError:checks['negative_control_refusal']=True
        else:raise AssertionError('Unequal active policies were reported identical')
        return checks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project');ap.add_argument('--output');args=ap.parse_args()
    rt.configure(1)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GeometryTests))
    if not result.wasSuccessful():raise SystemExit(1)
    report=dict(unit_tests=result.testsRun,unit_passed=True,torch=torch.__version__,device='cpu',time=io.now())
    if args.project:report['actual_backend_integration']=integration(Path(args.project).resolve())
    if args.output:io.atomic_json(args.output,report)
    print('CMS1 CPU TESTS: PASS',json.dumps(report,sort_keys=True),flush=True)

if __name__=='__main__':main()
