"""Actual pinned-backend CPU integration. Tiny geometry, never scientific seeds."""
from __future__ import annotations
import argparse,contextlib,io as textio,json,os,sys,tempfile,time,hashlib,copy
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core

def data_fixture(project):
    import numpy as np
    d=project/'data';d.mkdir(parents=True)
    gen=np.random.default_rng(99021)
    gen.integers(0,97,size=100000,dtype=np.uint16).tofile(d/'train.bin')
    gen.integers(0,97,size=3600000,dtype=np.uint16).tofile(d/'val.bin')
    (d/'metadata.json').write_text('{"fixture": true}\n')
    names=['train.bin','val.bin','metadata.json']
    return dict(path=str(d),hashes={n:core.sha(d/n) for n in names},stat={n:core.stamp(d/n) for n in names},label='CPU_SYNTHETIC_ONLY')

def run(study):
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='gct2_bco1_cpu_') as tmp:
        project=Path(tmp);(project/'runs').mkdir();data=data_fixture(project)
        import torch
        tr,rt,mods=core.engine(project,study);rt.configure(4)
        result=dict(study=study,device='cpu',torch=torch.__version__,source=core.verify()[0])
        if study=='gct2':
            import gct2
            d=core.root(project)/'gates'/'cpu'
            for j in gct2.gate_plan(d,tiny=True):gct2.execute(project,j,data,resume=True)
            result['actual_gate']=gct2.compare_gate(d)
            result['report_evaluator']=gct2.report_gate(project,d,data,tiny=True)
            r=core.verify_completed(d/'bdt_0',dict(seed=127,end=420,gate=True,tiny=True))
            gct2.verify_checkpoint_payload(d/'bdt_0')
            assert r['step']==420
            ck,meta=core.load_ck(d/'bdt_0',torch)
            assert ck['opt']['phase']==420 and ck['opt']['adam']['age']==620
            result['preserved_growth_age']=620
            # Byte identity on saved state, not merely filenames; tampering must stop.
            f=d/'bdt_0'/meta['path'];raw=f.read_bytes();f.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
            try:core.verify_completed(d/'bdt_0')
            except RuntimeError:result['corrupted_checkpoint_refused']=True
            else:raise AssertionError('Corrupt checkpoint accepted')
            f.write_bytes(raw)
            # Final REPORT entry point cannot run before a complete locked cohort.
            try:gct2.report_one(project,161,'bdt',data)
            except (FileNotFoundError,RuntimeError):result['early_report_refused']=True
            else:raise AssertionError('Premature REPORT allowed')
            # Separate counter fixture at the actual scientific Adam age, not a new training result.
            saved,_=core.load_ck(d/'parent',torch)
            old=rt.cpu(saved['opt']);old['phase']=7630;old['adam']['age']=7630
            for state in old['adam']['states'].values():
                if state['kind']=='rest' and state['state']:state['state']['step'].fill_(7630)
            smallcfg=tr.config(mods['native'],data['path'],False,True)
            small=tr.build(mods['native'],smallcfg,127)
            from gct_policy import Policy
            smallopt=Policy(small,smallcfg,mods,'adamw',torch.device('cpu'),127,False)
            small.load_state_dict(saved['model']);smallopt.load(old)
            targetcfg=tr.config(mods['native'],data['path'],True,True);grown=tr.build(mods['native'],targetcfg,127)
            grownopt=Policy(grown,targetcfg,mods,'adamw',torch.device('cpu'),127,True,7630)
            agecheck=tr.copy_growth(small,smallopt,grown,grownopt)
            for param in grown.parameters():param.grad=torch.zeros_like(param)
            grownopt.step(.0006/200,1)
            assert agecheck['grown_age']==7630 and grownopt.base.step_num==7631 and grownopt.phase==1
            result['scientific_age_counter_fixture']={'preserved_age':7630,'next_age':7631,'post_phase':1,'scope':'synthetic counter fixture'}
            result['gate_updates']=2560
        else:
            import bco
            d=core.root(project,'bco1')/'sessions'/'cpu'/'validation'
            jobs=[dict(condition='split_adam',end=200,parent=None,directory=str(d/'prefix'),tiny=True)]
            for name,condition in [('reference','reference_bdt'),('candidate','candidate_bdt'),('resumed','candidate_bdt')]:
                if name=='resumed':jobs.append(dict(condition=condition,end=1000,parent=str(d/'prefix'),directory=str(d/name),tiny=True,interrupt=310))
                jobs.append(dict(condition=condition,end=1000,parent=str(d/'prefix'),directory=str(d/name),tiny=True))
            for j in jobs:bco.gate_job(project,j,data,resume=True)
            result['actual_gate']=bco.gate_compare(d)
            mod,ocb,Candidate=bco.components(project)
            snap=ocb.build_state(project,data,12,torch.device('cpu'),tiny=True)
            # Saved snapshot integrity is tested through original OCB reader.
            f=project/'cpu_snapshot.pt';ocb.save_state(f,snap);restored=ocb.read_state(f)
            assert restored['common_state_sha']==snap['common_state_sha']
            result['profile']=bco.profile_reference(project,d.parent/'profile',data,torch.device('cpu'),tiny=True,snapshot=snap)
            # All work/restore methods exercised, but do not emit a fabricated CPU timing result.
            try:bco.measure(project,snap,torch.device('cpu'),'reference_bdt','full_step')
            except RuntimeError:result['cpu_timing_refused']=True
            else:raise AssertionError('CPU performance was substituted')
            raw=f.read_bytes();f.write_bytes(raw+b'tamper')
            try:ocb.read_state(f)
            except RuntimeError:result['snapshot_tamper_refused']=True
            else:raise AssertionError('Corrupt OCB snapshot accepted')
            result['gate_updates']=2600
            # Avoid storing the many profiler aggregate events in unit-test summary.
            result['profile']={k:v for k,v in result['profile'].items() if k not in ('all_events','regions')}
        result.update(state='passed',elapsed_seconds=time.monotonic()-started,completed_at=core.now(),scope='Actual pinned backend at tiny CPU geometry; not a target-H200 result or measured speedup.')
        return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True,choices=['gct2','bco1']);ap.add_argument('--output',type=Path);a=ap.parse_args()
    r=run(a.study)
    if a.output:core.atomic(a.output,r)
    print('ACTUAL CPU INTEGRATION PASS',json.dumps(r,sort_keys=True))
