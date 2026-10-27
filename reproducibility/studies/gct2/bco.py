"""One diagnostics-lite candidate: profile, actual-model equality, then balanced timing."""
from __future__ import annotations
import contextlib, copy, gc, json, math, os, sys, time
from pathlib import Path
import core as io


def components(project):
    model,ocb=io.ocb_modules(project)
    candidate=io.load_module('_bco_candidate',io.HERE/'candidate/diagnostics_lite.py')
    class Lite(candidate.OnlineDepthTemper):
        diagnostics_enabled=False
        def transform(self,u,step,*,active=True):
            return super().transform(u,step,active=active,diagnostics=self.diagnostics_enabled)
    class Candidate(model.LeanPolicy):
        def __init__(self,m,cfg,mods,device):
            super().__init__(m,cfg,mods,'bdt',device,127)
            self.canonical={q:Lite(t.roles,device,t.basis.dtype,t.config) for q,t in self.canonical.items()}
        def step(self,lr,phase,full=False,sampled=False):
            for t in self.canonical.values():t.diagnostics_enabled=False # Hash capture must not turn optional reporting back on.
            return super().step(lr,phase,full=full,sampled=sampled)
    return model,ocb,Candidate


def factory(project,snap,device,condition,profile=False):
    import torch
    mod,ocb,Candidate=components(project);mods=io.backend(project)
    cfg=mods['native'].ExperimentConfig(**snap['cfg']);m=mod.build(mods,cfg,127,device)
    if condition=='candidate_bdt':opt=Candidate(m,cfg,mods,device)
    else:opt=mod.LeanPolicy(m,cfg,mods,'split_ffn' if condition=='split_adam' else 'bdt',device,127)
    if profile:
        inst=io.load_module('_bco_profiled',io.HERE/'candidate/profiled_reference.py')
        opt.canonical={q:inst.OnlineDepthTemper(t.roles,device,t.basis.dtype,t.config) for q,t in opt.canonical.items()}
    ocb.restore(m,opt,snap)
    return m,opt,cfg,mods,mod,ocb

def final_identity(model,opt,mod):
    return dict(model=mod.treehash(model.state_dict()),adam=mod.normalized_hash(model,opt),
                covariance=mod.treehash({q:t.state_dict() for q,t in opt.canonical.items()}),rng=mod.treehash(mod.rng()))


def load_original_snapshot(project,data,device):
    _,ocb=io.ocb_modules(project);path=Path(project)/'runs/optimizer_cost_benchmark_v1/states/dense12.pt'
    if not path.is_file():raise RuntimeError('Missing OCB1 update-500 snapshot. No replacement state is generated.')
    s=ocb.read_state(path)
    from gct_runtime import runtime
    if s['source']!=io.OCB_SOURCE or s['runtime']!=runtime(device) or s['cache']!=data['hashes']:
        raise RuntimeError('OCB1 snapshot source/runtime/cache incompatible')
    if s['layers']!=12 or s['parameters']!=123551232 or s['phase']!=500:raise RuntimeError('Wrong OCB1 engineering state')
    gate=io.read(path.parent.parent/'gate.json')
    if gate['state']!='passed' or gate['source']!=io.OCB_SOURCE:raise RuntimeError('Unverified OCB1 state provenance')
    return s


def profile_reference(project,output,data,device,tiny=False,snapshot=None):
    import torch
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    snap=snapshot if snapshot is not None else load_original_snapshot(project,data,device)
    variants=[]
    for inst in (False,True):
        m,opt,cfg,mods,mod,ocb=factory(project,snap,device,'reference_bdt',profile=inst)
        rows=[]
        for i in range(20):
            mod.backward(m,cfg,mods,device,snap['batch_ring'][i],check=True)
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            rec=opt.step(.0006,501+i,full=True)
            rows.append({k:rec[k] for k in ('raw','applied')})
        variants.append(dict(rows=rows,final=final_identity(m,opt,mod)))
        del m,opt;gc.collect()
    if variants[0]!=variants[1]:raise RuntimeError('Profile instrumentation changed reference arithmetic')
    m,opt,cfg,mods,mod,ocb=factory(project,snap,device,'reference_bdt',profile=True)
    activities=[torch.profiler.ProfilerActivity.CPU]
    if device.type=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True) as prof:
        for i in range(20):
            phase=501+i;kind='refresh' if phase%10==0 else 'ordinary'
            with torch.profiler.record_function('BCO/step/'+kind):
                with torch.profiler.record_function('BCO/forward_backward'):
                    mod.backward(m,cfg,mods,device,snap['batch_ring'][i],check=False)
                with torch.profiler.record_function('BCO/clip'):torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
                with torch.profiler.record_function('BCO/propose'):raw=opt.base.propose(.0006)
                out={}
                for q,u in raw.items():
                    with torch.profiler.record_function('BCO/transform/'+kind):v,rep=opt.canonical[q].transform(u,phase,active=True)
                    with torch.profiler.record_function('BCO/outer_safety'):
                        if abs(rep['centered_norm_ratio']-1)>5e-5 or rep['operator_condition']>3.00005:raise RuntimeError('Profile safety failure')
                    out[q]=v
                with torch.profiler.record_function('BCO/apply_decay'):opt.base.apply(.0006,out)
                with torch.profiler.record_function('BCO/rest_adam'):
                    for g in opt.rest.param_groups:g['lr']=.0006
                    opt.rest.step();opt.phase=phase
        if device.type=='cuda':torch.cuda.synchronize(device)
    prof.export_chrome_trace(str(output/'trace.json'))
    events=[]
    for e in prof.events():
        ancestors=[];a=e.cpu_parent
        while a is not None:ancestors.append(a.name);a=a.cpu_parent
        regime=next((k for k in ('ordinary','refresh') if 'BCO/step/'+k in ancestors or e.name=='BCO/step/'+k),'other')
        events.append(dict(name=e.name,regime=regime,cpu_us=e.cpu_time_total,self_cpu_us=e.self_cpu_time_total,
          device_us=getattr(e,'device_time_total',0.),self_device_us=getattr(e,'self_device_time_total',0.),
          cpu_memory=getattr(e,'cpu_memory_usage',0),device_memory=getattr(e,'device_memory_usage',0)))
    aggregates={}
    for e in events:
        key=e['regime']+' | '+e['name'];a=aggregates.setdefault(key,dict(count=0,cpu_us=0.,self_cpu_us=0.,device_us=0.,self_device_us=0.))
        a['count']+=1
        for k in ('cpu_us','self_cpu_us','device_us','self_device_us'):a[k]+=e[k]
    if final_identity(m,opt,mod)!=variants[0]['final']:raise RuntimeError('Profiled orchestration changed final state')
    summary=dict(state='complete',profile_not_a_speed_sample=True,instrumentation_equivalence='exact',
       common_state_sha=snap['common_state_sha'],source=io.verify()[0],ordinary_steps=18,refresh_steps=[510,520],
       regions={k:v for k,v in aggregates.items() if 'BCO/' in k},all_events=aggregates,
       caveat='CPU waits may expose earlier GPU work; nested/overlapping durations are not additive.',time=io.now())
    io.atomic(output/'profile_summary.json',summary);del m,opt,prof;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return summary


def gate_job(project,j,data,resume=False):
    import torch
    mod,ocb,Candidate=components(project);mods=io.backend(project);tiny=j.get('tiny',False)
    device=torch.device('cpu' if tiny else 'cuda:0');mod.configure(4)
    cfg=mod.config(mods['native'],data['path'],12,3000,peak=.0006,tiny=tiny)
    directory=Path(j['directory']);source=io.verify()[0];condition=j['condition']
    ct=dict(source=source,protocol=io.compact(io.BCO1),stage='bco_gate',mode=condition,seed=127,end=j['end'],
            runtime=mod.runtime(device),cache=data['hashes'],tiny=tiny,gate=True)
    with io.lock(directory/'.writer.lock'):
        if (directory/'fatal.json').exists():raise RuntimeError('Fatal BCO gate failure requires investigation')
        io.immutable(directory/'contract.json',ct)
        if io.verify_completed(directory,ct):return io.read(directory/'result.json')
        own=(directory/'latest.json').exists()
        if own and not resume:raise RuntimeError('Existing gate checkpoint requires --resume')
        start=0;rows=[]
        if own or j.get('parent'):
            ck,meta=io.load_ck(directory if own else j['parent'],torch);snap=ck['snapshot'];start=ck['step']
            if own and ck['contract']!=ct:raise RuntimeError('BCO resume contract changed')
            if not own and start!=200:raise RuntimeError('BCO needs shared step200 parent')
            m,opt,cfg,mods,mod,ocb=factory(project,snap,device,condition)
            if own:rows=ck['rows']
        else:
            m=mod.build(mods,cfg,127,device);opt=mod.LeanPolicy(m,cfg,mods,'observe',device,127)
        store=mods['native'].TokenStore(Path(data['path'])/'train.bin')
        if own and final_identity(m,opt,mod)!=meta['identity']:raise RuntimeError('BCO saved state failed to restore')
        initial=ck['initial'] if own else final_identity(m,opt,mod)
        def save(step):
            snap=ocb.snapshot(m,opt,cfg,127,data)
            return io.save_ck(directory,dict(step=step,contract=ct,snapshot=snap,rows=rows,initial=initial),torch,
                              final_identity(m,opt,mod),keep=step==j['end'])
        try:
            for s in range(start+1,j['end']+1):
                xy=mod.batch(store,cfg,127,s);rb=mod.treehash(mod.rng());loss=mod.backward(m,cfg,mods,device,xy)
                full=s in (1,2,10,50,100,200,201,210,250,300,310,400,420,500,750,1000)
                gradient=mod.treehash({n:p.grad for n,p in m.named_parameters()}) if full else None
                gradient_sample=mod.samplehash((n,p.grad) for n,p in m.named_parameters())
                norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
                if not bool(torch.isfinite(norm)):raise FloatingPointError('BCO gate nonfinite gradient')
                rec=opt.step(mods['native'].learning_rate(cfg,s),s,full=full,sampled=True)
                row=dict(step=s,loss=loss,gradient=gradient,gradient_sample=gradient_sample,raw=rec.get('raw'),applied=rec.get('applied'),raw_sample=rec['raw_sample'],applied_sample=rec['applied_sample'],batch=mod.treehash(xy),
                         rng_before=rb,rng_after=mod.treehash(mod.rng()),parameter_sample=mod.samplehash(m.named_parameters()))
                if full:
                    row['full']=final_identity(m,opt,mod)
                    row['dev']=mod.evaluate(m,Path(data['path'])/'val.bin',cfg,mods['native'],device,tiny)
                rows.append(row)
                if s%50==0:print('BCO1 GATE',condition,s,'/',j['end'],flush=True);io.guard_cache(data)
                if s%250==0 or s in (200,j['end'],j.get('interrupt')):save(s)
                if s==j.get('interrupt'):return dict(state='test_paused',step=s)
            meta=io.read(directory/'latest.json')
            r=dict(state='complete',step=j['end'],contract=ct,contract_sha=io.digest(ct),initial=initial,final=meta['identity'],checkpoint=meta,rows=rows)
            io.atomic(directory/'result.json',r);return r
        except torch.OutOfMemoryError:raise
        except Exception as e:io.atomic(directory/'fatal.json',dict(error=str(e),time=io.now()));raise


def gate_compare(directory):
    d=Path(directory);rs={n:io.read(d/n/'result.json') for n in ('reference','candidate','resumed')}
    ref=rs['reference']
    if not any(r['raw']!=r['applied'] for r in ref['rows']):raise RuntimeError('Candidate gate treatment was inactive')
    for name,r in rs.items():
        if r['initial']!=ref['initial'] or r['rows']!=ref['rows'] or r['final']!=ref['final']:
            raise RuntimeError('BCO active gate differs: '+name)
    return dict(state='passed',steps=800,sampled_gradients_proposals_parameters_each_step=True,full_tensor_checks=[201,210,250,300,310,400,420,500,750,1000],
                checks=['reference_vs_candidate','candidate_interrupted_at310_resume','final_model_adam_covariance_rng_exact'],
                evidence={n:io.sha(d/n/'result.json') for n in rs},time=io.now())


def measure(project,snap,device,condition,family):
    import torch
    if device.type!='cuda':raise RuntimeError('Timing requires CUDA; no CPU timing substituted')
    m,opt,cfg,mods,mod,ocb=factory(project,snap,device,condition)
    count=20 if family=='full_step' else 100
    def work(n):
        for i in range(n):
            if family=='full_step':
                mod.backward(m,cfg,mods,device,snap['batch_ring'][i%20],check=False)
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            opt.step(.0006,501+i,full=False)
    if family=='optimizer_only':ocb.apply_fixed_gradients(m,snap,device)
    e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True)
    e0.record();e1.record();torch.cuda.synchronize(device);work(20);torch.cuda.synchronize(device)
    ocb.restore(m,opt,snap)
    if family=='optimizer_only':ocb.apply_fixed_gradients(m,snap,device)
    before={q:t.observations for q,t in opt.canonical.items()};torch.cuda.synchronize(device)
    baseline=torch.cuda.memory_allocated(device);reserved0=torch.cuda.memory_reserved(device);torch.cuda.reset_peak_memory_stats(device)
    at=time.perf_counter();e0.record();work(count);e1.record();torch.cuda.synchronize(device);wall=1000*(time.perf_counter()-at)
    event=e0.elapsed_time(e1);peak=torch.cuda.max_memory_allocated(device);reserved=torch.cuda.max_memory_reserved(device)
    # All validation follows the timing boundary.
    deltas=[t.observations-before[q] for q,t in opt.canonical.items()];expected=0 if condition=='split_adam' else count//10
    if (deltas and deltas!=[expected]*3) or (expected and not deltas):raise RuntimeError('Timing covariance cadence changed')
    if not all(math.isfinite(x) and x>0 for x in (wall,event)):raise RuntimeError('Invalid timing')
    state=mod.normalized(m,opt)
    if any(s['step']!=500+count or not torch.isfinite(s['m']).all() or not torch.isfinite(s['v']).all() for s in state.values()):raise RuntimeError('Invalid timed Adam state')
    f=final_identity(m,opt,mod)
    result=dict(condition=condition,updates=count,start=501,end=500+count,restored_after_warmup=True,synchronized=True,
                profiler=False,shadow_matching=False,audit_hashes_inside_timer=False,common_state_sha256=snap['common_state_sha'],
                input_sha256=snap['input_sha'],gradient_sha256=snap['gradient_sha'],observations_per_projection=expected,
                refresh_deltas=deltas,wall_ms=wall,cuda_ms=float(event),final_model_sha256=f['model'],final_adam_sha256=f['adam'],
                final_covariance_sha256=f['covariance'],final_rng_sha256=f['rng'],baseline_allocated_bytes=baseline,baseline_reserved_bytes=reserved0,
                peak_allocated_bytes=peak,peak_reserved_bytes=reserved,incremental_peak_bytes=peak-baseline)
    del m,opt,state;gc.collect();torch.cuda.empty_cache();return result


def run_timing(project,directory,data,gpu_uuid,telemetry):
    import torch
    d=Path(directory);device=torch.device('cuda:0');_,rt,_=io.engine(project,'bco1');rt.configure(4)
    snap=load_original_snapshot(project,data,device);gate=io.read(d/'gate.json')
    if gate['state']!='passed' or gate['source']!=io.verify()[0] or gate['runtime']!=rt.runtime(device):raise RuntimeError('No compatible candidate equivalence gate')
    for rel,h in gate['evidence'].items():
        if io.sha(io.safe_file(io.root(project,'bco1'),rel))!=h:raise RuntimeError('Candidate gate evidence changed')
    session=io.read(d/'SESSION.json');results={}
    sys.path.insert(0,str(io.HERE/'design/tools'));from analyze_bco1 import analyze
    for family in io.BCO1['timing']['families']:
        blocks=[]
        for index,order in enumerate(io.BCO1['timing']['orders']):
            path=d/'blocks'/family/f'{index:02d}.json';partial=path.with_suffix('.partial.json')
            if path.exists():
                b=io.read(path)
                if b['gpu_uuid']!=gpu_uuid or b['source_sha256']!=io.verify()[0] or b['runtime_sha256']!=io.compact(rt.runtime(device)):raise RuntimeError('Mixed timing session')
                blocks.append(b);continue
            if partial.exists():raise RuntimeError('Partial timing block retained; explicit review required, never select fastest repetitions')
            io.guard_cache(data);io.verify();before=telemetry()
            if before['uuid']!=gpu_uuid:raise RuntimeError('Timing GPU changed')
            b=dict(state='running',protocol_digest=io.compact(io.BCO1),family=family,block=index,gpu_uuid=gpu_uuid,
                   runtime_sha256=io.compact(rt.runtime(device)),source_sha256=io.verify()[0],exact_equivalence_gate=True,
                   measurements=[],started=io.now(),telemetry_before=before,session_sha=io.sha(d/'SESSION.json'))
            io.atomic(partial,b)
            for name in ['sentinel_before',*order,'sentinel_after']:
                print('BCO1 MEASURE',family,index,name,flush=True)
                io.atomic(d/'progress.json',dict(family=family,block=index,condition=name,time=io.now()))
                r=measure(project,snap,device,'split_adam' if name.startswith('sentinel_') else name,family)
                if name.startswith('sentinel_'):b[name]=r
                else:b['measurements'].append(r)
                io.atomic(partial,b)
            b['telemetry_after']=telemetry()
            if b['telemetry_after']['uuid']!=gpu_uuid:raise RuntimeError('GPU changed within block')
            paired={r['condition']:r for r in b['measurements']}
            for field in ('final_model_sha256','final_adam_sha256','final_covariance_sha256','final_rng_sha256'):
                if paired['reference_bdt'][field]!=paired['candidate_bdt'][field]:raise RuntimeError('Timed candidate differs from reference')
            b.update(state='complete',finished=io.now());io.atomic(path,b);partial.unlink();blocks.append(b)
        
        for block in blocks:
            for r in [block['sentinel_before'],*block['measurements'],block['sentinel_after']]:
                if r['input_sha256']!=snap['input_sha'] or r['gradient_sha256']!=snap['gradient_sha'] or r['common_state_sha256']!=snap['common_state_sha']:raise RuntimeError('Restored timing inputs changed')
        results[family]=analyze(blocks,family)
    io.atomic(d/'analysis.json',results)
    (d/'READOUT.md').write_text('# BCO1 diagnostics-lite candidate\n\n'+json.dumps(results,indent=2)+'\n\nPaired shared-GPU implementation measurements, not time-to-quality.\n')
    return results
