"""BAA1 implementation core: graded *copy-pair* correspondence, not a cap sweep.

This module is usable on real proposal stacks, but is not a model-training
coordinator. Histories must be accumulated on the branch's raw proposals from
post-growth phase one; a saved covariance in another chart cannot be reused.
"""
from __future__ import annotations
import hashlib,math
from pathlib import Path
import torch
import common as c


def libs():
    m=c.backend();return m['geometry'],m['temper']

def local_seed(parent,draw,label):
    return int.from_bytes(hashlib.sha256(f'BAA1:{parent}:{draw}:{label}'.encode()).digest()[:8],'little')%((1<<63)-1)

def derangement(hidden,generator,also_not=None):
    if hidden<4:raise ValueError('At least four hidden units required')
    ident=torch.arange(hidden)
    for _ in range(10000):
        p=torch.randperm(hidden,generator=generator)
        if bool((p!=ident).all()) and (also_not is None or bool((p!=also_not).all())):return p
    raise RuntimeError('Derangement generation failed; no fallback')

def make_family(pairs,hidden,parent,draw):
    if pairs<2 or hidden<4:raise ValueError('Need two ancestry pairs and four hidden units')
    gen=torch.Generator(device='cpu').manual_seed(local_seed(parent,draw,'charts'))
    rows=[]
    for _ in range(pairs):
        for attempt in range(10000):
            p=derangement(hidden,gen)
            if not any(torch.equal(p,q) for q in rows):rows.append(p);break
        else:raise RuntimeError('Distinct-chart generation failed')
    first=torch.stack(rows)
    sg=torch.Generator(device='cpu').manual_seed(local_seed(parent,draw,'assignment'))
    for attempt in range(10000):
        sigma=torch.randperm(pairs,generator=sg)
        if bool((sigma!=torch.arange(pairs)).all()):break
    else:raise RuntimeError('Assignment derangement generation failed')
    result={}
    for name,assignment in [('preserved',torch.arange(pairs)),('reassigned',sigma)]:
        p=torch.cat([first,first[assignment]]);audit=audit_permutations(p,pairs)
        expected=pairs if name=='preserved' else 0
        if audit['preserved_ancestry_pairs']!=expected or audit['chart_multiplicities']!=[2]*pairs:raise AssertionError('Balanced assignment violated')
        if not audit['all_layers_fully_reindexed']:raise AssertionError('Unpermuted layer')
        result[name]=dict(permutations=p,assignment=assignment.clone(),audit=audit)
    if result['preserved']['audit']['chart_multiset_sha256']!=result['reassigned']['audit']['chart_multiset_sha256']:raise AssertionError('Different chart multiset')
    return result

def audit_permutations(p,pairs):
    geometry,_=libs();geometry.validate_permutations(p)
    if p.shape[0]!=2*pairs:raise ValueError('Incorrect duplicated-block map')
    rows=[tuple(int(v) for v in row) for row in p.detach().cpu().tolist()]
    from collections import Counter
    counts=Counter(rows);agreements=[float((p[j]==p[j+pairs]).double().mean()) for j in range(pairs)]
    identity=torch.arange(p.shape[1],device=p.device);off=[];false_reuse=0
    for i in range(len(p)):
        for j in range(i+1,len(p)):
            if i%pairs!=j%pairs:
                off.append(float((p[i]==p[j]).double().mean()));false_reuse+=int(torch.equal(p[i],p[j]))
    import json
    return dict(pair_coordinate_agreements=agreements,preserved_ancestry_pairs=sum(torch.equal(p[j],p[j+pairs]) for j in range(pairs)),
        false_shared_chart_pairs=false_reuse,chart_multiplicities=sorted(counts.values()),distinct_charts=len(counts),
        chart_multiset_sha256=hashlib.sha256(json.dumps(sorted(rows)).encode()).hexdigest(),
        all_layers_fully_reindexed=bool((p!=identity).all()),mean_unrelated_coordinate_agreement=sum(off)/len(off))

@torch.no_grad()
def raw_ancestry_geometry(raw,pairs):
    if raw.shape[0]!=2*pairs:raise ValueError('Wrong ancestry stack')
    u=raw.reshape(2*pairs,-1).double();norms=torch.linalg.vector_norm(u,dim=1);valid=norms>0
    normalized=u/torch.where(valid,norms,torch.ones_like(norms))[:,None];cosine=normalized@normalized.T
    true=[];other=[]
    for i in range(2*pairs):
        for j in range(i+1,2*pairs):
            if not bool(valid[i] and valid[j]):continue
            (true if i%pairs==j%pairs else other).append(float(cosine[i,j]))
    mt=sum(true)/len(true) if true else None;mo=sum(other)/len(other) if other else None
    matrix=[[float(cosine[i,j]) if bool(valid[i] and valid[j]) else None for j in range(2*pairs)] for i in range(2*pairs)]
    return dict(true_pair_mean_cosine=mt,unrelated_mean_cosine=mo,true_minus_unrelated=mt-mo if mt is not None and mo is not None else None,
        valid_true_pairs=len(true),valid_unrelated_pairs=len(other),zero_norm_layers=int((~valid).sum()),cosines=matrix)

@torch.no_grad()
def layer_geometry(raw,applied):
    if raw.shape!=applied.shape or raw.ndim<2:raise ValueError('Wrong proposal geometry')
    u=raw.reshape(raw.shape[0],-1).double();v=applied.reshape_as(raw).reshape(raw.shape[0],-1).double()
    if not bool(torch.isfinite(u).all() and torch.isfinite(v).all()):raise FloatingPointError('Nonfinite layer diagnostics')
    uc=u-u.mean(0,keepdim=True);vc=v-v.mean(0,keepdim=True)
    n=torch.linalg.vector_norm(u,dim=1);m=torch.linalg.vector_norm(v,dim=1)
    nc=torch.linalg.vector_norm(uc,dim=1);mc=torch.linalg.vector_norm(vc,dim=1)
    ratio=lambda a,b:[float(x/y) if float(y)>0 else None for x,y in zip(a,b)]
    dr=torch.linalg.vector_norm(v-u,dim=1)
    return dict(raw_layer_norms=n.tolist(),applied_layer_norms=m.tolist(),layer_norm_ratios=ratio(m,n),
        centered_layer_norm_ratios=ratio(mc,nc),layer_displacements=dr.tolist(),
        zero_raw_layers=int((n==0).sum()),pooled_centered_norm_ratio=float(torch.linalg.vector_norm(vc)/torch.linalg.vector_norm(uc)) if float(torch.linalg.vector_norm(uc))>0 else None)

@torch.no_grad()
def rotational_direction(raw,applied):
    u=raw.reshape(raw.shape[0],-1).double();v=applied.reshape_as(raw).reshape(raw.shape[0],-1).double()
    u=u-u.mean(0,keepdim=True);d=v-raw.reshape_as(v).double();d=d-d.mean(0,keepdim=True)
    n=u.square().sum()
    if float(n)==0:return None
    d-=((d*u).sum()/n)*u;dn=torch.linalg.vector_norm(d)
    if float(dn)<=1e-7*math.sqrt(float(n)):return None
    return d/dn

class CorrespondenceBank:
    """Same workload for Adam/canonical/all chart policies; fresh local histories."""
    def __init__(self,pairs,hidden,parent,device='cpu',draws=(0,1),dtype=torch.float32):
        self.pairs,self.hidden,self.parent,self.draws=pairs,hidden,parent,tuple(draws)
        self.geometry,temper=libs();self.phase=0;self.dtype=dtype;self.device=torch.device(device)
        self.config=temper.TemperConfig(200,200,10,.9,3.)
        self.permutations={};self.design={}
        for draw in self.draws:
            for kind,v in make_family(pairs,hidden,parent,draw).items():
                name=f'draw{draw}_{kind}'
                self.permutations[name]=v['permutations'].to(device)
                self.design[name]=dict(v['audit'],second_assignment=v['assignment'].tolist())
        self.canonical={q:temper.OnlineDepthTemper(2*pairs,device,dtype,self.config) for q in ('gate','up','down')}
        self.charts={name:{q:temper.OnlineDepthTemper(2*pairs,device,dtype,self.config) for q in self.canonical} for name in self.permutations}
        self.previous={}
    @property
    def arms(self):return ('adamw','bdt',*self.charts)
    @torch.no_grad()
    def step(self,raw,phase,applied_arm,*,diagnostics=False,temporal=False):
        if phase!=self.phase+1 or applied_arm not in self.arms or set(raw)!=set(self.canonical):raise ValueError('Phase/arm/group mismatch')
        outputs={};metrics={};canonical_outputs={}
        for q,u in raw.items():
            can,report=self.canonical[q].transform(u,phase,active=True);canonical_outputs[q]=can
            if abs(report['centered_norm_ratio']-1)>5e-5 or report['operator_condition']>3.00005:raise RuntimeError('Canonical invariant failed')
            actual=u if applied_arm=='adamw' else can if applied_arm=='bdt' else None
            group={'canonical':report,'charts':{}}
            if diagnostics:group['raw_ancestry']=raw_ancestry_geometry(u,self.pairs)
            if temporal:
                direction=rotational_direction(u,can);key=('canonical',q)
                report['consecutive_rotation_cosine']=float((direction*self.previous[key][1]).sum()) if direction is not None and key in self.previous and self.previous[key][0]==phase-1 else None
                if direction is None:self.previous.pop(key,None)
                else:self.previous[key]=(phase,direction.clone())
            for name,chart in self.charts.items():
                p=self.permutations[name]
                z=self.geometry.transport(u,p,q);t,tr=chart[q].transform(z,phase,active=True)
                mapped=self.geometry.transport(t,p,q,inverse=True)
                v,m=self.geometry.match_direction(u,can,mapped)
                if name==applied_arm:actual=v
                if diagnostics:m=dict(m,layer_geometry=layer_geometry(u,v))
                if temporal:
                    direction=rotational_direction(u,v);key=(name,q)
                    if direction is not None and key in self.previous and self.previous[key][0]==phase-1:
                        m['consecutive_rotation_cosine']=float((direction*self.previous[key][1]).sum())
                    else:m['consecutive_rotation_cosine']=None
                    if direction is None:self.previous.pop(key,None)
                    else:self.previous[key]=(phase,direction.clone())
                group['charts'][name]=dict(m,unmatched_operator_condition=tr['operator_condition'])
            if actual is None:raise AssertionError('No applied policy')
            if phase<=200 and not torch.equal(actual,u):raise AssertionError('Nonidentity prefix')
            outputs[q]=actual;metrics[q]=group
        self.phase=phase;self.last_canonical=canonical_outputs;return outputs,metrics
    def state(self):
        # Full previous directions are optional: new training runners must record
        # whether consecutive diagnostics were resumed, never imply continuity
        # across a missing diagnostic window.
        return dict(pairs=self.pairs,hidden=self.hidden,parent=self.parent,draws=self.draws,phase=self.phase,
          permutations={k:v.cpu().clone() for k,v in self.permutations.items()},
          canonical={q:t.state_dict() for q,t in self.canonical.items()},
          charts={n:{q:t.state_dict() for q,t in bank.items()} for n,bank in self.charts.items()})
    def load(self,s):
        if any(s[k]!=getattr(self,k) for k in ('pairs','hidden','parent','draws')):raise ValueError('Correspondence configuration changed')
        if set(s['charts'])!=set(self.charts) or set(s['permutations'])!=set(self.permutations):raise ValueError('Chart set changed')
        for name,p in self.permutations.items():
            if not torch.equal(s['permutations'][name].cpu(),p.cpu()):raise ValueError('Chart identity changed')
        for q,t in self.canonical.items():t.load_state_dict(s['canonical'][q])
        for name,bank in self.charts.items():
            for q,t in bank.items():t.load_state_dict(s['charts'][name][q])
        self.phase=int(s['phase']);self.previous={}
        if any(t.last_step!=self.phase for bank in (self.canonical,*self.charts.values()) for t in bank.values()):raise ValueError('Missing proposal history')
