"""ART1: fixed ancestry allocation, optional D-only refinement, optional attention BDT."""
import copy,types
import torch
import common as io
from gct_runtime import cpu,treehash,samplehash
from legacy_scope_policy import Policy as Legacy,PROJS
from attention_reference import Policy as AttentionReference
from sector_reference import SectorConditionedBank
from selective_reference import selective_output
from fixed_rule import transform as fixed_transform
MODES={'warmup','adamw','fixed','difference','attention','assembled','combined'}

class DifferenceBank:
    def __init__(self,canonical):
        r=canonical.roles
        self.order=torch.tensor(list(range(0,r,2))+list(range(1,r,2)),device=canonical.basis.device)
        self.inverse=torch.argsort(self.order)
        # Permute the ambient coordinates, not covariance eigen-coordinates.
        view=types.SimpleNamespace(roles=r,config=canonical.config,basis=canonical.basis.index_select(0,self.order),
            covariance=canonical.covariance,last_step=canonical.last_step,observations=canonical.observations)
        self.inner=SectorConditionedBank(view,False,2.)
    def state_dict(self):return dict(kind='ART1_adjacent_SFL1_D_only',inner=self.inner.state_dict())
    def load_state_dict(self,s):
        if set(s)!={'kind','inner'} or s['kind']!='ART1_adjacent_SFL1_D_only':raise RuntimeError('D-bank format')
        self.inner.load_state_dict(s['inner'])
    @torch.no_grad()
    def transform(self,u,phase,capture=False):
        x=u.index_select(0,self.order);self.inner.capture=capture
        # Original SCB1 statistic update, with its output inactive. Only D is applied.
        self.inner.transform(x,phase,active=False)
        v,rec=selective_output(self.inner,x,phase,(False,True))
        return v.index_select(0,self.inverse),dict(rec,ancestry='adjacent',mean_sector_conditioning=False)

class Policy(Legacy):
    def __init__(self,model,cfg,mods,mode,device,seed,post=False,age_offset=0):
        if mode not in MODES or not post:raise ValueError('ART1 is a post-growth continuation only')
        super().__init__(model,cfg,mods,'adamw',device,seed,post,age_offset)
        self.mode='reference_adamw';self.applied_arm='adamw';self.art_mode=mode
        self.attn_active=mode in {'attention','assembled','combined'}
        self.attention_groups={q:[getattr(b.attn,k).weight for b in model.blocks] for q,k in [('attention_qkv','qkv'),('attention_out','out')]}
        if any(self.names[id(p)] not in self.rest_names for ps in self.attention_groups.values() for p in ps):raise RuntimeError('Attention partition changed')
        tc=mods['temper'].TemperConfig(200,200,10,.9,3.)
        self.attention={q:mods['temper'].OnlineDepthTemper(len(model.blocks),device,torch.float32,tc) for q in self.attention_groups}
        self.difference={};self.full={};self.attention_summary={'active_projection_steps':0,'max_norm_error':0.,'max_condition':1.}
    def extra(self):
        return dict(format='ART1_warm_observers_v1',attention={q:t.state_dict() for q,t in self.attention.items()},
          difference={q:t.state_dict() for q,t in self.difference.items()},full={q:t.state_dict() for q,t in self.full.items()},attention_summary=copy.deepcopy(self.attention_summary))
    def state(self):
        s=super().state();s['art']=self.extra();return cpu(s)
    def identity(self):return dict(super().identity(),art=treehash(self.extra()))
    def load_old_grown(self,s):
        if 'art' in s or s['phase']!=0:raise RuntimeError('Only exact old immediate-grown input allowed')
        super().load(s)
        if any(t.last_step or t.observations for t in self.attention.values()):raise RuntimeError('Nonzero new observer')
    def load(self,s):
        if 'art' not in s:raise RuntimeError('Missing explicit ART1 prefix adaptation')
        super().load(s);x=s['art']
        if set(x)!={'format','attention','difference','full','attention_summary'} or x['format']!='ART1_warm_observers_v1':raise RuntimeError('ART1 schema')
        if set(x['attention'])!=set(self.attention):raise RuntimeError('Missing attention history')
        for q,t in self.attention.items():t.load_state_dict(x['attention'][q])
        self.difference={};self.full={}
        if x['difference']:
            if self.art_mode not in {'difference','assembled'} or set(x['difference'])!=set(PROJS):raise RuntimeError('D mode/state conflict')
            self.difference={q:DifferenceBank(t) for q,t in self.canonical.items()}
            for q,t in self.difference.items():t.load_state_dict(x['difference'][q])
        if x['full']:
            if self.art_mode!='combined' or set(x['full'])!=set(PROJS):raise RuntimeError('Full mode/state conflict')
            self.full={q:copy.deepcopy(t) for q,t in self.canonical.items()}
            for q,t in self.full.items():t.load_state_dict(x['full'][q])
        self.attention_summary=copy.deepcopy(x['attention_summary'])
        if self.phase>200:
            expected=self.phase if self.attn_active else 200
            if any(t.last_step!=expected for t in self.attention.values()):raise RuntimeError('Attention observer age')
            if self.art_mode in {'difference','assembled'} and any(t.inner.last_step!=self.phase for t in self.difference.values()):raise RuntimeError('D observer age')
            if self.art_mode=='combined' and any(t.last_step!=self.phase for t in self.full.values()):raise RuntimeError('Full observer age')
    @torch.no_grad()
    def step(self,lr,phase,full=False,sampled=False):
        if phase!=self.phase+1 or self.base.step_num!=self.offset+phase-1:raise RuntimeError('ART1 phase/Adam age')
        if self.art_mode=='warmup' and phase>200:raise RuntimeError('Warm reconstruction exceeded200')
        before={q:[p.detach().clone() for p in ps] for q,ps in self.attention_groups.items()} if self.attn_active and phase>200 else {}
        if phase<=200:
            # Exact original AGT1 prefix, including original observer histories.
            rec=Legacy.step(self,lr,phase,full=full,sampled=sampled)
        else:
            if phase==201:
                if self.art_mode in {'difference','assembled'}:self.difference={q:DifferenceBank(t) for q,t in self.canonical.items()}
                if self.art_mode=='combined':self.full={q:copy.deepcopy(t) for q,t in self.canonical.items()}
            if self.art_mode in {'difference','assembled'} and set(self.difference)!=set(PROJS):raise RuntimeError('No warm D history')
            if self.art_mode=='combined' and set(self.full)!=set(PROJS):raise RuntimeError('No warm full history')
            raw=self.base.propose(lr);out={};metrics={}
            for q,u in raw.items():
                if self.art_mode=='adamw':v=u;report={'identity':True}
                elif self.art_mode in {'fixed','attention'}:v,report=fixed_transform(u,phase,full)
                elif self.art_mode in {'difference','assembled'}:v,report=self.difference[q].transform(u,phase,full)
                else:v,report=self.full[q].transform(u,phase,active=True)
                out[q]=v;metrics[q]=report
            rec=dict(raw=treehash(raw) if full else None,applied=treehash(out) if full else None,
              raw_sample=samplehash(sorted(raw.items())) if sampled else None,applied_sample=samplehash(sorted(out.items())) if sampled else None,operators=metrics)
            self.base.apply(lr,out)
            for g in self.rest.param_groups:g['lr']=lr
            self.rest.step();self.phase=phase
        # Shadow through200 reconstructs the original attention warmup without changing weights.
        if phase<=200 or self.attn_active:
            raw=AttentionReference.attention_proposals(self,lr);reports={};applied={}
            for q,u in raw.items():
                v,r=self.attention[q].transform(u,phase,active=True)
                if abs(r['centered_norm_ratio']-1)>5e-5 or r['operator_condition']>3.00005:raise RuntimeError('Attention invariant')
                changed=self.attn_active and phase>200 and not torch.equal(v,u)
                if phase<=200 and not torch.equal(v,u):raise RuntimeError('Attention active in prefix')
                if changed:
                    for i,p in enumerate(self.attention_groups[q]):p.copy_(before[q][i]).mul_(1-float(lr)*self.base.weight_decay).add_(v[i])
                applied[q]=v if changed else u;reports[q]=dict(r,applied=changed)
                self.attention_summary['active_projection_steps']+=int(changed)
                self.attention_summary['max_norm_error']=max(self.attention_summary['max_norm_error'],abs(r['centered_norm_ratio']-1))
                self.attention_summary['max_condition']=max(self.attention_summary['max_condition'],float(r['operator_condition']))
            if full:
                rec['attention_raw']=treehash(raw);rec['attention_applied']=treehash(applied);rec['operators']['attention']=reports
        return rec
