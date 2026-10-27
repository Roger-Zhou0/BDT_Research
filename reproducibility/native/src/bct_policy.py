"""BCT1: two hypothetical FFN operators, one gradient and one committed Adam step."""
import copy
import torch
import common as io
from carrier import smooth_window
from gct_runtime import cpu,treehash,samplehash
from art_policy import Policy as ArtPolicy,DifferenceBank,PROJS
from legacy_scope_policy import Policy as Legacy
from attention_reference import Policy as AttentionReference
from bct_geometry import exchange,discrepancy
MODES={'warmup','reference_bdt','reference_structured','bdt_budget_gs','structured_budget_gs'}

class Policy(ArtPolicy):
    def __init__(self,model,cfg,mods,mode,device,seed,post=False,age_offset=0):
        if mode not in MODES:raise ValueError('Unregistered BCT1 policy')
        super().__init__(model,cfg,mods,'warmup' if mode=='warmup' else 'assembled',device,seed,post,age_offset)
        self.bct_mode=mode
        self.crossover_summary=dict(projection_steps=0,fallback_sectors=0,max_total_norm_error=0.,max_sector_budget_error=0.,max_common_error=0.)
    def bct_state(self):return dict(format='BCT1_dual_own_state_v1',mode=self.bct_mode,carrier=copy.deepcopy(io.SPEC['carrier']),summary=copy.deepcopy(self.crossover_summary))
    def state(self):
        s=super().state()
        if self.phase>200:s['bct']=self.bct_state()
        return s
    def identity(self):
        s=super().identity()
        if self.phase>200:s['bct']=treehash(self.bct_state())
        return s
    def load(self,s):
        if 'art' not in s:raise RuntimeError('Exact warm ART1 prefix required')
        Legacy.load(self,s);x=s['art']
        if set(x)!={'format','attention','difference','full','attention_summary'} or x['format']!='ART1_warm_observers_v1':raise RuntimeError('ART state schema')
        if set(x['attention'])!=set(self.attention):raise RuntimeError('Attention observer missing')
        for q,t in self.attention.items():t.load_state_dict(x['attention'][q])
        self.attention_summary=copy.deepcopy(x['attention_summary']);self.difference={};self.full={}
        if self.phase<=200:
            if 'bct' in s or x['difference'] or x['full']:raise RuntimeError('Expected unmodified pre-treatment state')
        else:
            a=s.get('bct',{})
            if a.get('format')!='BCT1_dual_own_state_v1' or a.get('mode')!=self.bct_mode or a.get('carrier')!=io.SPEC['carrier']:raise RuntimeError('Carrier mode/state/configuration mismatch')
            if set(x['difference'])!=set(PROJS) or set(x['full'])!=set(PROJS):raise RuntimeError('Both own-state histories are required')
            self.crossover_summary=copy.deepcopy(a['summary'])
            self.difference={q:DifferenceBank(t) for q,t in self.canonical.items()}
            self.full={q:copy.deepcopy(t) for q,t in self.canonical.items()}
            for q,t in self.difference.items():t.load_state_dict(x['difference'][q])
            for q,t in self.full.items():t.load_state_dict(x['full'][q])
            if any(t.inner.last_step!=self.phase for t in self.difference.values()) or any(t.last_step!=self.phase for t in self.full.values()):raise RuntimeError('Candidate history ages differ')
        if any(t.last_step!=self.phase for t in self.attention.values()):raise RuntimeError('Warm attention age differs')
    def candidates(self,raw,phase,capture=False,reverse=False):
        b={};s={};reports={}
        for q,u in raw.items():
            if reverse:
                s[q],sr=self.difference[q].transform(u,phase,capture)
                b[q],br=self.full[q].transform(u,phase,active=True)
            else:
                b[q],br=self.full[q].transform(u,phase,active=True)
                s[q],sr=self.difference[q].transform(u,phase,capture)
            reports[q]=dict(bdt=br,structured=sr)
        return b,s,reports
    @torch.no_grad()
    def step(self,lr,phase,full=False,sampled=False):
        if phase<=200:return super().step(lr,phase,full=full,sampled=sampled)
        if self.bct_mode=='warmup' or phase!=self.phase+1 or self.base.step_num!=self.offset+phase-1:raise RuntimeError('BCT1 phase/Adam age mismatch')
        if phase==201:
            self.difference={q:DifferenceBank(t) for q,t in self.canonical.items()}
            self.full={q:copy.deepcopy(t) for q,t in self.canonical.items()}
        if set(self.difference)!=set(PROJS) or set(self.full)!=set(PROJS):raise RuntimeError('Missing warm candidate histories')
        before={q:[p.detach().clone() for p in ps] for q,ps in self.attention_groups.items()}
        raw=self.base.propose(lr)
        b,s,reports=self.candidates(raw,phase,full)
        out={}
        for q,u in raw.items():
            if self.bct_mode=='reference_bdt':v=b[q];mix=None
            elif self.bct_mode=='reference_structured':v=s[q];mix=None
            else:
                g=smooth_window(u,phase,io.SPEC['carrier']['alpha'],io.SPEC['onset'],io.SPEC['ramp'])
                budget=b[q] if self.bct_mode=='bdt_budget_gs' else s[q]
                v,mix=exchange(u,budget,g,full)
                if full:
                    reports[q]['carrier']={'family':'ramped_standard_window_GS','alpha':io.SPEC['carrier']['alpha'],'phase':phase}
                    reports[q]['carrier_versus_bdt']=discrepancy(b[q]-u.mean(0,keepdim=True),g-u.mean(0,keepdim=True))
                    reports[q]['carrier_versus_structured']=discrepancy(s[q]-u.mean(0,keepdim=True),g-u.mean(0,keepdim=True))
            out[q]=v
            if mix:
                self.crossover_summary['projection_steps']+=1
                self.crossover_summary['fallback_sectors']+=len(mix['fallback_sectors'])
                for k in ['total_norm_error','sector_budget_error','common_error']:
                    self.crossover_summary['max_'+k]=max(self.crossover_summary['max_'+k],mix[k])
            if full:reports[q].update(crossover=mix,proposal_discrepancy=discrepancy(b[q]-u.mean(0,keepdim=True),s[q]-u.mean(0,keepdim=True)))
        rec=dict(raw=treehash(raw) if full else None,applied=treehash(out) if full else None,
            raw_sample=samplehash(sorted(raw.items())) if sampled else None,applied_sample=samplehash(sorted(out.items())) if sampled else None,operators=reports)
        self.base.apply(lr,out)
        for g in self.rest.param_groups:g['lr']=lr
        self.rest.step();self.phase=phase
        # Same attention rule and native-moment update order as ART1, applied exactly once.
        att_raw=AttentionReference.attention_proposals(self,lr);att_reports={};att_applied={}
        for q,u in att_raw.items():
            v,r=self.attention[q].transform(u,phase,active=True)
            if abs(r['centered_norm_ratio']-1)>5e-5 or r['operator_condition']>3.00005:raise RuntimeError('Attention invariant')
            changed=not torch.equal(v,u)
            if changed:
                for i,p in enumerate(self.attention_groups[q]):p.copy_(before[q][i]).mul_(1-float(lr)*self.base.weight_decay).add_(v[i])
            att_applied[q]=v if changed else u;att_reports[q]=dict(r,applied=changed)
            self.attention_summary['active_projection_steps']+=int(changed)
            self.attention_summary['max_norm_error']=max(self.attention_summary['max_norm_error'],abs(r['centered_norm_ratio']-1))
            self.attention_summary['max_condition']=max(self.attention_summary['max_condition'],float(r['operator_condition']))
        if full:
            rec['attention_raw']=treehash(att_raw);rec['attention_applied']=treehash(att_applied);rec['operators']['attention']=att_reports
        return rec
