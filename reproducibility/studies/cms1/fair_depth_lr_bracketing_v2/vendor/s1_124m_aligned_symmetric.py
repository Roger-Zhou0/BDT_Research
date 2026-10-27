#!/usr/bin/env python3
"""V6.18 — 12.5/87.5 shared + round-private FFN parity experiment.

Keep the proven V6.5 ingredients (8 full round-specific attention modules,
round embeddings, fixed maximally-separated sparse SwiGLU routing), but split
FFN capacity into:
  * shared routed sparse path: 974 stored units, 128 active units;
  * round-private path: one 896-wide SwiGLU for each of 8 rounds.

Thus every round executes exactly 128+896=1024 SwiGLU hidden units, identical
to V6.5 active FFN matrix compute. Stored FFN parameters are within 144 params
of V6.5's FFN allocation; total model remains expected 33,459,792 parameters, 1,200
below the dense Transformer. This changes only the shared/private allocation.
"""
from __future__ import annotations
import argparse,hashlib,json
from dataclasses import asdict,dataclass
from pathlib import Path
from typing import Any,Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import validation_v6_1 as v61
import validation_v6_3 as v63

VERSION = 's1_124m_aligned_symmetric_k11_v1'
CONDITION = 's1_aligned_symmetric_k11'
DENSE='dense_stack'
V65='recurrent_sparse_roundaware_full_attn_parity8092'
SHARED_BANK = 3016
SHARED_ACTIVE = 256
PRIVATE_HIDDEN = 1792
FULLRANK_BASIS_K = 11

class FullRankBasisRoundSwiGLU(nn.Module):
    """
    One shared 896-wide SwiGLU coordinate system plus K reusable FULL-RANK
    weight-basis directions.

    Round t synthesizes:
        W_t = W_base + sum_k alpha[t,k] * B_k

    Gate/up/down have independent coefficients.

    All alpha coefficients start at zero, so every round starts from the
    exact same FFN coordinate system. Round differentiation must emerge
    through combinations of the shared full-rank bases.
    """
    def __init__(self, d_model, hidden, n_rounds, n_basis):
        super().__init__()

        self.d_model = d_model
        self.hidden = hidden
        self.n_rounds = n_rounds
        self.n_basis = n_basis

        # Shared base SwiGLU.
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

        # Full-rank reusable basis matrices.
        self.gate_basis = nn.Parameter(
            torch.empty(n_basis, hidden, d_model)
        )
        self.up_basis = nn.Parameter(
            torch.empty(n_basis, hidden, d_model)
        )
        self.down_basis = nn.Parameter(
            torch.empty(n_basis, d_model, hidden)
        )

        # Tiny round-specific coordinates only.
        self.gate_alpha = nn.Parameter(
            torch.zeros(n_rounds, n_basis)
        )
        self.up_alpha = nn.Parameter(
            torch.zeros(n_rounds, n_basis)
        )
        self.down_alpha = nn.Parameter(
            torch.zeros(n_rounds, n_basis)
        )

        self.reset_basis()

    def reset_basis(self):
        # Basis scale comparable to normal Transformer matrix init.
        nn.init.normal_(self.gate_basis, mean=0.0, std=0.02)
        nn.init.normal_(self.up_basis, mean=0.0, std=0.02)
        nn.init.normal_(self.down_basis, mean=0.0, std=0.02)

        # Exact round alignment at initialization.
        nn.init.zeros_(self.gate_alpha)
        nn.init.zeros_(self.up_alpha)
        nn.init.zeros_(self.down_alpha)

    def _combine(self, base, basis, alpha):
        # alpha: [K]
        # basis: [K, out, in]
        return base + torch.einsum("k,koi->oi", alpha, basis)

    def forward(self, x, round_idx):
        r = int(round_idx)

        # CRITICAL:
        # The surrounding training/eval context uses BF16 autocast.
        # torch.einsum is autocast-eligible, so `.float()` alone is
        # insufficient to guarantee FP32 effective-weight synthesis.
        #
        # Construct W_t in genuine FP32 first. Then leave this block;
        # the subsequent linear operations are allowed to autocast
        # normally, exactly like the round-specialized oracle.
        with torch.autocast(
            device_type=x.device.type,
            enabled=False,
        ):
            wg = (
                self.gate.weight.float()
                + torch.einsum(
                    "k,koi->oi",
                    self.gate_alpha[r].float(),
                    self.gate_basis.float(),
                )
            )

            wu = (
                self.up.weight.float()
                + torch.einsum(
                    "k,koi->oi",
                    self.up_alpha[r].float(),
                    self.up_basis.float(),
                )
            )

            wd = (
                self.down.weight.float()
                + torch.einsum(
                    "k,koi->oi",
                    self.down_alpha[r].float(),
                    self.down_basis.float(),
                )
            )

        # These execute under the caller's normal autocast policy.
        # Thus both the oracle and aligned model feed BF16 matmuls
        # weights obtained by quantizing one complete effective matrix.
        g = torch.nn.functional.linear(
            x,
            wg,
            self.gate.bias,
        )

        u = torch.nn.functional.linear(
            x,
            wu,
            self.up.bias,
        )

        z = torch.nn.functional.silu(g) * u

        return torch.nn.functional.linear(
            z,
            wd,
            self.down.bias,
        )

TARGET_DENSE_PARAMS=33_460_992
TARGET_V65_PARAMS=33_459_936
TARGET_V65_MAC=37_001_600
DEFAULT_SEEDS=(41,42,43)
_V61_EVALUATE=v61.evaluate
_V61_LOGICAL_MACS=v61.logical_macs_per_sequence

def maximal_coverage_masks(n_modes:int,bank_hidden:int,active_hidden:int,seed:int)->torch.Tensor:
    total=n_modes*active_hidden
    if total < bank_hidden: raise ValueError('cannot achieve full bank coverage')
    g=torch.Generator(device='cpu').manual_seed(67_000_000+int(seed))
    perm=torch.randperm(bank_hidden,generator=g)
    memberships=perm.repeat((total+bank_hidden-1)//bank_hidden)[:total]
    # Randomize membership order after ensuring every unit appears once before repeats.
    # This preserves the minimum possible total duplicate memberships.
    order=torch.randperm(total,generator=g)
    memberships=memberships.index_select(0,order)
    masks=torch.zeros(n_modes,bank_hidden,dtype=torch.bool)
    for m in range(n_modes):
        idx=memberships[m*active_hidden:(m+1)*active_hidden]
        # Extremely unlikely duplicate within a mode can occur because repeated
        # units are shuffled. Repair deterministically while preserving coverage.
        uniq=torch.unique(idx)
        if uniq.numel()!=active_hidden:
            used=set(int(x) for x in uniq.tolist())
            candidates=[i for i in range(bank_hidden) if i not in used]
            need=active_hidden-uniq.numel()
            idx=torch.cat([uniq,torch.tensor(candidates[:need],dtype=torch.long)])
        masks[m,idx]=True
    # Repair any uncovered units by swapping into modes that contain duplicated-across-mode units.
    uncovered=torch.nonzero(~masks.any(dim=0),as_tuple=False).flatten().tolist()
    if uncovered:
        counts=masks.sum(dim=0)
        for u in uncovered:
            placed=False
            for m in range(n_modes):
                candidates=torch.nonzero(masks[m] & (counts>1),as_tuple=False).flatten()
                if candidates.numel():
                    old=int(candidates[0]); masks[m,old]=False; counts[old]-=1; masks[m,u]=True; counts[u]+=1; placed=True; break
            if not placed: raise RuntimeError('coverage repair failed')
    if not torch.equal(masks.sum(1),torch.full((n_modes,),active_hidden,dtype=torch.long)): raise RuntimeError('bad mask cardinality')
    if not masks.any(0).all(): raise RuntimeError('coverage failed')
    return masks

@dataclass
class ExperimentConfig(v63.ExperimentConfig):
    # S1 — canonical 124M-class scaling geometry.
    d_model: int = 768
    n_heads: int = 12
    n_layers_or_rounds: int = 12
    n_modes: int = 12
    ffn_hidden: int = 2048
    attention_adapter_modes: int = 12

    sparse_bank_hidden:int=SHARED_BANK
    output_root:str='runs/v6_18_fullrank_basis7'
    def __post_init__(self):
        # V6.1 assumes the sparse bank must be >= the full dense FFN width
        # (ffn_hidden=1024). That is not the correct constraint for V6.18:
        # the sparse bank serves only the 128-wide shared branch.
        actual_bank = self.sparse_bank_hidden
        self.sparse_bank_hidden = max(actual_bank, self.ffn_hidden)
        try:
            super().__post_init__()
        finally:
            self.sparse_bank_hidden = actual_bank

        # Correct V6.18 constraint.
        if self.sparse_bank_hidden < SHARED_ACTIVE:
            raise ValueError(
                f"shared bank {self.sparse_bank_hidden} must be >= "
                f"shared active width {SHARED_ACTIVE}"
            )
        if self.d_model!=768 or self.n_layers_or_rounds!=12 or self.n_modes!=12: raise ValueError('S1 freezes D=768, rounds=12, modes=12')
        if self.ffn_hidden!=2048: raise ValueError('S1 active FFN width must be 2048')
        if self.sparse_bank_hidden!=SHARED_BANK: raise ValueError(f'V6.18 shared bank must be {SHARED_BANK}')
    def signature(self,condition:str,seed:int)->str:
        payload={'version':VERSION,'condition':condition,'seed':int(seed),'config':asdict(self),'shared_active':SHARED_ACTIVE,'private_hidden':PRIVATE_HIDDEN}
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

class SharedPrivateBlock(nn.Module):
    def __init__(self,config:ExperimentConfig,*,seed:int):
        super().__init__(); self.config=config
        self.norm1=v61.RMSNorm(config.d_model); self.norm2=v61.RMSNorm(config.d_model)
        self.attn_rounds=nn.ModuleList([v61.CausalSelfAttention(config) for _ in range(config.n_layers_or_rounds)])
        self.router_in=nn.Linear(config.d_model,config.router_hidden,bias=False); self.router_out=nn.Linear(config.router_hidden,config.n_modes,bias=False)
        vals=asdict(config); vals['ffn_hidden']=SHARED_ACTIVE; vals['sparse_bank_hidden']=SHARED_BANK; vals['output_root']=config.output_root
        shared_cfg=v63.ExperimentConfig(**vals)
        masks=maximal_coverage_masks(config.n_modes,SHARED_BANK,SHARED_ACTIVE,seed)
        # Keep the name ffn so existing routing/topology evaluation works.
        self.ffn=v61.SharedSparseSwiGLU(shared_cfg,fixed_masks=masks)
        self.private_ffn=FullRankBasisRoundSwiGLU(
            config.d_model,
            PRIVATE_HIDDEN,
            config.n_layers_or_rounds,
            FULLRANK_BASIS_K,
        )
    def forward(self,x,*,tau:float,stochastic:bool,sparse_eval:bool,round_embedding:torch.Tensor,round_index:int):
        attn_input=self.norm1(x+round_embedding); x=x+self.attn_rounds[round_index](attn_input)
        y=self.norm2(x+round_embedding); logits=self.router_out(F.silu(self.router_in(y))).float()
        if stochastic: route_weights=F.gumbel_softmax(logits,tau=tau,hard=True,dim=-1)
        else:
            route_ids=logits.argmax(-1); route_weights=F.one_hot(route_ids,self.config.n_modes).float()
        route_ids=route_weights.argmax(-1)
        fy=y.reshape(-1,y.shape[-1]); fr=route_weights.reshape(-1,self.config.n_modes); fi=route_ids.reshape(-1)
        shared=self.ffn.forward_sparse_eval(fy,fi) if sparse_eval else self.ffn.forward_dense_train(fy,fr)
        private=self.private_ffn(fy,round_index)
        x=x+(shared+private).view_as(y)
        aux,_p,balance,entropy=v63._router_aux(logits,route_ids,self.config.n_modes,self.config.router_balance_weight,self.config.router_z_weight)
        return x,{'auxiliary_loss':aux,'route_ids':route_ids,'entropy':entropy,'balance':balance}

class SharedPrivateLM(v61.RecurrentSparseLM):
    condition=CONDITION
    def __init__(self,config:ExperimentConfig,*,seed:int):
        v61.BaseLM.__init__(self,config); self.round_aware=True; self.block=SharedPrivateBlock(config,seed=seed); self.round_embedding=nn.Embedding(config.max_recurrent_rounds,config.d_model)
    def forward(self,input_ids,tau=None,stochastic=False,sparse_eval=True,rounds=None,collect_routes=False):
        h=self.token_embedding(input_ids); n=self.config.n_layers_or_rounds if rounds is None else rounds
        if n>self.config.max_recurrent_rounds: raise ValueError('rounds exceeds max')
        tau=self.config.router_tau_end if tau is None else tau; aux=h.new_zeros(()); routes=[]; ents=[]; bals=[]
        for rr in range(n):
            e=self.round_embedding.weight[rr][None,None,:]; idx=rr%len(self.block.attn_rounds)
            h,d=self.block(h,tau=tau,stochastic=stochastic,sparse_eval=sparse_eval,round_embedding=e,round_index=idx); aux=aux+d['auxiliary_loss']
            if collect_routes: routes.append(d['route_ids'].detach()); ents.append(float(d['entropy'])); bals.append(float(d['balance']))
        out={'auxiliary_loss':aux/n}
        if collect_routes: out.update({'route_ids':routes,'router_entropy':ents,'router_balance':bals})
        return self.logits(h),out

def construct_model(config,condition,seed=0):
    if condition!=CONDITION: raise ValueError(condition)
    v61.seed_all(seed); model=SharedPrivateLM(config,seed=seed); v61.initialize_model(model); model.block.private_ffn.reset_basis(); return model

def logical_macs_per_sequence(config,condition):
    if condition!=CONDITION: raise ValueError(condition)
    # Shared 128 + private 896 = exactly V6.5's 1024 active SwiGLU width.
    return _V61_LOGICAL_MACS(config,'recurrent_sparse')

def evaluate(model,store,config,device,seed,rounds=None,collect_routes=False):
    return _V61_EVALUATE(model,store,config,device,seed,rounds=rounds,collect_routes=collect_routes)

def _patch():
    v61.VERSION=VERSION; v61.CONDITIONS=(CONDITION,); v61.construct_model=construct_model; v61.logical_macs_per_sequence=logical_macs_per_sequence; v61.evaluate=evaluate

def train_one(config,seed,device,smoke=False):
    _patch(); path=v61.train_one(config,CONDITION,seed,device,smoke=smoke); payload=json.loads(Path(path).read_text()); payload['shared_private_ffn']={'shared_bank':SHARED_BANK,'shared_active':SHARED_ACTIVE,'private_per_round':PRIVATE_HIDDEN,'aligned_shared_private_base':PRIVATE_HIDDEN,'fullrank_basis_k':FULLRANK_BASIS_K,'round_ffn_parameterization':'shared_base_plus_fullrank_basis','active_total':SHARED_ACTIVE+PRIVATE_HIDDEN,'attention':'12_full_round_specific','topology':'fixed_maximal_coverage'}; v61.atomic_json(payload,Path(path)); return path

def preflight(config,device):
    print('='*144); print('V6.18 12.5/87.5 SHARED + ROUND-PRIVATE FFN PREFLIGHT'); print('='*144); print('device:',device)
    if device.type=='cuda': print('gpu:',torch.cuda.get_device_name(0)); print('bf16:',torch.cuda.is_bf16_supported())
    model=construct_model(config,CONDITION,123).to(device); dense_cfg=v63.ExperimentConfig(precision=config.precision,data_dir=config.data_dir); dense=v63.construct_model(dense_cfg,DENSE,seed=123)
    params=v61.parameter_count(model); dp=v61.parameter_count(dense); mac=logical_macs_per_sequence(config,CONDITION); dmac=v63.logical_macs_per_sequence(dense_cfg,DENSE); topo=model.block.ffn.topology_stats()
    print(f'V6.18 params={params:,} | dense params={dp:,} | difference={params-dp:+,}')
    print(f'V6.18 MAC/token={mac/config.context_length:,.0f} | dense MAC/token={dmac/config.context_length:,.0f} | ratio={mac/dmac:.5f}x')
    print(f'shared bank={SHARED_BANK} active={SHARED_ACTIVE} | aligned FFN={PRIVATE_HIDDEN} fullrank_basis_k={FULLRANK_BASIS_K} | total active={SHARED_ACTIVE+PRIVATE_HIDDEN}')
    print(f'topology J={topo["mean_pairwise_jaccard"]:.6f} overlap/active={topo["mean_pairwise_intersection_over_active"]:.6f} union={topo["bank_union_fraction"]:.6f}')
    old_private = (
        config.n_layers_or_rounds
        * 3
        * config.d_model
        * PRIVATE_HIDDEN
    )

    new_private = (
        (1 + FULLRANK_BASIS_K)
        * 3
        * config.d_model
        * PRIVATE_HIDDEN
        + 3
        * config.n_layers_or_rounds
        * FULLRANK_BASIS_K
    )

    # V6.10 total is 400 parameters below dense.
    expected = dp - 400 - old_private + new_private

    if params != expected:
        raise RuntimeError(
            f'parameter accounting failed: '
            f'expected {expected:,}, got {params:,}'
        )
    if abs(mac-TARGET_V65_MAC*config.context_length)>0: raise RuntimeError('active logical compute changed from V6.5')
    if abs(topo['bank_union_fraction']-1.0)>1e-6: raise RuntimeError('shared bank coverage failed')
    ptrs=[m.qkv.weight.data_ptr() for m in model.block.attn_rounds]
    if len(set(ptrs))!=8: raise RuntimeError('attention roles are shared')
    x=torch.randint(0,config.vocab_size,(2,32),device=device)
    with v61.autocast_context(device,config.precision):
        logits,diag=model(x,tau=1.0,stochastic=True,sparse_eval=False); loss=logits.float().square().mean()*1e-5+diag['auxiliary_loss']
    loss.backward()
    checks={'router':model.block.router_out.weight.grad,'shared_gate':model.block.ffn.gate_weight.grad,'basis_base_gate':model.block.private_ffn.gate.weight.grad,'basis_gate_alpha':model.block.private_ffn.gate_alpha.grad,'attention':model.block.attn_rounds[0].qkv.weight.grad}
    for name,g in checks.items():
        if g is None or not torch.isfinite(g).all() or float(g.norm())==0: raise RuntimeError(f'bad gradient {name}')
        print(f'gradient {name:12s}: {float(g.norm()):.3e}')
    # shared dense/indexed equivalence
    y=torch.randn(64,config.d_model,device=device); ids=torch.arange(64,device=device)%config.n_modes; rw=F.one_hot(ids,config.n_modes).float()
    with torch.no_grad(),v61.autocast_context(device,config.precision): yd=model.block.ffn.forward_dense_train(y,rw); ys=model.block.ffn.forward_sparse_eval(y,ids)
    diff=float((yd.float()-ys.float()).abs().max()); print(f'shared FFN dense/indexed max diff: {diff:.3e}')
    if diff>5e-2: raise RuntimeError('shared FFN dispatch mismatch')
    print('V6.18 12.5/87.5 SHARED + ROUND-PRIVATE FFN PREFLIGHT PASSED')

def _load(p): return json.loads(Path(p).read_text())
def _dense(seed):
    c=[]
    for p in sorted(Path('runs').glob(f'**/seed_{seed}/{DENSE}/result.json')):
        if 'smoke' in str(p).lower(): continue
        try:r=_load(p)
        except:continue
        if int(r.get('derived',{}).get('parameters',-1))==TARGET_DENSE_PARAMS:c.append(p)
    if not c: raise FileNotFoundError(f'dense seed {seed}')
    return c[0]
def aggregate(config,seeds,v65_root='runs/v6_5_transformer_parity'):
    raw={}
    for s in seeds:
        p=Path(config.output_root)/f'seed_{s}'/CONDITION/'result.json'; b=Path(v65_root)/f'seed_{s}'/V65/'result.json'; d=_dense(s)
        for q in (p,b,d):
            if not q.exists(): raise FileNotFoundError(q)
        raw[str(s)]={CONDITION:_load(p),V65:_load(b),DENSE:_load(d)}
    def vals(c,k): return [float(raw[str(s)][c]['validation'][k]) for s in seeds]
    sm={'version':VERSION,'seeds':seeds,'conditions':{},'paired':{},'raw':raw}
    for c in (DENSE,V65,CONDITION):
        sm['conditions'][c]={'loss':v61.stats(vals(c,'loss')),'perplexity':v61.stats(vals(c,'perplexity')),'parameters':v61.stats([float(raw[str(s)][c]['derived']['parameters']) for s in seeds]),'logical_macs_per_token':v61.stats([float(raw[str(s)][c]['derived']['logical_macs_per_token']) for s in seeds])}
    sm['paired']['v618_minus_v65']=v61.paired_stats(vals(CONDITION,'loss'),vals(V65,'loss')); sm['paired']['v618_minus_dense']=v61.paired_stats(vals(CONDITION,'loss'),vals(DENSE,'loss'))
    out=Path(config.output_root)/('aggregate_results_seeds_'+'_'.join(map(str,seeds))+'.json'); v61.atomic_json(sm,out)
    print('='*132); print('V6.18 12.5/87.5 SHARED + ROUND-PRIVATE FFN SUMMARY'); print('='*132)
    for c in (DENSE,V65,CONDITION): print(c,sm['conditions'][c]['loss']['mean'])
    for k,v in sm['paired'].items(): print(k,f"{v['mean']:+.4f}",f"95%CI [{v['ci95_low']:+.4f},{v['ci95_high']:+.4f}]")
    print('Saved:',out); return out

def parse_args():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    def common(s):
        s.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); s.add_argument('--precision',choices=('bf16','fp16','fp32'),default='bf16'); s.add_argument('--data-dir',default='data/v6_fineweb_edu'); s.add_argument('--output-root',default='runs/v6_18_fullrank_basis7'); s.add_argument('--train-steps',type=int); s.add_argument('--micro-batch',type=int); s.add_argument('--grad-accum',type=int)
    s=sub.add_parser('preflight'); common(s); s=sub.add_parser('train'); common(s); s.add_argument('--seed',type=int,required=True); s.add_argument('--smoke',action='store_true'); s=sub.add_parser('aggregate'); common(s); s.add_argument('--seeds',default='41,42,43'); s.add_argument('--v65-root',default='runs/v6_5_transformer_parity'); return p.parse_args()
def config_from_args(a):
    vals=asdict(ExperimentConfig())
    for k in ('precision','data_dir','output_root','train_steps','micro_batch','grad_accum'):
        v=getattr(a,k,None)
        if v is not None: vals[k]=v
    return ExperimentConfig(**vals)
def main():
    a=parse_args(); c=config_from_args(a); d=torch.device(a.device); torch.set_float32_matmul_precision('high')
    if a.command=='preflight':preflight(c,d)
    elif a.command=='train':train_one(c,a.seed,d,a.smoke)
    else:aggregate(c,[int(x) for x in a.seeds.split(',') if x],a.v65_root)
if __name__=='__main__':main()
