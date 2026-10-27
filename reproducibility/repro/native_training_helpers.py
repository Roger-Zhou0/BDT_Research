"""Exact function excerpts from the frozen trainer, with only import scaffolding replaced.
No queue, historical admission logic, or automatic job creation.
"""
from __future__ import annotations
import copy
import torch
import common as io
from types import SimpleNamespace
from gct_runtime import cpu,treehash
p=SimpleNamespace(POST_STEPS=3000,PARENT_STEPS=7630)

def config(native,data,post,tiny=False):
 return native.ExperimentConfig(d_model=48 if tiny else 768,n_heads=4 if tiny else 12,
  n_layers_or_rounds=(4 if post else 2) if tiny else (12 if post else 6),max_recurrent_rounds=12,
  ffn_hidden=64 if tiny else 2048,vocab_size=97 if tiny else 50257,context_length=16 if tiny else 512,
  micro_batch=2 if tiny else 16,grad_accum=1 if tiny else 4,train_steps=p.POST_STEPS if post else p.PARENT_STEPS,
  learning_rate=.0006 if post else .0008,warmup_steps=200,min_lr_ratio=.1,weight_decay=.1,beta1=.9,beta2=.95,
  grad_clip=1.,precision='fp32' if tiny else 'bf16',data_dir=str(data),output_root='unused',checkpoint_every=100)

def build(native,cfg,seed):
 native.seed_all(seed);m=native.DenseStackLM(cfg);native.initialize_model(m);return m

def source_name(name,parent_layers):
 mapping=list(range(parent_layers))*2 if io.SPEC['short']=='sbr1' else [i for i in range(parent_layers) for _ in range(2)]
 if name.startswith('blocks.'):
  a=name.split('.');index=int(a[1])
  if index<0 or index>=len(mapping):raise ValueError('Invalid target block')
  a[1]=str(mapping[index]);return '.'.join(a)
 return name

@torch.no_grad()
def copy_growth(parent_model,parent_opt,target_model,target_opt):
 """Map every weight and Adam moment, verify equality AND distinct storage."""
 layers=len(parent_model.blocks)
 if len(target_model.blocks)!=2*layers:raise RuntimeError('Growth must double depth')
 src=parent_model.state_dict();dst=target_model.state_dict()
 mapped={n:src[source_name(n,layers)].clone() for n in dst}
 target_model.load_state_dict(mapped,strict=True)
 del mapped
 old=parent_opt.named_adam();new=target_opt.named_adam();new['age']=old['age'];new['rest_groups']=copy.deepcopy(old['rest_groups'])
 for n in new['states']:new['states'][n]=cpu(old['states'][source_name(n,layers)])
 target_opt.load_adam(new);del new
 if target_opt.base.step_num!=target_opt.offset or target_opt.phase!=0:raise RuntimeError('Growth failed to preserve global age/reset phase')
 param_ptrs=[];moment_ptrs=[];checked=[];old_named=old['states'];new_named=target_opt.named_adam()['states']
 for n,t in target_model.named_parameters():
  sn=source_name(n,layers)
  if not torch.equal(t.cpu(),src[sn].cpu()):raise RuntimeError('Copied weight mismatch '+n)
  param_ptrs.append(t.data_ptr());a=old_named[sn];b=new_named[n]
  if treehash(a)!=treehash(b):raise RuntimeError('Copied named Adam mismatch '+n)
  if b['kind']=='projection':mt=[b['m'],b['v']]
  else:mt=[b['state'][k] for k in ('exp_avg','exp_avg_sq') if k in b['state']]
  moment_ptrs.extend(x.data_ptr() for x in mt);checked.append(dict(target=n,source=sn))
 if len(set(param_ptrs))!=len(param_ptrs) or len(set(moment_ptrs))!=len(moment_ptrs):raise RuntimeError('Duplicated storage is aliased')
 # Check no target moments/parameters alias the source tensors on the same device.
 old_p={v.data_ptr() for v in parent_model.parameters()}
 old_m={t.data_ptr() for row in old_named.values() for t in ([row['m'],row['v']] if row['kind']=='projection' else [row['state'][k] for k in ('exp_avg','exp_avg_sq') if k in row['state']])}
 if old_p.intersection(param_ptrs) or old_m.intersection(moment_ptrs):raise RuntimeError('Parent and grown storage alias')
 for group in (target_opt.canonical,target_opt.scrambled):
  for t in group.values():
   if t.last_step or t.observations or bool(torch.count_nonzero(t.covariance)) or t.basis.shape!=(2*layers,2*layers-1):raise RuntimeError('Covariance did not reset at growth')
 return dict(mapped_parameters=len(checked),mapping=checked,weights_exact=True,named_moments_exact=True,
   independent_parameter_storage=True,independent_moment_storage=True,parent_age=old['age'],grown_age=target_opt.base.step_num,
   phase=0,next_global_data_step=old['age']+1,covariance_depth_dimension=2*layers-1,both_covariance_histories_zero=True)
