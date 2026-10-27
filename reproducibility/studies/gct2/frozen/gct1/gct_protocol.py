"""GCT1 frozen two-parent application pilot; no automatic expansion."""
from pathlib import Path
import hashlib,json
VERSION='growth_correspondence_transfer_v1'
HERE=Path(__file__).resolve().parent
SEEDS=(159,160)
ARMS=('adamw','bdt','scrambled_matched')
PARENT_STEPS=7630
POST_STEPS=3000
PREFIX=200
GATE_SEED=127
GATE_PARENT=200
GATE_END=420
GATE_INTERRUPT=310
PARAMETERS={'parent':81074688,'post':123551232}
DEPENDENCIES={
 'native':('fair_depth_lr_bracketing_v2/vendor/validation_v6_1.py','ee16f65a5b948b0e645131a01ddc437edb5806e671c44c80792b40efecc9dd8b'),
 'role':('fair_depth_lr_bracketing_v2/role_adamw.py','653afb6a644d634355801731e4ba492a10aa68a70ab5a1cca8adbb55ba2a8ed4'),
 'temper':('fair_depth_lr_bracketing_v2/tempering_reference.py','d1bd72d9ccd18019aea83551b838d5915f727a0a7b6a0a2fd1006fb1c47969f1'),
 'cms_protocol':('correspondence_matched_s1_v1/cms_protocol.py','e941b4274b27b1642e9467fdafe144fa79c378c199d2b5d35358fb7428261b7c'),
 'geometry':('correspondence_matched_s1_v1/cms_geometry.py','770a655ae6dc5e099b2239780ead32ace134d414d232c7e15d0a595d16e41e1c'),
}
DEPENDENCIES={k:list(v) for k,v in DEPENDENCIES.items()}
CACHE={
 'metadata.json':'64d9e563d8b12517cfe123560d39ab4f579ef9c4ec5482e215c20cb4f4108feb',
 'train.bin':'760fe8d0a55959dc322643fa4ac856e542e11bd59ac84c2f48f6e00a3d587ce6',
 'val.bin':'5dd2caeb46e3c237e6c9291d5964c548ae004cd87b3c29f01188b444af2aff0f',
}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def root(project):return Path(project).resolve()/'runs'/VERSION
SPEC=dict(version=VERSION,date='2026-09-10',account='account_b',session='good',cache='S1-B',
 seeds=list(SEEDS),arms=list(ARMS),model='ordinary Dense, independently initialized six-layer parent; stacked twice to twelve',
 parent=dict(layers=6,parameters=PARAMETERS['parent'],steps=PARENT_STEPS,lr=.0008),
 post_growth=dict(layers=12,parameters=PARAMETERS['post'],steps=POST_STEPS,lr=.0006,shared_prefix=PREFIX),
 mapping=list(range(6))*2,optimizer_age='7630 + post-growth phase update; moments and per-parameter ages preserved',
 covariance='reset at growth; whole 12-depth stack, two histories, newly observed during post-growth prefix',
 data_index='parent: step; post-growth: actual parent age + phase step; one batch hash each update',
 bdt=dict(cap=3.,observation_every=10,covariance_beta=.9,warmup=200,ramp=200),
 matching='Unchanged CMS1 FP64 own-state geometry with FP32 proposal cast; no random fallback',
 micro_batch=16,grad_accum=4,context=512,tokens_per_update=32768,precision='bf16',
 adam=dict(beta1=.9,beta2=.95,eps=1e-8,weight_decay=.1,clip=1.),warmup=200,min_lr_ratio=.1,
 strict=dict(deterministic=True,warn_only=False,cudnn_deterministic=True,cudnn_benchmark=False,tf32=False,workspace=':4096:8'),
 dev=dict(offset=0,sequences=2048,context=512,scored_tokens=1048576),report_access=False,
 order={'159':['adamw','bdt','scrambled_matched'],'160':['scrambled_matched','bdt','adamw']},
 retention='Final small parents, grown starts, post-growth prefixes, arm phase1000/2000/3000; rolling250; own passed-gate payloads pruned with manifest',
 checks='Copy weights+named moments, age, nonaliasing, RNG and next data index, zero reset histories, reference and active repeat/resume gate',
 gate=dict(seed=GATE_SEED,parent_steps=GATE_PARENT,prefix_repeats=2,active_end=GATE_END,interrupt=GATE_INTERRUPT,max_updates=2360),
 analysis='Two-seed DEV pilot: all endpoints, both paired differences and descriptive means, no confirmatory p-values or REPORT selection',
 decision='Complete both seeds. If mean BDT-Adam >=0 stop application expansion. If only Adam contrast favorable report incremental signal, not directional transfer. If both favorable consider separate confirmation; never auto-launch it.',
 accounting='Parent cost counted fully once per candidate training path, not divided by three branches. No scratch comparator and no end-to-end speedup claim.',
 retries='At most three same-recipe OOM attempts from committed state; other failures stop. External interruption requires explicit resume; no seed/batch/LR substitution.',
 automatic_extensions=False,scientific_optimizer_updates=2*(PARENT_STEPS+PREFIX+3*(POST_STEPS-PREFIX)))
PROTOCOL_SHA=digest(SPEC)
