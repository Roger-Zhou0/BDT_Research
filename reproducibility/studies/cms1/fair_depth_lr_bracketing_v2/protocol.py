"""Frozen fair-comparison protocol. No torch import: safe for CPU-only orchestration."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path

HERE=Path(__file__).resolve().parent
PARENT_SPEC=json.loads((HERE/'PARENT_PROTOCOL.json').read_text())
PARENT_EVIDENCE=json.loads((HERE/'PARENT_EVIDENCE.json').read_text())
VERSION='fair_depth_lr_bracketing_v2'
PARENT_VERSION='fair_depth_optimizer_v1'
METHODS=('adamw','temper_cap3','gs_window')
INHERITED_LRS=(0.0001,0.0002,0.0003,0.0004,0.0006,0.0008,0.0012)
NEW_LRS=(0.0010,0.0016,0.0024)
INITIAL_LRS=NEW_LRS
EDGE_LRS=()
SEARCH_LRS=tuple(sorted(INHERITED_LRS+NEW_LRS))
DEV_SEEDS=(127,128)
REPORT_SEEDS=(129,130,131,132,133,134)
ALL_SEEDS=DEV_SEEDS+REPORT_SEEDS
STEPS=3000
VAL_STEPS=(1,50,100,200,250,400,500,750,1000,1500,2000,2500,3000)
GS_ALPHA=0.1
TOKENS_PER_STEP=32768
EVAL_SEQUENCES=2048
EVAL_CONTEXT=512
DEV_OFFSET=0
REPORT_OFFSET=2500000
SPEC=json.loads(json.dumps(PARENT_SPEC))
SPEC.update(version=VERSION,parent_version=PARENT_VERSION,
    parent_protocol_sha=PARENT_EVIDENCE['parent_protocol_sha'],
    parent_evidence_sha=hashlib.sha256((HERE/'PARENT_EVIDENCE.json').read_bytes()).hexdigest(),
    initial_lrs=list(NEW_LRS),edge_lrs=[],inherited_lrs=list(INHERITED_LRS),
    complete_search_lrs=list(SEARCH_LRS),
    amendment='Specified AFTER observing 42 v1 development results; BEFORE any new bracketing trial or reporting access. Not the original v1 preregistration.',
    reason='All three v1 recipes selected the upper 0.0012 boundary; v1 correctly stopped.',
    extension='All three methods receive exactly {0.0010,0.0016,0.0024} on BOTH development seeds; no further automatic expansion.',
    final_boundary='If ANY selected LR is the lower or upper outer boundary of the combined ten-point grid, stop before reporting. Otherwise lock and report regardless of development method ranking.',
    budget='18 new development trials; 60 total including 42 preserved v1 trials, 20 per method. Then 18 fresh reporting trials. Equal LR search only; no new optimizer hyperparameters.')

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
PROTOCOL_SHA = digest(SPEC)

def lr_tag(lr):
    return f'lr_{round(float(lr)*10000000):05d}'
def campaign(project):
    return Path(project).resolve()/'runs'/VERSION

def trial_dir(project,stage,seed,method,lr):
    return campaign(project)/stage/f'seed_{seed}'/method/lr_tag(lr)

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def verify_sources(root=None):
    root=Path(root or Path(__file__).parent)
    for name in ('SOURCE_MANIFEST.json','FROZEN_KERNELS.json'):
        for rel,h in json.loads((root/name).read_text()).items():
            q=root/rel
            if not q.is_file() or file_sha(q)!=h:
                raise RuntimeError(f'{name}: source mismatch {rel}; do not overwrite or bypass manifests')
    # Catch the previous import-shadowing packaging failure explicitly.
    if (root/'queue.py').exists():raise RuntimeError('Unexpected queue.py shadows the standard library')
