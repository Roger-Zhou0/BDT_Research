"""Read-only candidate-seed audit. Does NOT register seeds or start a job."""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, re
from pathlib import Path
from design import SEEDS, GCT2, digest

SEED_NAME = re.compile(r'(?<![0-9])seed[_-]?(\d+)(?![0-9])', re.I)
INTERESTING = {'result.json','contract.json','registration.json','PREREG.json','PROTOCOL.json','protocol.json','prereg.json'}
SEED_KEYS={'seed','seeds','candidate_seeds','expected_seeds','training_seeds','parent_seed','parent_seeds'}

def seed_values(value):
    found=[]
    def walk(x,keys=()):
        if isinstance(x,dict):
            for key,v in x.items():
                if key in SEED_KEYS:
                    vals=v if isinstance(v,list) else [v]
                    for item in vals:
                        if type(item) is int:found.append((item,'.'.join(keys+(key,))))
                # Do not traverse model/optimizer states or long scalar training histories.
                if key not in ('history','curve','proof','model','optimizer','state_dict') and isinstance(v,(dict,list)):
                    walk(v,keys+(key,))
        elif isinstance(x,list):
            for i,v in enumerate(x):
                if isinstance(v,(dict,list)):walk(v,keys+(str(i),))
    walk(value)
    return found

def scan(project:Path,candidates=SEEDS):
    project=project.resolve();root=project/'runs';wanted=set(candidates)
    if not root.is_dir():raise ValueError('Project runs directory does not exist')
    collisions=[];errors=[];checked=[];names_seen=0
    own_root=root/'growth_correspondence_confirmation_v1'
    if own_root.exists():
        errors.append(dict(path=str(own_root),reason='Existing GCT2 needs its resume validator, not fresh registration'))
    def error(e):errors.append(dict(path=str(e.filename),reason=str(e)))
    for directory,dirs,names in os.walk(root,followlinks=False,onerror=error):
        here=Path(directory)
        keep=[]
        for name in dirs:
            p=here/name;names_seen+=1
            for m in SEED_NAME.finditer(name):
                if int(m.group(1)) in wanted:collisions.append(dict(path=str(p),seed=int(m.group(1)),kind='directory_name'))
            if p==own_root:continue
            if name in ('checkpoints','__pycache__'):continue
            if p.is_symlink():errors.append(dict(path=str(p),reason='Unresolved symlink in inventory'));continue
            keep.append(name)
        dirs[:]=keep
        for name in names:
            path=here/name;names_seen+=1
            for m in SEED_NAME.finditer(name):
                if int(m.group(1)) in wanted:collisions.append(dict(path=str(path),seed=int(m.group(1)),kind='file_name'))
            metadata=(name in INTERESTING or ('jobs' in path.parts and path.suffix=='.json') or
                      name.endswith('_registration.json') or name.endswith('_protocol.json'))
            if not metadata:continue
            try:
                if path.is_symlink():raise ValueError('Metadata symlink is not an audited file')
                before=path.stat()
                if before.st_size>16<<20:raise ValueError('Metadata exceeds audit cap; review rather than silently skip')
                raw=path.read_bytes();after=path.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Metadata changed while reading')
                content=json.loads(raw)
                checked.append(dict(path=str(path.relative_to(project)),sha256=hashlib.sha256(raw).hexdigest()))
                for seed,locator in seed_values(content):
                    if seed in wanted:collisions.append(dict(path=str(path),seed=seed,kind='metadata',locator=locator))
            except (OSError,ValueError) as ex:errors.append(dict(path=str(path),reason=str(ex)))
    return dict(state='clear' if not collisions and not errors else 'blocked',
                complete=not errors,project=str(project),candidate_seeds=list(candidates),
                names_seen=names_seen,metadata_records=len(checked),metadata_manifest_digest=digest(checked),
                checked_metadata=checked,collisions=collisions,errors=errors,
                time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                scope='Read-only scan of this target project/runs. Not a global account inventory or reservation.',
                no_side_effects=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project',type=Path,required=True)
    args=ap.parse_args();r=scan(args.project);print(json.dumps(r,indent=2));raise SystemExit(0 if r['state']=='clear' else 2)
