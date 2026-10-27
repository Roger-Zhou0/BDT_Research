"""Validate and read the original stopped campaign without modifying it."""
from pathlib import Path
import json,math
import protocol as p

def reject_open_parent_report(project):
    root=Path(project).resolve()/'runs'/p.PARENT_VERSION
    if (root/'selection.lock.json').exists():
        raise RuntimeError('Original v1 already has selection.lock.json. Do not run this amendment; audit first.')
    if (root/'report').exists() and any(q.is_file() for q in (root/'report').rglob('*')):
        raise RuntimeError('Original reporting directory has files. Resolve reporting access before this follow-up.')

def verify_parent(project):
    root=Path(project).resolve()/'runs'/p.PARENT_VERSION
    reject_open_parent_report(project)
    if json.loads((root/'PROTOCOL.json').read_text())!=p.PARENT_SPEC:
        raise RuntimeError('Parent protocol differs from the supplied v1 evidence.')
    stop=json.loads((root/'selection.unbracketed.json').read_text())
    if stop.get('state')!='unbracketed' or any(stop['chosen_lr'].get(m)!=0.0012 for m in p.METHODS):
        raise RuntimeError('Expected the original stopped all-method 0.0012 boundary selection.')
    src=Path(project).resolve()/p.PARENT_VERSION
    for rel,h in p.PARENT_EVIDENCE['source_hashes'].items():
        if p.file_sha(src/rel)!=h:raise RuntimeError('Original source changed: '+str(src/rel))
    for rel,h in p.PARENT_EVIDENCE['files'].items():
        if p.file_sha(root/rel)!=h:raise RuntimeError('Original development evidence changed: '+str(root/rel))
    rows=[]
    for rel,h in p.PARENT_EVIDENCE['files'].items():
        if not rel.endswith('/result.json'):continue
        path=root/rel;obj=json.loads(path.read_text());ct=obj['contract']
        if obj['step']!=p.STEPS or obj['state']!='complete' or obj.get('report') is not None:
            raise RuntimeError('Invalid inherited endpoint: '+str(path))
        if ct['protocol_sha']!=p.PARENT_EVIDENCE['parent_protocol_sha'] or obj['contract_sha256']!=p.digest(ct):
            raise RuntimeError('Invalid inherited contract: '+str(path))
        rows.append(dict(seed=ct['seed'],method=ct['method'],lr=ct['lr'],dev_loss=obj['dev']['loss'],
            result=str(path),sha256=p.file_sha(path),stage_origin='inherited_v1'))
    if len(rows)!=42:raise RuntimeError('Expected exactly 42 inherited trials.')
    return rows

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--project',default='.');x=a.parse_args()
    r=verify_parent(x.project)
    print('PARENT EVIDENCE: PASS; 42 unchanged results; reporting absent; original source unchanged.')
