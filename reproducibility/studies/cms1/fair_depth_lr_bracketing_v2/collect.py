#!/usr/bin/env python3
"""Read-only snapshot of small campaign evidence; no model/data binaries, no GPU."""
from __future__ import annotations
import argparse,hashlib,json,time,zipfile
from pathlib import Path
import protocol as p

def stable_bytes(path):
    for _ in range(3):
        a=path.stat();data=path.read_bytes();b=path.stat()
        if (a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns) and len(data)==a.st_size:return data
        time.sleep(.2)
    raise RuntimeError(f'File is still changing; retry collection when quiet: {path}')
def main():
    a=argparse.ArgumentParser();a.add_argument('--project',default='.');x=a.parse_args();project=Path(x.project).resolve();root=p.campaign(project)
    if not root.is_dir():raise FileNotFoundError(root)
    allowed={'.json','.txt','.log','.py','.md','.sh'}
    files={}
    for q in root.rglob('*'):
        if q.is_file() and q.suffix in allowed and '__pycache__' not in q.parts:files['campaign/'+q.relative_to(root).as_posix()]=q
    here=Path(__file__).resolve().parent
    for q in here.rglob('*'):
        if q.is_file() and q.suffix in allowed and '__pycache__' not in q.parts:files['source/'+q.relative_to(here).as_posix()]=q
    for q in (project/'logs').glob('e15_e16_bracket_*.log'):files['logs/'+q.name]=q
    for name in ('fair_bracketing_analysis.json','fair_bracketing_analysis.txt'):
        q=project/name
        if q.is_file():files['analysis/'+name]=q
    from parent_evidence import verify_parent
    verify_parent(project)
    old=project/'runs'/p.PARENT_VERSION
    for rel in p.PARENT_EVIDENCE['files']:
        files['parent_campaign/'+rel]=old/rel
    for rel in ('PROTOCOL.json','selection.unbracketed.json'):
        files['parent_campaign/'+rel]=old/rel
    for q in (project/p.PARENT_VERSION).rglob('*'):
        if q.is_file() and q.suffix in allowed and '__pycache__' not in q.parts and '.ipynb_checkpoints' not in q.parts:
            files['parent_source/'+q.relative_to(project/p.PARENT_VERSION).as_posix()]=q
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());out=project/f'e15_e16_bracket_results_{stamp}.zip'
    manifest={}
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,q in sorted(files.items()):
            if q.stat().st_size>100*1024**2:raise RuntimeError(f'Unexpected large non-checkpoint file {q}')
            data=stable_bytes(q);manifest[name]=hashlib.sha256(data).hexdigest();z.writestr(name,data)
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    print('Payload files:',len(manifest));print('SHA256='+p.file_sha(out));print('UPLOAD='+str(out))
if __name__=='__main__':main()
