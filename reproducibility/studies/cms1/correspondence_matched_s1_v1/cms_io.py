"""Integrity, bounded output, and transactional checkpoint helpers."""
from __future__ import annotations
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid
import cms_protocol as p


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with Path(path).open() as f:
        return json.load(f)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('x') as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def append_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n'); f.flush()


def verify_manifest(root, manifest):
    root = Path(root).resolve()
    for rel, expected in manifest.items():
        path = root / rel
        if path.is_symlink() or not path.is_file():
            raise RuntimeError('Missing file or symlink in frozen tree: ' + str(path))
        path.resolve().relative_to(root)
        if sha(path) != expected:
            raise RuntimeError('Frozen source mismatch: ' + str(path))


def verify_sources(project=None):
    manifest = read_json(p.HERE / 'SOURCE_MANIFEST.json')
    verify_manifest(p.HERE, manifest)
    tracked = {str(q.relative_to(p.HERE)) for q in p.HERE.rglob('*.py')
               if '__pycache__' not in q.parts}
    if tracked - set(manifest):
        raise RuntimeError('Untracked executable source: ' + repr(tracked - set(manifest)))
    if project is not None:
        verify_manifest(Path(project) / 'fair_depth_lr_bracketing_v2',
                        read_json(p.HERE / 'FROZEN_BACKEND.json'))
    return p.digest(manifest)


@contextlib.contextmanager
def file_lock(path, blocking=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as ex:
            raise RuntimeError('Another process owns lock: ' + str(path)) from ex
        try: yield f
        finally: fcntl.flock(f, fcntl.LOCK_UN)


def stamp(path):
    s = Path(path).stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino)


def verify_data(project):
    data = Path(project).resolve() / 'data_s1/fineweb_edu_3b'
    before = {n: stamp(data/n) for n in p.DATA_HASHES}
    for name, expected in p.DATA_HASHES.items():
        print('[cache] read-only SHA256', name, before[name]['size'], flush=True)
        if sha(data/name) != expected: raise RuntimeError('Wrong S1-B token realization: ' + name)
    if before != {n: stamp(data/n) for n in p.DATA_HASHES}:
        raise RuntimeError('Cache changed during verification')
    return dict(path=str(data), hashes=p.DATA_HASHES, stat=before, label='S1-B')


def cache_guard(identity):
    if {n: stamp(Path(identity['path'])/n) for n in identity['stat']} != identity['stat']:
        raise RuntimeError('Token cache changed after registration')


def write_contract(directory, contract):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory/'contract.json'
    if path.exists():
        if read_json(path) != contract: raise RuntimeError('Existing contract differs: ' + str(path))
    else: atomic_json(path, contract)


def commit_checkpoint(directory, payload, torch, state_identity):
    """Atomic pointer commit. Delete only the superseded checkpoint created here.

    Orphaned uncommitted files from killed writers are preserved, never adopted.
    The published pointer always identifies a complete file by SHA256.
    """
    directory = Path(directory); store = directory/'checkpoints'
    store.mkdir(parents=True, exist_ok=True)
    pointer = directory/'latest.json'
    previous = read_json(pointer) if pointer.exists() else None
    stem = f'step_{payload["step"]:06d}_{uuid.uuid4().hex}'
    tmp = store/('.'+stem+'.tmp'); final = store/(stem+'.pt')
    try:
        with tmp.open('xb') as f:
            torch.save(payload, f); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, final)
        info = dict(path=str(final.relative_to(directory)), sha256=sha(final),
                    bytes=final.stat().st_size, step=payload['step'],
                    contract_sha=p.digest(payload['contract']), identity=state_identity)
        atomic_json(pointer, info)
    finally:
        if tmp.exists(): tmp.unlink()
    # Parents are immutable once published. Only this trial's old rolling file is removed.
    if previous and previous['path'] != info['path']:
        old = directory/previous['path']
        if old.parent.resolve() != store.resolve() or old.suffix != '.pt':
            raise RuntimeError('Refusing unsafe checkpoint cleanup')
        if old.exists():
            append_json(directory/'checkpoint_retention.jsonl',
                        dict(time=now(), superseded=previous, retained=info))
            old.unlink()
    return info


def load_checkpoint(directory, torch):
    directory = Path(directory); info = read_json(directory/'latest.json')
    path = directory/info['path']
    path.resolve().relative_to(directory.resolve())
    if path.is_symlink() or sha(path) != info['sha256']:
        raise RuntimeError('Checkpoint checksum mismatch: ' + str(path))
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if obj['step'] != info['step'] or p.digest(obj['contract']) != info['contract_sha']:
        raise RuntimeError('Checkpoint pointer/contract mismatch')
    return obj, info
