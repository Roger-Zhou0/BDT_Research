"""Metadata-first, read-only inspection of an externally supplied token cache."""
from __future__ import annotations
from pathlib import Path
import hashlib, importlib.metadata, json, os, platform

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_BYTES = {'metadata.json': 346, 'train.bin': 6_000_000_000, 'val.bin': 10_000_000}


def expected_cache() -> dict:
    # Authoritative cache hashes are also in the unchanged native study protocol.
    hashes = json.loads((ROOT/'configs/bct1_historical_protocol.json').read_text())['cache_sha256']
    return {name: {'bytes': HISTORICAL_BYTES[name], 'sha256': h} for name, h in hashes.items()}


def _stamp(p: Path) -> tuple:
    s = p.stat(); return s.st_size, s.st_mtime_ns, s.st_ino, s.st_dev


def inspect_cache(path: Path, *, hash_bytes: bool = False, expected: dict | None = None) -> dict:
    """No network, copy, cache creation, or tensor deserialization occurs here."""
    expected = expected_cache() if expected is None else expected
    if set(expected) != set(HISTORICAL_BYTES):
        raise ValueError('A cache contract must include all three explicit files')
    for contract in expected.values():
        if (not isinstance(contract.get('bytes'), int) or isinstance(contract['bytes'], bool)
                or contract['bytes'] <= 0 or not isinstance(contract.get('sha256'), str)
                or len(contract['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in contract['sha256'])):
            raise ValueError('Invalid size/hash in cache contract')
    rows = {}; path = Path(path); valid = True; read_bytes = 0
    for name, contract in expected.items():
        if name not in HISTORICAL_BYTES:
            raise ValueError('Only the three explicit token-cache files are accepted')
        p = path/name
        if p.is_symlink():
            raise ValueError('Symlink cache file refused')
        if not p.is_file():
            rows[name] = {'exists': False}; valid = False; continue
        before = _stamp(p)
        row = {'exists': True, 'bytes': before[0], 'expected_bytes': contract['bytes'],
               'size_matches': before[0] == contract['bytes'], 'sha256_checked': False}
        valid = valid and row['size_matches']
        # Never hash an unexpectedly sized file, which could be a huge unrelated artifact.
        if hash_bytes and row['size_matches']:
            h = hashlib.sha256()
            with p.open('rb') as f:
                while b := f.read(4 * 1024 * 1024):
                    read_bytes += len(b); h.update(b)
            if before != _stamp(p):
                raise RuntimeError('Cache changed during hashing')
            row.update(sha256=h.hexdigest(), sha256_checked=True,
                       hash_matches=h.hexdigest() == contract['sha256'])
            valid = valid and row['hash_matches']
        rows[name] = row
    return {'mode': 'IN_PLACE_HASH_VERIFICATION' if hash_bytes else 'SIZE_ONLY_PREFLIGHT',
            'exact_cache_verified': bool(hash_bytes and valid), 'sizes_match': all(r.get('size_matches',False) for r in rows.values()),
            'files': rows, 'payload_bytes_read': read_bytes, 'files_written': 0,
            'network_requests': 0, 'local_cache_copies_created': 0,
            'boundary': 'Size-only checks are not integrity checks. Hash success identifies token-cache bytes, not document provenance or end-to-end reproduction.'}


def recorded_environment_comparison() -> dict:
    record = json.loads((ROOT/'reference/ARCHIVED_DATA_AND_RUNTIME.json').read_text())
    variants = record['cms1_runtime_variants']
    if len(variants) != 1:
        raise RuntimeError('Expected one archived CMS1 runtime; compare variants explicitly')
    expected = next(iter(variants.values()))
    current = {'python': platform.python_version()}
    for package in ('numpy', 'torch'):
        try:
            current[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            current[package] = None
    equal = {key: current[key] == expected[key] for key in current}
    return {'historical_study': 'CMS1', 'contracts_checked_in_archive': len(record['cms1_contract_bindings']),
            'current_package_versions': current, 'recorded_package_versions': {key: expected[key] for key in current},
            'package_version_matches': equal, 'all_package_versions_match': all(equal.values()),
            'gpu_environment_verified': False, 'environment_modified': False,
            'boundary': 'Recorded contract fields, not a complete environment lock or install-availability check. No CUDA context or training is started.'}
