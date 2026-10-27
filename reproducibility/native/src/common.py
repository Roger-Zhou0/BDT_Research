"""Portable import adapter. This file is new packaging infrastructure, not study code.

Only verified local kernel imports and the historical numerical configuration are
exposed. There is no network, scheduler, GPU admission, or experiment-state I/O.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent
PACKAGE = NATIVE.parent
SPEC = json.loads((PACKAGE / 'configs/bct1_historical_protocol.json').read_text())


def module(name: str, path: Path):
    path = Path(path).resolve()
    if not path.is_relative_to(NATIVE):
        raise ValueError('Kernel path outside the packaged native directory')
    if name in sys.modules:
        found = sys.modules[name]
        if Path(found.__file__).resolve() != path:
            raise RuntimeError(f'Import collision: {name}')
        return found
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(name)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    try:
        spec.loader.exec_module(value)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return value


@lru_cache(maxsize=1)
def backend(project=None):
    if project is not None:
        raise ValueError('This reviewer adapter does not accept a live project')
    manifest = json.loads((NATIVE / 'NATIVE_KERNEL_MANIFEST.json').read_text())
    for name, expected in manifest.items():
        path = NATIVE / name
        if path.is_symlink() or not path.resolve().is_relative_to(NATIVE):
            raise RuntimeError('Unsafe kernel path')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Native source changed: {name}')
    base = NATIVE / 'vendor/bac1/backend'
    return {key: module('cms_protocol' if key == 'cms_protocol' else '_asf_backend_' + key, base / filename)
            for key, filename in [('native', 'validation_v6_1.py'), ('role', 'role_adamw.py'),
                                  ('temper', 'tempering_reference.py'), ('cms_protocol', 'cms_protocol.py'),
                                  ('geometry', 'cms_geometry.py')]}
