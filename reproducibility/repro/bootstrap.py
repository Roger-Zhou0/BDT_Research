"""Local-only loading. Importing this module never starts training."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
# Set before the first torch import. CPU checks do not require CUDA.
old = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if old != ':4096:8':
    raise RuntimeError('CUBLAS_WORKSPACE_CONFIG must be :4096:8 before importing torch')
for path in [ROOT / 'native/src', ROOT / 'native/vendor/bac1/frozen']:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import common
MODS = common.backend()
import gct_runtime
import gct_policy
import bct_policy
import art_policy


def configure_cpu(threads: int = 1):
    if threads < 1:
        raise ValueError('Positive thread count required')
    gct_runtime.configure(threads)
    return MODS
