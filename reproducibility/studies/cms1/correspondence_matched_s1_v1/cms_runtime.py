"""Verified frozen-backend loading and the strict-auto execution policy."""
from __future__ import annotations
import importlib
import os
from pathlib import Path
import platform
import sys


def configure(threads=4):
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before importing PyTorch')
    import torch
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    # Do not force an attention backend. Retain the strict-auto policy actually tested.
    return torch


def backend(project):
    import cms_io as io
    io.verify_sources(project)
    root = (Path(project)/'fair_depth_lr_bracketing_v2').resolve()
    sys.path.insert(0, str(root))
    mods = {}
    for name in ('model_support', 'engine', 'optimizers', 'tempering_reference'):
        m = importlib.import_module(name)
        if Path(m.__file__).resolve().parent != root:
            raise RuntimeError('Import shadowing detected: ' + name)
        mods[name] = m
    return mods


def signature(torch, ms, device):
    r = ms.runtime_identity(device)
    r.update(deterministic=torch.are_deterministic_algorithms_enabled(),
             warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
             workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
             cudnn_benchmark=torch.backends.cudnn.benchmark,
             cudnn_deterministic=torch.backends.cudnn.deterministic,
             matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
             flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
             math_sdp=torch.backends.cuda.math_sdp_enabled(),
             efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled())
    if hasattr(torch.backends.cuda, 'cudnn_sdp_enabled'):
        r['cudnn_sdp'] = torch.backends.cuda.cudnn_sdp_enabled()
    if not r['deterministic'] or r['warn_only'] or r['workspace'] != ':4096:8':
        raise RuntimeError('Strict execution settings changed')
    return r


def cpu_tree(value, torch):
    import copy
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k:cpu_tree(v,torch) for k,v in value.items()}
    if isinstance(value, tuple): return tuple(cpu_tree(v,torch) for v in value)
    if isinstance(value, list): return [cpu_tree(v,torch) for v in value]
    return copy.deepcopy(value)
