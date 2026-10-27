"""E14: frozen E9 cap-three depth tempering at the existing 501.9M S1 geometry.
Same explicit cloned private FFNs; no RAR factors, positive prefix, calibration,
external FULL schedule, or additional optimizer tuning. Two exact Adam-prefix forks.
"""
from __future__ import annotations
import contextlib, copy, hashlib, json, math, os, platform, random, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
import validation_v6_1 as native
import validation_v6_3 as generic
import s1_124m_aligned_symmetric as sym

VERSION = 'bounded_temper_scale500_v1'
MODEL_SIZE = '500m'
ARMS = ('adam', 'temper_strong')
PREFIXES = ('adam_prefix',)
ARM_PREFIX = {a: 'adam_prefix' for a in ARMS}
TEMPER_ARMS = set(ARMS) - {'adam'}

PROJS = ('gate', 'up', 'down')
PREFIX_STEPS = 200
TRAIN_STEPS = 3000
STOP_STEP = 3000
RAMP_STEPS = 200
OBSERVATION_EVERY = 10
COVARIANCE_BETA = 0.9
MAX_OPERATOR_CONDITION = 2.0
CALIBRATION_BATCHES = 8
HOLDOUT_BATCHES = 2
ANISO_KAPPA = 50.0
VAL_STEPS = (1, 50, 100, 150, 200, 201, 250, 400, 500, 750, 1000, 1500, 2000, 2500, 3000)
DIAG_STEPS = VAL_STEPS
CHECKPOINT_EVERY = 250

# The inactive prefix retains E9's cap-two covariance collector. Its cap is
# irrelevant while active=False. Only cap three is applied in the treatment.
TEMPER_CONFIGS = {
    'adam': dict(warmup_steps=200, ramp_steps=200, max_operator_condition=1.0),
    'temper_v1': dict(warmup_steps=200, ramp_steps=200, max_operator_condition=2.0),
    'temper_strong': dict(warmup_steps=200, ramp_steps=200, max_operator_condition=3.0),
}

def temper_config_dict(arm):
    if arm not in TEMPER_CONFIGS: raise ValueError(arm)
    d = dict(TEMPER_CONFIGS[arm])
    d.update(observation_every=OBSERVATION_EVERY, covariance_beta=COVARIANCE_BETA)
    return d


SPECS = {
    '500m': dict(d_model=1536, n_heads=12, n_layers_or_rounds=12, n_modes=12,
                 ffn_hidden=5632, shared_active=704, private_hidden=4928,
                 bank=8386, vocab_size=50257, context=512),
    '124m': dict(d_model=768, n_heads=12, n_layers_or_rounds=12, n_modes=12,
                 ffn_hidden=2048, shared_active=256, private_hidden=1792,
                 bank=3016, vocab_size=50257, context=512),
    '_tiny': dict(d_model=48, n_heads=4, n_layers_or_rounds=4, n_modes=4,
                  ffn_hidden=64, shared_active=8, private_hidden=56,
                  bank=30, vocab_size=97, context=16),
}
EXPECTED_COUNT = 501904152
EXPECTED_B = {
    'metadata_sha256': '64d9e563d8b12517cfe123560d39ab4f579ef9c4ec5482e215c20cb4f4108feb',
    'val_sha256': '5dd2caeb46e3c237e6c9291d5964c548ae004cd87b3c29f01188b444af2aff0f',
    'train_sha256': '760fe8d0a55959dc322643fa4ac856e542e11bd59ac84c2f48f6e00a3d587ce6',
    'train_bytes': 6000000000,
    'val_bytes': 10000000,
}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def json_digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {str(p.relative_to(HERE)): sha_file(p)
            for p in sorted(HERE.rglob('*.py')) if '__pycache__' not in p.parts}


def stamp(path):
    s = Path(path).stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino)


def runtime_identity(device):
    r = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
             cuda_runtime=torch.version.cuda, device_type=device.type,
             cpu_threads=torch.get_num_threads())
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(device)
        r.update(gpu_model=p.name,
                 capability=list(torch.cuda.get_device_capability(device)),
                 total_memory=p.total_memory,
                 cudnn=torch.backends.cudnn.version())
    return r


def verify_cache(data, root):
    data = Path(data).resolve()
    paths = {n: data / n for n in ('metadata.json', 'train.bin', 'val.bin')}
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(p)
    if paths['train.bin'].stat().st_size != EXPECTED_B['train_bytes'] or paths['val.bin'].stat().st_size != EXPECTED_B['val_bytes']:
        raise RuntimeError('Wrong S1-B cache sizes')
    before = {n: stamp(p) for n, p in paths.items()}
    got = {}
    for n, k in [('metadata.json', 'metadata_sha256'), ('val.bin', 'val_sha256'), ('train.bin', 'train_sha256')]:
        print(f'[cache] SHA-256 {n} ({before[n]["size"]:,} bytes); read-only', flush=True)
        got[k] = sha_file(paths[n])
        if got[k] != EXPECTED_B[k]:
            raise RuntimeError(f'Cache hash mismatch: {n}')
    after = {n: stamp(p) for n, p in paths.items()}
    if before != after:
        raise RuntimeError('Cache changed while hashing')
    info = dict(label='S1-B', path=str(data), hashes=got, stat=after,
                assurance='complete SHA-256 train/val')
    atomic_json(Path(root) / 'cache_identity.json', info)
    return info


def quick_cache_guard(identity):
    for n, old in identity['stat'].items():
        if stamp(Path(identity['path']) / n) != old:
            raise RuntimeError(f'Cache stat changed: {n}')


def make_config(size, data_dir, micro=16, accum=4, lr=0.0004, precision='bf16'):
    s = SPECS[size]
    cfg = generic.ExperimentConfig(
        d_model=s['d_model'], n_heads=s['n_heads'],
        n_layers_or_rounds=s['n_layers_or_rounds'],
        max_recurrent_rounds=s['n_layers_or_rounds'], n_modes=s['n_modes'],
        attention_adapter_modes=s['n_layers_or_rounds'], ffn_hidden=s['ffn_hidden'],
        sparse_bank_hidden=max(s['bank'], s['ffn_hidden']), context_length=s['context'],
        vocab_size=s['vocab_size'], micro_batch=micro, grad_accum=accum,
        train_steps=TRAIN_STEPS, warmup_steps=200, learning_rate=lr,
        min_lr_ratio=0.1, weight_decay=0.1, beta1=0.9, beta2=0.95,
        grad_clip=1.0, precision=precision, data_dir=data_dir, output_root='unused',
        validate_every=250, checkpoint_every=250, eval_batches=32)
    cfg.sparse_bank_hidden = s['bank']
    return cfg


def rng_capture():
    return dict(torch=torch.get_rng_state().clone(), numpy=copy.deepcopy(np.random.get_state()),
                python=random.getstate(),
                cuda=[x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None)


def rng_restore(r):
    torch.set_rng_state(r['torch'].cpu())
    np.random.set_state(r['numpy'])
    random.setstate(r['python'])
    if r['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda']])


@contextlib.contextmanager
def preserved_rng():
    r = rng_capture()
    try:
        yield
    finally:
        rng_restore(r)


class ExplicitPrivate(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.roles = nn.ModuleList([native.SwiGLU(source.d_model, source.hidden)
                                    for _ in range(source.n_rounds)])
        with torch.no_grad():
            for ff in self.roles:
                for proj in PROJS:
                    getattr(ff, proj).weight.copy_(getattr(source, proj).weight)

    def forward(self, x, round_idx):
        return self.roles[int(round_idx)](x)


def build(size, cfg, seed):
    s = SPECS[size]
    sym.SHARED_ACTIVE = s['shared_active']
    sym.PRIVATE_HIDDEN = s['private_hidden']
    sym.SHARED_BANK = s['bank']
    sym.FULLRANK_BASIS_K = cfg.n_layers_or_rounds - 1
    native.seed_all(seed)
    m = sym.SharedPrivateLM(cfg, seed=seed)
    native.initialize_model(m)
    m.block.private_ffn.reset_basis()
    p = m.block.private_ffn
    r = rng_capture()
    m.block.private_ffn = ExplicitPrivate(p)
    rng_restore(r)
    m.condition = f'{VERSION}_{size}_explicit_cloned'
    count = sum(x.numel() for x in m.parameters())
    if size == MODEL_SIZE and count != EXPECTED_COUNT:
        raise RuntimeError(f'count {count} != {EXPECTED_COUNT}')
    return m


def tensor_hash(t):
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def fingerprint(m):
    h = hashlib.sha256()
    for n, t in sorted(m.state_dict().items()):
        h.update(n.encode()); h.update(tensor_hash(t).encode())
    roles = m.block.private_ffn.roles
    for proj in PROJS:
        ws = [getattr(ff, proj).weight for ff in roles]
        if len({w.data_ptr() for w in ws}) != len(ws):
            raise RuntimeError('role weights alias')
        if not all(torch.equal(ws[0], w) for w in ws):
            raise RuntimeError('roles not cloned')
    return dict(state_sha256=h.hexdigest(), parameters=sum(p.numel() for p in m.parameters()),
                post_constructor_cpu_rng=tensor_hash(torch.get_rng_state()))


def role_parameters(model, proj):
    return [getattr(ff, proj).weight for ff in model.block.private_ffn.roles]


def role_parameter_ids(model):
    return {id(p) for q in PROJS for p in role_parameters(model, q)}


def rest_parameters(model):
    ids = role_parameter_ids(model)
    return [p for p in model.parameters() if id(p) not in ids]


def centered(x):
    return x - x.mean(0, keepdim=True)


def centered_norm(x):
    return float(centered(x).float().norm())


def centered_role_basis_from_gradient(g):
    G = centered(g.reshape(g.shape[0], -1).float())
    gram = (G @ G.T).double()
    evals, evecs = torch.linalg.eigh(gram)
    idx = torch.argsort(evals, descending=True)
    evals, evecs = evals[idx], evecs[:, idx]
    k = G.shape[0] - 1
    Q = evecs[:, :k].float()
    Q = Q - Q.mean(0, keepdim=True)
    gg = (Q.T @ Q).double()
    ev, u = torch.linalg.eigh(gg)
    Q = Q @ (u @ torch.diag(ev.clamp_min(1e-20).rsqrt()) @ u.T).float()
    pos = evals[:k].clamp_min(0)
    tot = float(pos.sum())
    fr = {str(j): float(pos[:min(j, k)].sum() / tot) if tot > 0 else None for j in (1, 3, 5)}
    return Q, dict(eigenvalues=evals.tolist(), q=k,
                   center_error=float(Q.mean(0).abs().max()),
                   orth_error=float((Q.T @ Q - torch.eye(k)).abs().max()),
                   gradient_energy_fraction_topk=fr)


def anisotropy_spectrum(k, kappa=ANISO_KAPPA):
    if k < 1 or kappa <= 1:
        raise ValueError((k, kappa))
    return torch.logspace(0.0, -math.log10(float(kappa)), steps=k, dtype=torch.float32)


def operator_stats(lam):
    l = lam.double()
    return dict(kappa=float(l.max()/l.min()),
                stable_rank=float(l.square().sum()/l.max().square()),
                eigenvalues=l.tolist())


def apply_role_operator(x, Q, lam):
    shp = x.shape
    flat = centered(x.reshape(shp[0], -1).float())
    Q = Q.to(flat); lam = lam.to(flat)
    return (Q @ (lam[:, None] * (Q.T @ flat))).reshape(shp)


def cosine(a, b):
    a = a.float().reshape(-1); b = b.float().reshape(-1)
    den = float(a.norm() * b.norm())
    return float((a @ b) / den) if den > 0 else None


def operator_gradient_alignment(g, Q, lam):
    G = centered(g.reshape(g.shape[0], -1).float())
    MG = apply_role_operator(G, Q, lam).reshape_as(G)
    return cosine(MG, G)


def save_operators(path, task, lam, meta):
    tmp = Path(path).with_name(Path(path).name + f'.{os.getpid()}.tmp')
    torch.save(dict(version=VERSION, task=task, spectrum=lam, meta=meta), tmp)
    os.replace(tmp, path)


def load_operators(path, device):
    d = torch.load(path, map_location='cpu', weights_only=False)
    if d.get('version') != VERSION:
        raise RuntimeError('operator version mismatch')
    return {'task': {q: t.to(device) for q, t in d['task'].items()},
            'spectrum': d['spectrum'].to(device)}, d['meta']


def prefix_contract(prefix, seed, cfg, data, runtime):
    cc = asdict(cfg); cc.pop('output_root', None)
    return dict(version=VERSION, kind='prefix', prefix=prefix, seed=seed, config=cc,
                model_geometry=SPECS[MODEL_SIZE], source_sha256=source_hashes(),
                runtime=runtime, data_label=data['label'], data_hashes=data['hashes'],
                prefix_steps=PREFIX_STEPS,
                optimizer='ordinary AdamW prefix with exposed explicit-role AdamW proposals; raw update covariance observed only')


def arm_contract(arm, seed, cfg, data, runtime, parent_digest, parent_prefix):
    cc = asdict(cfg); cc.pop('output_root', None)
    return dict(version=VERSION, kind='arm', arm=arm, seed=seed, config=cc,
                model_geometry=SPECS[MODEL_SIZE], source_sha256=source_hashes(),
                runtime=runtime, data_label=data['label'], data_hashes=data['hashes'],
                optimization_stop_step=STOP_STEP, prefix_steps=PREFIX_STEPS,
                temper_config=temper_config_dict(arm),
                parent_prefix=parent_prefix, parent_checkpoint_sha256=parent_digest,
                late_tempering=(arm in TEMPER_ARMS),
                scope='prespecified four-seed frozen cap-three optimizer scale transfer; same 3000-update schedule; no tuning, no companion norm-reference run')


def state_dict_hash(model):
    h = hashlib.sha256()
    for n, t in sorted(model.state_dict().items()):
        h.update(n.encode()); h.update(tensor_hash(t).encode())
    return h.hexdigest()


def state_tree_hash(obj):
    """Value hash for model/optimizer/RNG audit trees (not Python object IDs)."""
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu().contiguous()
            h.update(b'tensor'); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(b'array'); h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            h.update(b'dict')
            for k in sorted(x, key=lambda k: (type(k).__name__, str(k))):
                visit(k); visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode()); h.update(str(len(x)).encode())
            for v in x: visit(v)
        else:
            h.update(type(x).__name__.encode()); h.update(repr(x).encode())
        h.update(b'\0')
    visit(obj)
    return h.hexdigest()
