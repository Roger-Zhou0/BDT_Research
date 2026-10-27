"""Frozen scientific configuration. Standard-library only; safe before CUDA setup."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSION = 'correspondence_matched_s1_v1'
SEEDS = (147, 148, 149, 150, 151, 152)
ARMS = ('adamw', 'bdt', 'scrambled_matched')
ORDER = {
    147: ARMS,
    148: ('bdt', 'scrambled_matched', 'adamw'),
    149: ('scrambled_matched', 'adamw', 'bdt'),
    150: ('scrambled_matched', 'bdt', 'adamw'),
    151: ('bdt', 'adamw', 'scrambled_matched'),
    152: ('adamw', 'scrambled_matched', 'bdt'),
}
GATE_SEED = 127
PREFIX = 200
END = 3000
LR = .0012
DEV_OFFSET = 0
REPORT_OFFSET = 2500000
EVAL_SEQUENCES = 2048
CHECKPOINT_EVERY = 250
VAL_STEPS = (1, 50, 100, 200, 201, 250, 400, 500, 750, 1000, 1500, 2000, 2500, 3000)
DIAG_STEPS = (1, 200, 201, 250, 400, 500, 750, 1000, 1500, 2000, 2500, 3000)
DATA_HASHES = {
    'metadata.json': '64d9e563d8b12517cfe123560d39ab4f579ef9c4ec5482e215c20cb4f4108feb',
    'train.bin': '760fe8d0a55959dc322643fa4ac856e542e11bd59ac84c2f48f6e00a3d587ce6',
    'val.bin': '5dd2caeb46e3c237e6c9291d5964c548ae004cd87b3c29f01188b444af2aff0f',
}
GEOMETRY = dict(zero_angle_sine=1e-7, minimum_candidate_sine=1e-7,
                norm_relative_tolerance=1e-5, common_relative_tolerance=1e-5,
                angle_absolute_tolerance=5e-6, displacement_absolute_tolerance=2e-6,
                displacement_relative_tolerance=2e-4)
SPEC = dict(
    version=VERSION, date='2026-09-10', seeds=list(SEEDS), gate_seed=GATE_SEED,
    model='explicit-cloned S1', parameters=123550560, cache='S1-B', account='account_b',
    arms=list(ARMS), order={str(k):list(v) for k,v in ORDER.items()},
    prefix_updates=PREFIX, total_updates=END, peak_lr=LR, context=512,
    microbatch=16, accumulation=4, tokens_per_update=32768, precision='bf16',
    lr_warmup=200, min_lr_ratio=.1, beta1=.9, beta2=.95, eps=1e-8,
    weight_decay=.1, gradient_clip=1.,
    bdt=dict(cap=3., warmup=200, ramp=200, observation_every=10, covariance_beta=.9),
    geometry=GEOMETRY,
    dev=dict(offset=DEV_OFFSET, sequences=EVAL_SEQUENCES, context=512),
    endpoint=dict(offset=REPORT_OFFSET, sequences=EVAL_SEQUENCES, context=512,
                  evaluate='only after all 18 continuations complete; same established REPORT partition, not newly unseen data'),
    strict_execution=dict(workspace=':4096:8', deterministic_algorithms=True,
        warn_only=False, cudnn_benchmark=False, cudnn_deterministic=True,
        matmul_tf32=False, attention='automatic backend selection'),
    parent='One immutable update-200 model, named Adam state, RNG, canonical and scrambled covariance histories, and permutations per seed',
    shadows='All arms compute both operators and the matched control at their OWN state; only the applied private-FFN proposal differs',
    permutation='Independent hidden-unit permutation per depth, consistent gate/up rows and down columns; local CPU generator seeded by SHA256, not training RNG',
    numerics='Frozen BDT and proposal covariance in FP32; diagnostic geometry and reductions in FP64, matched proposal cast back to FP32',
    degenerate='Canonical sine <=1e-7: use exact raw proposal and log numerical-zero event. Otherwise scrambled orthogonal norm/raw-centered norm <=1e-7: STOP; no random fallback or seed replacement',
    primary='Final REPORT loss BDT minus scrambled_matched, paired by six training seeds',
    constructive='Final REPORT loss BDT minus AdamW',
    inference='All six endpoints, mean, SD, paired Student-t 95% CIs; also Bonferroni simultaneous 97.5% CIs for the two comparisons. No equivalence claim or adaptive sample size',
    execution_amendment='New strict-execution cohort; does not overwrite or retroactively confirm earlier non-deterministic REPORT runs',
    gate=dict(parent_updates=200, active_end=420, parents=2,
              active_runs=8, max_optimizer_updates=2160,
              tests='reference Adam warmup equality, active-policy repeatability, exact saved-state resume, frozen Adam/BDT equivalence, matched-strength invariants and transport'),
    actual_scientific_optimizer_updates=51600,
    checkpoint_every=CHECKPOINT_EVERY,
    failure='Stop on invariant/nonfinite/source/control mismatch. OOM: at most 3 same-recipe attempts from last committed checkpoint; never change batch size. Interrupted runs require explicit --resume',
    automatic_extensions=False,
)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()

PROTOCOL_SHA = digest(SPEC)

def root(project):
    return Path(project).resolve() / 'runs' / VERSION
