# Full-size BCT1 reproduction template

**Status: implemented, CPU-engineering tested, not GPU-parity certified.**
This route reproduces the two-hybrid BCT1 recipe on newly regenerated parents.
It does not resume, read, or write the original experiment. It is not a new
scientific result or a replacement for its already completed REPORT closure.

## Preconditions

Provide the exact S1-B cache and an appropriate CUDA BF16 environment. The
historical full BCT1 runtime still needs recovery; the CMS1 runtime recovered in
`reference/` is not automatically a BCT1 environment lock. No wall-clock or memory admission
number is asserted from the small CPU test. A failed operation does not authorize
weakening a geometry guard, changing endpoints, replacing seeds, or using another
cache.

Training each of the four families executes a six-layer parent for 7,630 updates
at peak LR 0.0008, adjacent copying into twelve independent blocks while retaining
Adam moments and age, a 200-update shared prefix, and two continuations from 200
to 3,000 at peak LR 0.0006. The family has 13,430 scientific updates of mixed
model size. Four families have 53,720 such updates. The order of the two arms is
retained from the historical protocol. Both hypothetical FFN budgets are updated
at each branch's own state; no other branch's schedule or tensor is borrowed.

The complete model has width 768, 12 heads, FFN width 2,048 and 123,551,232
parameters. Context 512, microbatch 16 and accumulation 4 give 32,768 tokens per
update. Warmup and BDT ramp are each 200 updates; observation cadence is ten;
BDT covariance EMA coefficient is 0.9 per observation; cap is three. Full details
remain in `configs/bct1_historical_protocol.json`.

## Explicit commands

These commands can require substantial compute. They are not run by installation,
imports, tests, or package checks. Use a local directory that does not exist yet.
The script refuses an existing output and does not auto-resume full-size jobs.

```bash
python scripts/train_bct1.py --data /path/to/exact_s1b \
  --out /path/to/new_cohort/seed_946201 --seed 946201 --allow-training
```

Execute separate explicit commands for the other historical families 946202,
946203, and 946204, changing both seed and output directory. There is no cluster
submission wrapper, automated queue, or interaction with the authors' workers.
The driver does not perform intermediate DEV or partial-cohort REPORT evaluation.

After all four family directories contain completed results and eight endpoint
checkpoints, the following command verifies all eight byte bindings before any
loss evaluation:

```bash
python scripts/evaluate_bct1.py --cohort-root /path/to/new_cohort \
  --data /path/to/exact_s1b --out /path/to/new_report --allow-evaluation
```

The output is explicitly labelled a regenerated portable cohort. It is NOT
pooled with the paper's four reused parents and does not overwrite the recorded
results. The linked combined-BDT/assembled reference rerun is not exposed in this
release; therefore the template reproduces the primary two-hybrid comparison,
not the full historical 16-logical-cell evaluation closure or residual contrast.
The residual contrast can already be checked from the packaged original scores.

## Checkpoints and restart

Portable checkpoints use a new, deliberately distinct format and local checksum
sidecars. Loading uses PyTorch's restricted `weights_only=True` path. They
include both candidate histories, the attention history, model/optimizer state,
phase/Adam age, and random states. A file checksum is an integrity check, not a
cryptographic trust signature. Only load your own outputs or trusted files.
The CPU test confirms exact uninterrupted-versus-restarted outcomes from phase
310 to 420 for both arms. Full-size interruption/resume and cross-GPU portability
are not validated; do not infer them from the CPU check.
