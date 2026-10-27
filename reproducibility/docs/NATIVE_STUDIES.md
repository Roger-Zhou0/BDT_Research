# Original study entrypoints and portable dispatch

## Original versus new code

The three `studies/` trees contain the original experiment sources and frozen
backends. They were recovered from archived releases, matched against original
manifests, and compared with the executed-source digests in recorded contracts
or registrations. A conveniently located older development tree was not treated
as the executed GCT2 release. See `studies/ORIGIN.json` for original and public
hashes and per-file identity-only transformations.

The original training policies, model construction, seeds, schedules, parameter
scope, evaluation slices and refusal conditions are retained. The public
manifests are regenerated for anonymous derivatives, not passed off as the
original private manifests. Some nested historical upstream hashes are retained
as provenance; they do not validate the redacted public source tree.

`scripts/native_study.py` is a NEW serial workspace/dispatch adapter. It does not
modify an original run or reproduce the original parallel scheduler history.
The following commands only print plans:

```bash
python scripts/native_study.py --study cms1 --action plan
python scripts/native_study.py --study gct2 --action plan
python scripts/native_study.py --study bac1 --action plan
```

| Study | Training units | Plan records | Scientific pipeline |
|---|---|---:|---|
| CMS1 | Six seeds, 147–152 | 24 | One 200-update explicit-cloned S1 prefix per seed; AdamW, BDT and matched-scrambled branches to 3,000. |
| GCT2 | Six new pretrained parents, 161–166 | 36 | Six-layer 7,630-update parents, stacked six-to-twelve copying (`r % 6`), 200-update common prefix, three branches to 3,000. |
| BAC1 | Six new pretrained parents, 175–180 | 42 | Stacked copying pipeline (`r % 6`) with AdamW, BDT, balanced ancestry-preserved and balanced reassigned branches. |

Plan records include growth conversions with zero optimizer updates. They are
not all equally expensive training jobs. Seed 127 is reserved for engineering
gates and is absent from all three scientific cohorts.

### Copying maps (zero-based indices)

GCT2 and BAC1 use **stacked copying**. Target block `r` copies pretrained source
block `r % 6`, producing:

```text
GCT2/BAC1: 0,1,2,3,4,5,0,1,2,3,4,5
```

The `source_name` functions in `studies/gct2/gct_train_adapter.py` and
`studies/bac1/train.py` implement `target_index % parent_layers`; the full-size
parent has six layers. Their `copy_growth` functions use that mapping for both
model parameters and named Adam moments, with distinct target storage.

BCT1 uses **adjacent copying**, with source block `r // 2`:

```text
Adjacent:  0,0,1,1,2,2,3,3,4,4,5,5
```

The BCT1 route retains this mapping in `repro/native_training_helpers.py`.
Stacked copying gives ancestry pairs `(j, j + 6)`; adjacent copying gives pairs
`(2j, 2j + 1)`. These layouts must not be substituted for one another. V9 changes
only their explanatory terminology, not code, mappings or historical records.

## CPU engineering checks

```bash
python scripts/native_study.py --study cms1 --action cpu-test --out build/cms1_cpu
python scripts/native_study.py --study gct2 --action cpu-test --out build/gct2_cpu
python scripts/native_study.py --study bac1 --action cpu-test --out build/bac1_cpu
```

These run the recovered original suites in fresh temporary or explicit output
directories. They preserve the original numerical tests, trusted checkpoint
formats, tiny model configurations and synthetic token streams. No real cache,
model weights, network request or CUDA device is used. Original CMS1 integration
uses 200 warmup updates and short active continuations through 214; the GCT2 and
BAC1 release suites also exercise through phase 420 with saved-state resumption
from phase 310. Do not report the original CMS1 short integration as a full
420-phase or CUDA validation. The BCT1 synthetic test is separate.

The results are in `validation/`. Complete transcripts in a local invocation
remain in the chosen output directory. Kernel errors, invalid geometry or corrupt
checkpoints are refusals; tolerances are not relaxed to obtain a pass.

## Explicit GPU templates — not CUDA-tested in this packaging pass

Use compute storage with at least 80 GiB free, expose exactly one suitable CUDA
BF16 GPU, and supply the exact external S1-B files. This is a conservative
storage admission requirement, not a measured peak-memory benchmark. Dependency
installation and full scientific compute are separate from inspecting this ZIP.
No cluster job or allocation is submitted by these commands.

```bash
# CMS1 example: change study, work and job count for GCT2 or BAC1.
CUDA_VISIBLE_DEVICES=0 python scripts/native_study.py --study cms1 --action init \
  --work /compute/new_cms1 --cache /compute/exact_s1b --allow-gpu
CUDA_VISIBLE_DEVICES=0 python scripts/native_study.py --study cms1 --action gate \
  --work /compute/new_cms1 --allow-gpu

# Run ONE indexed plan record at a time, in printed dependency order.
CUDA_VISIBLE_DEVICES=0 python scripts/native_study.py --study cms1 --action job \
  --work /compute/new_cms1 --index 0 --allow-gpu
# Continue explicitly with each subsequent plan index; use --resume only to
# resume that same declared job in this new workspace.

# Only after all declared endpoints are complete:
CUDA_VISIBLE_DEVICES=0 python scripts/native_study.py --study cms1 --action report \
  --work /compute/new_cms1 --allow-gpu
```

Initialization refuses an existing workspace, verifies exact cache bytes in
place, and links the cache rather than copying it. A source/runtime-bound full
GPU engineering gate is mandatory before scientific jobs. The gate's pass is
computed from actual reference/policy/repeat/resume checks, not from a manually
set success flag. If the gate fails, scientific jobs remain inadmissible.

CMS1 retains the original 18-endpoint seal and REPORT functions. GCT2 uses its
original 18-endpoint seal and evaluator; BAC1 requires all 24 endpoints. The new
serial GCT2 adapter labels one-device engineering provenance as serial rather
than inventing two historical hardware-lane observations. Completed historical
REPORT files are never rerun by the offline analysis commands.

This template exposes the original training functions but has not been exercised
end-to-end on CUDA. The historical runtime and CPU suites do not establish GPU
parity, installation availability, deterministic equivalence across different
GPUs, or an observed reproduction of the final scientific effects. Original
full schedules can require substantial compute; no completion-time estimate is
inferred from the small CPU gates.

## Fresh-document FEV1 evaluation

Original GCT2 training source and dependencies are included. FEV1 is supplied as
audited score/lock records and offline recomputation; its document-builder and
evaluation source are not included in this release. The document cache and its
original preparation bindings are distinct from S1-B. The convenient
serial adapter's `report` action does NOT run FEV1 or create a substitute document
cache. `reproduce_all.py` verifies the archived FEV1 score extraction and its
pairing with GCT2 models. Running a new FEV1 evaluation additionally requires the
correct external document cache, original builder/evaluator implementation and
contract; no arbitrary new sample
should be represented as that historical evaluation.
