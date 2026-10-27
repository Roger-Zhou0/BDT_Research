# GCT2 + BCO1 launch release — September 10, 2026

This release supplies the actual two-lane GCT2 coordinator, cohort barrier and deferred REPORT evaluator, and the BCO1 profile/equivalence/timing collector. The approved protocols in `design/` are retained verbatim. Their `status` strings describe the design-stage document, not the live run; consult runtime status files below.

## Upload once per account; launch once per assigned session

Place `gct2_bco1_launch_2026-09-10.tar.gz` and `launch_gct2_bco1.sh` in `~/recurrent-graph/recurrent_graph_s1_124m/` on each account.

* account_b/good: `bash launch_gct2_bco1.sh gct2`
* account_b/default: `bash launch_gct2_bco1.sh gct2`
* account_a/good: `bash launch_gct2_bco1.sh bco1`

Do not use account_a/default for either study. The two account_b launches share an installation but execute different seed lanes. GCT2 B-default waits for GSR1's recorded successful completion before touching its GPU. B-good requires completed GCT1; BCO1 requires completed OCB1. A busy/failing/missing prerequisite is not bypassed. No existing job is stopped.

The installer is checksum-verified, atomic, serialized across a shared home, idempotent only for identical releases, and refuses overwrites. It installs no packages. All five training backend modules are bundled and hash-pinned; there is no dependency on the old fair/practical backend directory layout. BCO1 still requires the original OCB1 `states/dense12.pt` and its metadata, because the prescribed benchmark reuses that exact engineering state.

## GCT2

Six new parent seeds 161–166, subject to complete recursive live target-project seed inventory under a registration lock. B-good handles 161/163/165; B-default handles 162/164/166. S1-B alone. No substitutions or pilot pooling.

Each seed trains Dense6 for 7,630 updates (LR .0008), copies learned blocks and Adam moments/age into Dense12, runs 200 shared post-growth Adam updates, and forks AdamW/BDT/matched-scrambled through post-growth update 3,000 (LR .0006, cap 3). All scientific candidates/shadows remain as in GCT1. No BCO1 candidate is admitted here.

A new GPU/runtime/source-specific gate tests actual full-size parent/reference equality, growth and optimizer-state transport, prefix equality, active-policy repeats, interrupted/resumed matched control, reference Adam/BDT agreement, and evaluator repeatability. This release adds one independent 200-update parent-reference check to the GCT1-style gate: **2,560 engineering optimizer updates per admitted target**, not a scientific-budget change. REPORT-evaluator engineering checks use seed127; final scientific REPORT remains unavailable until the cohort barrier.

One shared filesystem needs **128 GiB free** initially. Actual gate checkpoint sizes are used to check the projected remaining retained payload. Each GPU child requires 32 GiB free VRAM on two polls; these are admission checks, not reservations against unrelated tenants. All six scientific families retain parents, grown starts, prefixes and arm checkpoints at 1,000/2,000/3,000. Temporary engineering checkpoint payloads alone are removed after their proof records pass and are preserved.

After both lanes finish training, one finalizer verifies all 36 stage results, full payload hashes and CPU-recomputed tensor identities, growth mapping, ages, complete batch/RNG histories, first active proposals, source/cache/runtime and gate links. It commits an immutable 18-endpoint `COHORT_LOCK.json` before any scientific REPORT evaluation. One lane may finish and exit while waiting for the other; the last lane finalizes both. REPORT is the established offset-2,500,000 partition (1,048,576 scored tokens), not newly unseen text. The final evaluator restores the full final state and RNG and verifies the exact locked checkpoint.

The output uses six paired seed differences, ordinary 95% and individual Bonferroni-adjusted 97.5% t intervals. Both adjusted upper bounds must be below zero for joint confirmation. No optional stopping, extra seeds, or automatic follow-up. This confirms the fixed growth recipe, not superiority over GS/scratch or wall-clock savings.

## BCO1

One engineering seed, one candidate, one fixed timing session. It reads OCB1's hash-verified update-500 Dense12 snapshot on OWN-S1-R1, including its input ring and moments. Missing or incompatible snapshots stop; no replacement state is selected or silently regenerated.

First, the reference-only profile separates ordinary and every-tenth-step covariance-refresh updates over 501–520. Its instrumented copy strips back to the identical reference AST and must also pass exact execution comparison. The profiler is never used as a primary speed measurement.

Candidate H1 omits only optional cosine/common-error report calculations. Condition-number and centered-norm guard fields, input/finite/order checks, update arithmetic and covariance/eigen state are unchanged. A new actual-model gate trains a shared 200-step prefix and reference, candidate, and candidate-interrupted-at310/resumed paths through 1,000 (2,600 engineering updates). Every step has gradient/proposal/parameter samples; registered checkpoints have full gradient/proposal/model/Adam/covariance identities and evaluations. The candidate remains in diagnostics-lite mode during these checks even when hashes are collected.

Only after exact gate agreement do 12 counterbalanced blocks in each of two families run (24 total): full step20 and optimizer-only100. Each block brackets split Adam/reference BDT/candidate BDT with identical Adam sentinels. Warm20, then restore common state before each measured window. Final candidate/reference model, Adam and covariance hashes must agree. All complete blocks remain primary; stable-subset reporting follows the fixed10% sentinel rule. The engineering advancement screen remains at least2% median full-step improvement, faster in10/12 blocks and at least8 stable blocks. A partial block stops for review; automatic fast-block selection/device shopping is forbidden.

BCO1 requires32GiB free disk initially and32GiB free GPU memory. Its output does not overwrite OCB1 or imply fewer updates/time-to-quality. New GCT2 training remains pinned to the original reference regardless of BCO1's result.

## Status and logs

Run from the project root (not from `logs/`):

```bash
/opt/conda/bin/python growth_confirmation_cost_v1/study.py status --study gct2 --project "$PWD"
/opt/conda/bin/python growth_confirmation_cost_v1/study.py status --study bco1 --project "$PWD"
```

Coordinator pointers: `logs/gct2_B-good_latest.txt`, `logs/gct2_B-default_latest.txt`, `logs/bco1_A-good_latest.txt`. Current child pointers add `_child` before `_latest`. A PID proves detachment, not validation. Look for `GCT2 FULL-SIZE GROWTH + REPORT GATE: PASS` or `BCO1 FULL-SIZE EXACT-EQUIVALENCE GATE: PASS`, followed by advancing training/timing output.

## Interruption and migration

After inspecting a known interruption, use the corresponding launcher with `--resume`:

```bash
bash growth_confirmation_cost_v1/launch.sh gct2 --resume
bash growth_confirmation_cost_v1/launch.sh bco1 --resume
```

GCT2 uses new gate/job namespaces for a new physical GPU with compatible runtime. Do not move/delete old registrations. Valid completed scientific work is verified before a child log is opened and emits `SKIP_COMPLETE`. OOM counts persist across coordinator restarts; at most three same-recipe OOM attempts. Source, nonfinite, state and equivalence errors remain fatal. A changed runtime is not silently substituted.

BCO1 never mixes completed or partial timing blocks across GPUs/runtimes. Such an interruption needs review and a separately documented complete timing session; `--resume` is not permission to discard an unfavorable/partial block. Keep notebook allocations alive; detachment does not survive cancellation of a server.

## Results and export

GCT2: `runs/growth_correspondence_confirmation_v1/{READOUT.md,analysis.json,endpoints.csv,normalized_reports.json}`.
BCO1: `runs/bdt_cost_optimization_v1/{READOUT.md,analysis.json}`, with full blocks and profiling evidence under `sessions/`.

```bash
/opt/conda/bin/python growth_confirmation_cost_v1/study.py export --study gct2 --project "$PWD"
/opt/conda/bin/python growth_confirmation_cost_v1/study.py export --study bco1 --project "$PWD"
```

Exports collect source, normalized evidence, results and bounded logs into one ZIP under `exports/`, excluding full checkpoints, token arrays and large profiler traces. They are per-file snapshots, not atomic running-cohort snapshots. No individual browser downloads are required for normal inspection.

## Validation boundary

The release includes a CPU admission suite (design/analysis fixtures, operational/migration/barrier tests, candidate transform tests and actual pinned-backend tiny-model integration). Release receipts are supplied in the separate handoff ZIP. These do not replace the full-size H200 gates, which run on target after launch. No new GCT2 or BCO1 GPU workload was executed during bundle creation.
