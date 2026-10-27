# Execution contract and release checklist

This file specifies the **future launcher behavior**. It is not itself a GPU launcher.

## Allocation

1. `account_b/good`: GCT2 B-good, candidate parents 161/163/165, three complete families.
2. `account_b/default`: finish GSR1 unchanged, then GCT2 B-default, candidate parents 162/164/166.
3. `account_a/good`: BCO1 engineering profile, one candidate, equivalence, before/after timing. No simultaneous scientific training on the same physical GPU during accepted timing windows.
4. `account_a/default`: no extra optimizer search is scheduled merely to occupy the fourth GPU. Retain capacity for recovery or a separately justified baseline-fairness study.

## Before registration

- Verify exact installed backend bytes at the resolved paths. Package local hash-pinned dependencies or explicitly resolve approved byte-identical locations; do not assume `fair_depth_lr_bracketing_v2` exists on every account.
- Check no conflicting live job, old active coordinator, or held lane/job lock. A PID alone is not portable across restarted container namespaces; inspect command/start-time/session identity.
- Audit candidate seed names and registered metadata under the live target project. Record scope, errors, hashes, and time. Stop on an occupied ID or incomplete audit. A complete audit is not an exclusive reservation until the shared registration lock commits the cohort.
- Verify S1-B metadata/train/validation hashes and expected sizes. Do not change/cache-copy data silently.
- Verify resource headroom and create the complete immutable six-parent job manifest before any new scientific training. Do not infer memory availability from 0% GPU utilization.

## Engineering admission

Use seed 127 only. Gate paths and registrations include logical lane, current physical GPU UUID, runtime digest and source digest. Lock by logical job and physical GPU. Passed-gate evidence is reused only if all identities and evidence hashes match.

## Scientific jobs

Skip a validated completed result in the coordinator **before** spawning a child or creating a new child log. Log one concise `SKIP_COMPLETE` line with result hash. This is the intended fix for empty resume-log clutter; existing empty logs need not be deleted.

Persist OOM attempts across coordinator restarts. Never modify batch size, cap, data, seeds or steps as a recovery action. Atomic checkpoint publication and exact state restoration remain mandatory. New hardware requires target revalidation and a migration record, not renaming or deletion of old gate records.

## Reporting barrier

The two lanes share one finalizer lock. Only a complete, validated cohort can produce `COHORT_LOCK.json`, after which REPORT evaluation is admitted. Repeated finalization is idempotent only for matching immutable inputs. Every report is tied to its checkpoint, source and partition. Missing/failed records produce an incomplete readout, not an inferred result.

## Release tests required beyond this design package

1. Tiny-model GCT1-equivalence tests for parent, growth, prefix and active continuations.
2. Exact report-slice and RNG-preservation fixtures, family-wise statistics and complete-cohort refusal.
3. Two-lane concurrency, duplicate-launch, complete-skip, atomic-finalization, persistent-retry and UUID-migration fixtures.
4. Actual target-GPU growth/reference/resume gate before scientific parent admission.
5. BCO1 full-model reference/candidate equivalence and genuine timed collectors before any overhead claim.

## CPU tools available now

Run all design/analysis fixtures from this directory:

```bash
python -m unittest discover -s tests -v
```

A read-only live seed audit (not registration or launch):

```bash
python tools/admission.py --project ~/recurrent-graph/recurrent_graph_s1_124m
```

Generate the isolated CPU-testable candidate, without overwriting any historical source:

```bash
python candidate/prepare_candidate.py \
  --reference ~/recurrent-graph/recurrent_graph_s1_124m/practical_advantage_v1/tempering_reference.py \
  --output /tmp/bco1_diagnostics_lite_v1.py

python candidate/test_candidate.py \
  --reference ~/recurrent-graph/recurrent_graph_s1_124m/practical_advantage_v1/tempering_reference.py
```

The reference path may use the `fair_depth_lr_bracketing_v2` copy instead, but only an exact matching hash is accepted. The output is a prototype for tests, not an instruction to replace a running optimizer.

Do not use the old GCT1, ODC or OCB1 launcher to run this new design. No new GPU launch command is included until the new coordinator/collectors and their integration gates are implemented and tested.
