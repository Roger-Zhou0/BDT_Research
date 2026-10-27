# Approved follow-up design: GCT2 + BCO1

Prepared September 10, 2026 after the approved post-OCB1 plan.

## Two independent workstreams

**GCT2:** six new independently pretrained parents and eighteen continuations, keeping the successful GCT1 recipe fixed. Candidate IDs 161–166 are not yet reserved. B-good owns 161/163/165; B-default owns 162/164/166 after GSR1. Both use the exact S1-B cache. The two final paired REPORT contrasts use the same two-contrast Bonferroni convention as CMS1. No pilot pooling, optional expansion or early REPORT evaluation.

**BCO1:** profile lean canonical BDT, test one diagnostics-only implementation candidate, and measure it against the frozen reference and split Adam after exact equivalence gates. It cannot change the GCT2 treatment. No kernel-efficiency result is claimed from source inspection or CPU tests.

## Files

- `GCT2_PREREGISTRATION.md`: scientific protocol, endpoints, inference, failure handling, resources and claim boundaries.
- `BCO1_PROTOCOL.md`: profiling, one source-informed candidate, equivalence and balanced timing plan.
- `protocols/`: machine-readable specifications, all 36 scientific-stage jobs, and resource accounting.
- `tools/`: pure-CPU manifest generation, read-only inventory audit, and strict prospective analyses.
- `candidate/`: exact-reference candidate builder and CPU transformation tests. The full model and CUDA adapter/collector are not included.
- `tests/`: 49 synthetic design/admission/analysis tests, not experiment outcomes.
- `evidence/`: test log, source provenance, limited design-time inventory record and CPU prototype-check record.
- `RUNBOOK.md` and `INPUT_SCHEMA.md`: implementation/release requirements and collector data contracts.

## What is complete

The written designs, budget, counterbalanced job orders, complete-cohort/refusal logic and testable reporting-only prototype are supplied. All 49 design/analysis fixtures passed. Separately, actual CPU reference/prototype proposal and covariance-state checks passed, including FP32 through 1,000 updates and an in-memory restoration at 310.

## What this is not

This is **not yet a deployable GCT2/BCO1 training-and-benchmark release**. No new scientific parent, REPORT evaluation, full-model candidate CUDA gate or candidate speed measurement has been run. The new GPU coordinators, collectors and final evaluation barriers still need implementation/integration validation. Neither study was registered or launched on the notebooks during this design turn. The candidate cannot silently inherit full-model CUDA equivalence from a small-stack CPU test.

The original GCT1, ODC, OCB1 and GSR1 sources/results are unchanged. No individual evidence files were downloaded into the user's Downloads directories.
