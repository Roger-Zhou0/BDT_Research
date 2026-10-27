# BCO1 — one bounded reference-preserving BDT cost-reduction study

**Design date: September 10, 2026. Status: specified; a diagnostics-only transform prototype has CPU checks. No new CUDA profile, full-model equivalence result, or speed measurement is claimed.**

## 1. Objective

Attribute the cost of OCB1's already-lean canonical BDT implementation, remove avoidable work without changing its treatment, and measure the candidate against the frozen reference contemporaneously. This is engineering support for the existing scientific result, not optimizer selection on validation loss.

OCB1 measured approximately 8.93% additional full-step time versus FFN-split AdamW, despite already excluding the scrambled shadow path. Its separate audit paths cost about 41–42% above the split baseline. Turning off those shadows is therefore not a new solution to the remaining canonical cost.

Target `account_a/good`, after OCB1 completion and fresh capacity checks. Keep GCT2/GSR1 and all prior results unchanged. Use no new scientific seeds and no REPORT-based decisions.

## 2. Stage A: profile the actual reference, separately from timing

Use ordinary Dense12 at the same context/microbatch/accumulation/BF16 configuration as OCB1. Start from its fixed, hash-verified engineering update-500 state, including canonical covariance history, moments and fixed input ring. The state is not selected by validation performance. Validate the source/runtime and state compatibility before reuse; do not silently regenerate a different state if the saved one is unavailable.

Profile updates **501–520**. Separate ordinary steps from covariance-refresh steps **510 and 520**. Attribute the actual call tree to proposal formation, centering and projection to depth coordinates, normalized covariance formation/EMA, eigendecomposition, operator application, norm restoration, safety checks, optional reporting, proposal application/decay, allocations, host/device copies and synchronization.

Use an isolated instrumentation copy whose tensor expressions and order match the reference. Record source hash and verify raw/applied proposals and optimizer/model state outside the profile against the uninstrumented path. Store CPU and CUDA event keys and a trace plus a compact summary. Report exclusive durations when available; nested profiler regions and overlapping CPU/GPU times are not additive. A scalar conversion may expose a wait caused by an earlier GPU operation, so do not assign every wait to the Python line that observes it.

The profiler is absent from all primary timing windows. No profiler-derived duration becomes the final speedup estimate.

## 3. Stage B: one explicit candidate

**Candidate H1: `diagnostics_lite_v1`.** The archived `OnlineDepthTemper.transform` computes the raw-update cosine and maximum common-component error on every call, including when its caller discards those report fields. The first candidate omits only those two optional reporting calculations when `diagnostics=False`.

It retains all output-producing tensor expressions in their original order, the covariance/eigenvalue updates, zero/numerical branches, cap, precision, every-tenth-step observation schedule, input validation and nonfinite checks. It still computes `operator_condition` and `centered_norm_ratio`, the values used by the outer production safety guard. Full reporting remains available, with the original fields, on requested diagnostic calls.

This is a **source-derived optimization hypothesis**, not a measured attribution that those calculations dominate runtime. Source inspection alone cannot say how much of the 9% will disappear. Even exact CPU equality cannot establish full CUDA trajectory equality.

`candidate/prepare_candidate.py` accepts only the exact archived reference SHA-256 `d1bd72d9ccd18019aea83551b838d5915f727a0a7b6a0a2fd1006fb1c47969f1`, creates a separate candidate file, refuses overwriting a different file, and never changes the reference or historical study manifest. Its default is full diagnostics; a new BCO1 candidate adapter must explicitly request the lean mode.

Only one candidate implementation is allocated in this initial pass. Do not proceed automatically to new caps, lower precision, approximate eigendecompositions, changed observation frequency, altered norms or different learning rates. If profiling identifies an entirely different worthwhile optimization, record it and make a new scope decision rather than silently expanding this study.

## 4. Stage C: mandatory equivalence gates

**CPU transform checks.** Small proposal stacks with twelve depth rows, FP32 through step 1,000 and FP64 through step 420; full-report mode; inactive observation; zero/common-only cases; saved-state restoration at step 310; nonfinite/order/shape refusal. Require exact applied proposals and covariance/eigen/counter state. Compare retained report fields exactly. Optional omitted reporting fields are allowed to be absent; timing differences are not numerical failures.

**Full-size training check.** Engineering seed 127 only. Build a shared 200-update Adam prefix and run reference BDT, uninterrupted candidate BDT, and interrupted/resumed candidate BDT through update 1,000. Interrupt the third path at 310 and restore its committed state. Total planned model-training work: **2,600 updates** (200 shared + three 800-update continuations), excluding retries and later timing warmup.

Require agreement of full raw/applied proposals, model tensors, moments/counters, covariance/eigen state, batch/RNG histories and fixed evaluations at registered checkpoints. Include pre-onset, ramp and mature active steps. Hash/sample collection is outside timing. Record every failure; no relaxed tolerance introduced after seeing it. A mathematically equivalent candidate that is not bit-identical under the stipulated checks is a separately labeled treatment and does not inherit the reference's quality results.

Existing GCT1/OCB1 gates are evidence about their own implementations. They cannot be reused as proof of the new candidate's equivalence without this new check.

## 5. Stage D: paired before/after measurement

Use **three conditions**: unchanged FFN-split AdamW, frozen lean canonical BDT, and candidate BDT. Native fused AdamW and GS need not be rebenchmarked to isolate this one change; their original OCB1 measurements remain unchanged. Do not multiply an old native-Adam ratio into new measurements as if it were a contemporaneous observation.

Run the six permutations of these three conditions twice: **12 fixed blocks per family**, balanced over every position. Each block includes an identical split-Adam sentinel immediately before and after its three methods. Two families:

- Full step: 20 updates, with the same CPU input ring and H2D, forward/backward, clipping, proposal transformation and optimizer application included.
- Optimizer only: 100 calls using the same saved clipped gradients, excluding backward and clipping.

Both include the mandated covariance refresh cadence. Warm each method for 20 updates, then restore the same common state, RNG and history **after warmup** before the measured window. Timing begins at update 501. Capture synchronized wall time as primary, CUDA-event elapsed time as secondary, and memory separately. Exclude setup/restoration, checkpoint I/O, validation, audit hashes and profiling from primary timers. The before/after optimizer states must agree between the candidate and reference for every timed window as well as in the earlier training gate.

All comparisons use one GPU UUID/runtime, one common state/input identity, and one reference/candidate pair of source digests. Record telemetry and interference. A GPU migration creates a new complete timing session, never a pool of the fastest blocks from multiple devices.

## 6. Quality control and decision

Preserve all twelve completed blocks in the primary descriptive result. Flag a whole block if `abs(sentinel_after - sentinel_before) / sentinel_before > 0.10`. Show the predefined stable subset only secondarily. Fewer than eight stable full-step blocks makes the main result inconclusive; do not repeat until a favorable block set appears.

Report median paired candidate/reference, reference/split and candidate/split time ratios with interquartile ranges. These describe timing repetitions, not training-seed uncertainty. Retain all memory and optimizer-only results even when they look less favorable.

**Engineering advancement screen:** all equivalence checks pass, at least eight full-step blocks meet sentinel stability, the candidate is faster in at least ten of twelve blocks, and median full-step time is at least 2% below the contemporaneous reference. This is a resource-allocation screen, not a claim of statistical confidence in an uncontended-system speedup.

The aspirational target is to halve residual overhead relative to split Adam in this *new* paired session. The target is not a required paper result and does not justify broadening the optimization search. Missing it or finding no gain ends this initial engineering pass with a documented outcome.

Passing sentinel checks does not prove exclusive use or eliminate stable interference. A candidate can improve optimizer-only timing without a meaningful full-step gain; report both instead of presenting the larger optimizer-only factor as training speedup.

## 7. Separation from confirmation and publication claims

GCT2 remains pinned to the unmodified scientific reference regardless of BCO1 outcome. A validated candidate becomes a separately versioned deployment implementation; it does not retroactively change GCT2 source provenance or overwrite the OCB1 table.

BCO1 does not establish fewer tokens/updates to target quality, total-growth-pipeline speedup, or a hardware-independent lower bound on BDT cost. Any later time-to-quality comparison needs its own target, tuning, cost ledger and complete-path measurements. The current logical-MAC GSR1 budget remains 7,922 updates, not an endpoint retrospectively adjusted using timing.

## 8. Deliverable boundary

The design package supplies a candidate builder, CPU tests and prospective analysis tooling. It does **not** contain a completed full-model benchmark adapter, instrumented CUDA collector, or new speed result. Historical `launch_practical_followup.sh ocb1` runs OCB1, not BCO1; do not relaunch it expecting this design to be installed.
