# Preregistration: correspondence_matched_s1_v1

## Provenance and scope
Specified September 10, 2026, after the prior development studies, reviews and deterministic-execution diagnostic, before the new six-seed efficacy cohort. The locked code-level specification is `cms_protocol.SPEC`; its canonical digest is `PROTOCOL_SHA`. CPU/synthetic and target-GPU engineering tests use the previously used development seed 127, not the new scientific seeds.

## Hypothesis and fixed comparison
At matched realized intervention strength in one common coordinate system, ordinary corresponding-coordinate BDT produces more useful training directions than a direction computed after scrambling cross-depth hidden-unit correspondence. The primary contrast is final REPORT loss BDT minus strength-matched scrambled. The constructive contrast is BDT minus AdamW.

Explicit-cloned S1, 123,550,560 parameters, S1-B on account_b. Six reserved seeds **147–152**, three arms: `adamw`, `bdt`, `scrambled_matched`. Reserved-seed conflicts cause refusal, never automatic replacement. Each seed has one immutable update-200 parent followed by three 2,800-update continuations. Total path length is 3,000, not 3,200. Execution order uses all six permutations of the three arms, one per seed.

Peak LR 0.0012; LR warmup 200; cosine minimum ratio 0.1; context 512; microbatch 16; accumulation 4; 32,768 token exposures/update; BF16. AdamW betas (0.9,0.95), epsilon 1e-8, decay 0.1, global gradient clip 1. Cap-three BDT begins changing updates at 201, reaches full ramp at 400, observes raw proposal covariance every ten steps with EMA coefficient 0.9. No LR/cap/schedule search is included.

## Shared states and intervention
The parent stores model, named Adam states, RNG, fixed per-depth hidden permutations, and both canonical and scrambled raw-proposal covariance histories accumulated under identity-applied warmup. Parent bytes and all these state identities are checked on load. All arms compute canonical BDT, scrambled BDT, and the strength-matched candidate at their own current states; only the applied private-FFN proposal differs. This equalizes the shadow calculations and is not an optimizer-overhead benchmark. Attention and all other parameter updates follow the same frozen AdamW path.

Gate/up rows and down columns use consistent hidden-unit permutations within each role. Permutations are generated from a dedicated hash-seeded CPU generator, not the training RNG. No trained model weights are reindexed: this is a proposal-space diagnostic, not a claim that the third arm is unmodified BDT on a permuted full model.

The control restores the raw common-depth component and centered norm, and matches the angle of the canonical BDT correction computed at that branch's current state. Its orthogonal direction comes from the mapped-back scrambled correction. Geometry is evaluated in FP64 and the proposal is cast back to FP32; actual cast values are checked for norm, common-component, angular and displacement errors. The ordinary BDT path retains the frozen FP32 operator.

Numerical rules fixed in `cms_protocol.GEOMETRY`: canonical sine <=1e-7 returns raw and logs a numerical-zero event. For nonzero target angle, scrambled orthogonal norm/raw-centered norm <=1e-7 is a fatal undefined-direction condition; no random fallback. Relative norm/common tolerances 1e-5; angular absolute tolerance 5e-6 radians; normalized displacement tolerance 2e-6 + 2e-4 times target displacement. Failures stop the campaign without selecting another seed or silently changing thresholds.

## Execution and integration gate
All arms use strict automatic deterministic settings: CUBLAS_WORKSPACE_CONFIG=:4096:8 before PyTorch import, deterministic algorithms enabled with warn_only=False, cuDNN deterministic=True and benchmark=False, matmul TF32=False. This is a new execution cohort, not a retrofit to historical records.

The new full-size CUDA gate uses seed 127 and runs two 200-step parents plus eight active continuations through step 420, including repeated executions, an interrupted/resumed scrambled trajectory, and frozen AdamW/BDT references. It checks actual real-proposal strength invariants, matching first active gradients/raw proposals, sampled arrays, saved-state restoration and deterministic records. At most 2,160 optimizer updates are planned before the scientific cohort. The CPU gate retains its own synthetic-data test and does not read REPORT.

## Evaluation, analysis and stopping
DEV: offset 0; 2,048 records with 512 predicted tokens each. REPORT: existing S1-B offset 2,500,000; 2,048 records; 1,048,576 predicted tokens. REPORT is evaluated only after all eighteen final training checkpoints and results are frozen. This is the same REPORT partition used in earlier work, NOT a newly unseen dataset. No treatment is selected based on intermediate DEV curves.

Analysis reports all six per-seed endpoints and differences, means, standard deviations, paired Student-t 95% intervals, and Bonferroni 97.5% intervals for the family of two comparisons. Six training seeds are the replication units, not eighteen branches. No equivalence claim, adaptive seed count, automatic hyperparameter expansion or follow-on efficacy study is registered. Simultaneous intervals below zero on both contrasts support the combined positive conclusion within this setting; unresolved intervals remain unresolved.

Source/control/geometry/nonfinite failures stop before the next job. Up to three OOM attempts reuse the identical recipe and last committed checkpoint; batch size is never reduced. Checkpoint resume restores RNG, Adam, covariance and permutations. Only superseded rolling checkpoints belonging to this new trial are removed; immutable final parents are retained.

## Storage and output
Scientific work: 6*200 + 18*2800 = **51,600** optimizer updates, 1,690,828,800 actual training-token exposures. Each 3,000-update path contains 98,304,000 exposures. Require 64 GiB initial free disk for retained gate/parent/final checkpoints and headroom, and at least 32 GiB free GPU memory before child admission. Shared physical-GPU timing is observational only.
