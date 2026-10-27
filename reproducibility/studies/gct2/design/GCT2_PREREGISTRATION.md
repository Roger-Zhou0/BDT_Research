# GCT2 — independent confirmation of pretrained-growth transfer

**Design date: September 10, 2026. Status: complete specification and tested manifest/analysis tooling; not a registered or launched training study.**

## 1. Question and scope

Does the fixed GCT1 procedure reproduce (a) a beneficial canonical BDT update relative to AdamW and (b) a loss advantage over the tested own-state strength-matched scrambled direction after copying pretrained layers?

This confirms a particular transfer result, not a newly selected optimizer, the best growth procedure, a GS comparison, or end-to-end training efficiency. GCT1's two pilot parents (159–160) are excluded from the confirmation cohort and confidence intervals. GSR1's scratch results cannot change this protocol or determine which GCT2 endpoints are retained. GCT2 remains useful even if the complete growth pipeline fails against a stronger scratch reference.

The scientific treatment remains the original GCT1 implementation, including the three arms' symmetric hypothetical-candidate calculations. The parallel BCO1 performance candidate must not replace it during this study.

## 2. Cohort and assignment

Six **new independent small-parent training seeds**, not additional continuations of the pilot parents. Candidate IDs are fixed at **161–166**, subject to a complete live target-project inventory check. An occupied ID or an incomplete audit blocks registration; no automatic substitution is allowed. The design-time checks of archived metadata and top-level live run listings are not represented as an exhaustive reservation.

| Logical lane | Target account/session | Candidate seeds | Assigned arm order |
|---|---|---|---|
| B-good | account_b/good | 161, 163, 165 | 161: Adam, BDT, scrambled; 163: BDT, scrambled, Adam; 165: scrambled, Adam, BDT |
| B-default | account_b/default | 162, 164, 166 | 162: Adam, scrambled, BDT; 164: BDT, Adam, scrambled; 166: scrambled, BDT, Adam |

All six permutations occur once. Each arm occupies each position once within each three-parent lane. One complete parent family stays on its assigned lane unless a documented infrastructure migration is required. Within-parent arm differences therefore do not compare treatments systematically assigned to different GPUs.

Both lanes use exactly **S1-B**. The default lane waits for GSR1's terminal completion and actual process/lock/memory admission; it does not interrupt GSR1. The good lane can start independently once validated. The target configuration does not use account-A OWN-S1-R1 cache, even though GPUs are available on that account. An additional account requires a separately authorized identical-cache deployment, not substitution of its existing data.

The fixed cache hashes are in `protocols/GCT2.json`. Full token-file hashes, sizes, and stat identities must be checked before admission. Stat checks continue while training. A completed or favorable earlier run is never treated as proof of the current cache contents.

## 3. Unchanged training procedure

**Architecture.** Ordinary Dense Transformer. Six-layer parent: 81,074,688 parameters. Twelve-layer target: 123,551,232 parameters. Width 768, 12 attention heads, FFN width 2,048, vocabulary 50,257, context 512. No RAR factorization, recurrent sharing, or shared FFN bank.

**Small parent.** Initialize independently with the frozen initializer. Train 7,630 updates at peak LR 0.0008, 200-update warmup and cosine decay to 0.1 of the peak. Save final parameters, named moments and optimizer counters, RNG, data provenance, and endpoint records.

**Growth.** Map blocks as `[0,1,2,3,4,5,0,1,2,3,4,5]`. Copy all mapped block parameters, embeddings and final norm, and both Adam moments, into independent storage. Preserve Adam age 7,630 and inherited RNG. Continue global training-data indices from step 7,631. Start a new phase-relative LR schedule and zero both twelve-depth covariance histories. This operation is not function-preserving: record the immediate loss jump without selecting, retrying or excluding a parent because of its jump.

**Shared post-growth prefix.** Run 200 AdamW updates while accumulating both canonical and scrambled proposal histories. Save one immutable post-growth parent per seed containing the model, moments/counters, RNG, both histories and hidden-unit permutations.

**Three branches.** From that exact prefix, run AdamW, cap-three BDT, and the unchanged CMS1-style matched-scrambled construction through post-growth update 3,000. Each branch adds **2,800** updates, not 3,000. Final global Adam age is **10,630**. The post-growth LR is 0.0006, warmup 200, cosine minimum ratio 0.1.

**Common optimizer/training settings.** Adam beta1=0.9, beta2=0.95, epsilon=1e-8, weight decay 0.1, global gradient clipping 1.0. BF16 autocast, microbatch 16, gradient accumulation 4: 32,768 training-token exposures per update. Strict deterministic mode, warn-only false, cuDNN deterministic, benchmark disabled, TF32 matmul disabled, and cuBLAS workspace `:4096:8` set before CUDA initialization. Record actual attention dispatch. Do not install a new library version to get past a compatibility failure.

**BDT.** Whole twelve-depth FFN gate/up/down stacks; centered depth dimension eleven. Covariance observations every ten updates, EMA beta 0.9 per observation, 200-update inactive prefix, 200-update ramp, operator cap 3.0. Preserve the exact reference arithmetic and Adam moments in parameter coordinates. Do not normalize each two-layer copy pair separately: that is not the registered treatment.

**Matched control.** Canonical candidate, scrambled-derived candidate, and matching are computed at each branch's own current state. Retain the common-depth component, centered norm and angle from Adam prescribed by the original geometry. FP64 matching with FP32 applied proposal. Use the frozen tolerances and refusal rules in the JSON. No random fallback. Log identity/numerical-zero cases, maximum residual amplification, and all matching errors. The GCT1 pilot had no inactive post-onset control events; that fact does not justify discarding or silently replacing such events in GCT2.

## 4. Endpoints and delayed REPORT access

During training, use the inherited DEV evaluator at token offset 0: 2,048 contiguous records of 513 tokens, scoring 1,048,576 next-token targets. Monitoring does not permit recipe changes, early efficacy stopping, or selective parent replacement.

The primary endpoint is **final post-growth update-3,000 REPORT loss**, using the established S1-B REPORT partition at token offset **2,500,000**, 2,048 records of length 513, scoring **1,048,576 targets**, and ending at token index **3,550,624**. This partition is copied from the full CMS1 protocol, not inferred from the DEV offset. It lies within the existing five-million-token validation file.

This is **previously used evaluation text**, not a newly unseen test set. The new replication is in independent training parents and a fixed prospectively specified recipe.

REPORT evaluation is unavailable to the scientific loop. A finalizer first validates all six independent parents, six prefixes and **18 complete final training records/checkpoints**, then atomically commits one immutable cohort lock. Only after that lock may it evaluate the 18 REPORT endpoints. All report records identify the locked training result and checkpoint hashes, evaluator source, partition, token count, and start/end times. Two lanes must not launch concurrent report finalizers; use a shared finalizer lock and idempotent verified outputs.

An unfinished cohort may show operational counts and DEV curves, but not an inferential REPORT result. No old or pilot endpoint fills a missing cell.

## 5. Statistical analysis, fixed before execution

For each of the six training-parent seeds calculate:

- Directional contrast: final loss(BDT) minus final loss(matched scrambled).
- Constructive contrast: final loss(BDT) minus final loss(AdamW).

Publish all 18 absolute endpoints, the six paired values for each contrast, mean, sample SD, and ordinary 95% paired Student-t intervals. Also publish **97.5% two-sided intervals for each of the two contrasts**, the prespecified Bonferroni family-wise adjustment. With df=5, the critical values are 2.5705818356363146 and 3.163381449748609 respectively. These nominal t intervals rely on approximately normal, independent between-seed differences; six parents do not guarantee power or universal validity of that approximation.

**Joint support:** both adjusted upper confidence bounds are below zero and the validity checks pass. All six signs being favorable is descriptive, not a substitute for the prespecified analysis or an additional requirement for success. Zero observed between-seed variance triggers review rather than automatic zero-width confirmation. A wide interval is inconclusive, not evidence of equivalence.

Interpret contrasts separately. A supported constructive contrast without the directional contrast supports incremental utility but not confirmed directional transfer. The reverse supports the discriminating-control difference without a resolved AdamW improvement. Both unfavorable or unresolved is reported as such. Do not add parents, adjust alpha, change the reporting text, or combine pilot and confirmation to alter the decision.

The original E21 pilot advancement threshold remains unchanged. No new arbitrary 0.005 loss threshold is imposed on the confirmation: practical magnitude and statistical support are reported separately.

`tools/analyze_gct2.py` implements this prospective analysis on validated normalized records. Its metadata checks do not replace the future collector's actual source, history, tensor and checkpoint audits.

## 6. Required implementation gates

Before new scientific parents, each admitted GPU/runtime runs the actual-sized growth and active-control gate. Retain the GCT1 engineering recipe (seed 127, shortened 200-update small parent, two post-growth prefixes and active controls through phase 420), plus tests of the new two-lane scheduler, cohort barrier and REPORT evaluator. The gate is outcome-blind: passing depends on numerical and provenance agreement, not whether BDT wins.

Required checks include exact frozen-reference parent/prefix behavior; mapped weights and moments in distinct storage; Adam/global/phase counters; inherited RNG/data index; initially zero covariance histories; active repeatability; interrupted/resumed continuation; matching invariants; and refusal of damaged checkpoint/source/cache records. Verify a known final state evaluates reproducibly on the registered REPORT slice without changing training RNG. REPORT-gate evaluation uses engineering seed 127 only, never an early scientific checkpoint.

Keep the training arithmetic tied to GCT1. The new confirmation's operational source has its own manifest: inheriting tested arithmetic is not a claim that a new scheduler, exporter or evaluation stage is already CUDA-validated.

## 7. Crash recovery, migration and log hygiene

Scientific job identity is `(protocol, seed, stage, arm)`; attempt records separately include the physical GPU UUID/runtime and checkpoint lineage. Engineering job registrations must be namespaced by **lane + GPU UUID + runtime digest + source digest**. This fixes the old ODC collision between a new GPU-specific gate and a lane-only job-registration path. Preserve old engineering records; do not move them merely to trick an equality check.

On resume, validate a completed job **before** allocating a child or opening its child log. Emit a coordinator-level `SKIP_COMPLETE` with the result hash. This avoids the previous series of empty logs while preserving the audit trail. A skip must not be inferred from a filename or file size alone.

Write checkpoints every 250 updates and at stage endpoints, using write/fsync/hash/atomic publication. Keep small-parent finals, grown starts, shared prefixes, and branch steps 1,000/2,000/3,000. Resume from the newest valid committed checkpoint; never hand-edit its phase counter. Infrastructure interruption may repeat the same work but not change the recipe. Durably cap OOM attempts at three **across restarts**, not three per restarted coordinator. Fatal invariant/nonfinite/source/cache errors block further execution until investigated.

On a new GPU/runtime, validate the new target before continuation and record migration. Identical starting checkpoints do not by themselves guarantee identical floating-point trajectories. Runtime incompatibility stops rather than rewriting historical contracts. Locks coordinate our own workers; available-memory polls are not reservations against other users.

## 8. Resource ledger

Scientific work comprises 45,780 six-layer parent updates, 1,200 shared twelve-layer prefix updates, and 50,400 twelve-layer branch updates: **97,380 total optimizer updates**, or **3,190,947,840 campaign token exposures**. These are mixed model sizes, not homogeneous FLOPs. Each recommended complete growth path consumes 348,323,840 exposures, including its parent cost in full.

With the fixed allocation, each lane has three parents and nine branches: 48,690 mixed-size scientific updates. Engineering gates and failed/replayed attempts are additional and recorded separately. No completion-time promise follows from historical shared-GPU throughput.

Approximate retained FP32 parameters-plus-two-moments payloads are about 96.6 GiB for the complete six-parent campaign, before metadata and transient saves. Admission planning uses **128 GiB of free shared filesystem space for the cohort** and at least **32 GiB free GPU memory on two polls per child**. Measure actual payloads during the engineering gate and refuse insufficient capacity; do not delete historical evidence. Free-space readings may not represent every applicable quota.

## 9. Explicit exclusions

No GS arm, new LR bracket, new growth mapping, longer horizon, larger model, additional ODC seeds, optimized BCO1 treatment, or automatic efficiency-confirmation study is included. A stronger practical-efficiency study is a separate decision after the fixed GSR1 endpoints and fair baseline-selection design are assessed.
