# Prospective collector contract

The analyzers in this package validate **normalized evidence from a future collector**. They are not collectors, training runners, or substitutes for reading original checkpoints. The fixtures in `tests/test_design.py` are deliberately synthetic and must never be used as experiment evidence.

## GCT2

`tools/analyze_gct2.py <bundle.json>` requires one JSON object with:

- `schema: gct2_endpoint_bundle_v1`; the exact `protocol_digest` of `protocols/GCT2.json` under `tools/design.py:digest`; candidate cohort `[161,162,163,164,165,166]`; the exact S1-B `cache_sha256` and full `report_partition` descriptor.
- `parent_cost_counted_per_path: true`.
- `inventory`: `complete: true`, `collisions: []`, and `manifest_sha256`, derived by a collector from the complete target inventory/registration evidence. A design-time name scan is insufficient.
- `checks`: source, cache, growth_mapping, optimizer_age, all_parents_distinct, lineage, first_active_inputs, full_batch_rng_histories, matching, final_checkpoints, target_gates; all true only when the collector actually performs and preserves those audits.
- SHA-256 identities for `training_source_sha256` and `collector_source_sha256`.
- `cohort_lock`: state `sealed`, timezone-bearing `sealed_at`, and 18 endpoint records. Each endpoint gives seed, arm, step 3000, Adam age 10630, parent steps 7630, prefix steps 200, training completion time, training-result hash, final-checkpoint hash, small-parent hash, and shared-prefix hash.
- `reports`: exactly one final REPORT result for each locked endpoint. Each identifies seed, arm, positive finite loss, REPORT, 1,048,576 scored tokens, offset 2,500,000, started/completed timestamps, the linked training-result/checkpoint hashes, and both source hashes.

All reports start after the complete cohort lock. Each seed has one common parent/prefix across its three arms; the six parents must be distinct. The collector preserves raw results, contracts, report records, checkpoint links and its audit outputs in an evidence manifest. It must verify the checkpoint payloads it evaluates; merely asserting that a hash string has length 64 is not that verification.

The analyzer reports both contrasts regardless of outcome. It never silently drops incomplete or invalid rows. Tests exercise structural rejection; they do not certify future data as authentic.

## BCO1

`tools/analyze_bco1.py <blocks.json> --family full_step` (or `optimizer_only`) expects 12 normalized complete blocks from one source/runtime/GPU/common-state identity, with the exact prescribed order. Each block records a successful equivalence gate, three measurements, and bracketing split-Adam sentinels.

Every measurement has its condition, update window (501–520 or 501–600), positive synchronized wall time, common-state hash, post-warmup-restoration flag, false profiler/shadow-matching/in-timer-hash flags, and expected covariance-observation count. Candidate/reference final model, Adam, and covariance hashes must agree. Sentinel condition is split Adam. The future raw collector also preserves CUDA time, memory, timestamps, input identities and telemetry; the compact analyzer schema is not the full evidence archive.

All 12 blocks remain in the primary result. At least eight stable blocks are required for a qualified cost screen. A profiler run or partial block set is not accepted as a timing result.
