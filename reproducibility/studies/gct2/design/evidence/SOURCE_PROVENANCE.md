# Source basis and validation scope

The design is based on the approved `POST_OCB1_FORWARD_PLAN_2026-09-10.md`, the verified OCB1 final readout, the GCT1 implementation/handoff, and the actual archived code. It does not infer new efficacy results.

## Frozen scientific references

- GCT1 scientific source-manifest digest: `cb703e1fb6f619ae7044cb47807483be375f47b4dbeb7c7a75af09b7e6ad5048`.
- GCT1 protocol digest: `f3cd9097f66360d4e71a5932c780d3e216520f7a29222f6fe4a82bb44a975298`.
- Original `tempering_reference.py` SHA-256: `d1bd72d9ccd18019aea83551b838d5915f727a0a7b6a0a2fd1006fb1c47969f1`.
- Original full `cms_protocol.py` SHA-256: `e941b4274b27b1642e9467fdafe144fa79c378c199d2b5d35358fb7428261b7c`. This was read on the connected Mac to establish the REPORT offset 2,500,000 and geometry tolerances; the shortened constants file in the timing package is not substituted for the full protocol.
- OCB1 implementation digest: `e3a4d782bb662ce6eec7674aca00ce16304aa29d87673f101dbbfe4392b80d99`.
- Proposed diagnostics-only candidate SHA-256 from its actual CPU check: `ef3616d538a806cd01fe8941742f62beb2167e53c87b74f10fbd6f75d50c4ab5`.

## New executable tools versus a GPU release

The 49 local CPU fixtures exercise manifests, budgets, ordering, seed-inventory errors, complete-cohort analysis and timing-record validation. They do not train a model.

The separate connected-Mac test ran six actual reference/candidate proposal sequences on CPU using PyTorch 2.6.0, including FP32 through 1,000 steps. It checked exact tensors and covariance/eigen/counter state, retained reporting fields and unchanged global RNG, plus nonfinite/order/shape refusal. Its restore test is an in-memory state round trip, not a killed-process filesystem-resume experiment. The reference bytes were unchanged.

That check used the same source transformation supplied by `candidate/prepare_candidate.py`. The more detailed `candidate/test_candidate.py` is supplied for repeatable local use; the recorded remote six-scenario check was an independent inline verifier, not a claim that this entire test-file suite was executed remotely. No new full-model candidate CUDA test or performance measurement has occurred.

The design-time inventory checked archived metadata and fresh browser-rendered top-level run directory metadata in the allowed profiles. It did not complete a recursive live target-project seed scan or reserve IDs. Candidate IDs 161–166 cannot be used until the mandatory admission/registration step succeeds.

A transfer checksum mismatch in an initial candidate-script upload was rejected before files were created or any tests executed. The successful candidate check was subsequently built directly from the exact archived reference and recorded separately. This was local engineering work; no notebook training source or result was changed.
