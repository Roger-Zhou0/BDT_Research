# CMS1: strength-matched correspondence study

A bounded six-seed, three-arm experiment on explicit-cloned S1. This package uses the exact already-installed `fair_depth_lr_bracketing_v2` backend; it does not overwrite or bundle replacement training sources.

## Current validation scope
The CPU geometry and real-backend tiny-model integration suite passes. The full-size CUDA gate for this NEW control must run on the target notebook. Older identity-prefix CUDA results do not substitute for the active-control gate.

Use `bash correspondence_matched_s1_v1/launch.sh --run` in **account_b/good** after verifying the release archive. The detached coordinator first checks sources, S1-B cache, seed conflicts, CPU integration, available GPU memory, disk and physical-GPU locks. It then runs the full-size active-control gate. Only on success does it run six immutable update-200 parents and eighteen continuations through total update 3,000.

`--validate-only` stops after validation instead of training the six-seed cohort. Do not run both commands at once. Completed gates and completed trajectories are reused only after their checks. An interrupted campaign requires explicit `--resume`; invariant and nonfinite failures require inspection, not blind retries.

The launcher is fixed to the account and notebook name, not to an old GPU UUID. The coordinator discovers device index 0, locks its current physical UUID and checks that it stays unchanged. A changed GPU on a later invocation gets its own full-size gate. Recorded runtime/configuration contracts must still agree.

See `PREREG.md` for design and boundaries; `RUN_COMMANDS.md` for launch/monitor/export commands. Archive checkpoints and results belong only to `runs/correspondence_matched_s1_v1/`. No earlier result, token cache or backend file is modified.
