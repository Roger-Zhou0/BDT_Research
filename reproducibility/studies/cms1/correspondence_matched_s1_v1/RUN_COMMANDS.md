# Run commands

## Location
Upload `correspondence_matched_s1_v1_deploy_2026-09-10.tar.gz` to `~/recurrent-graph/recurrent_graph_s1_124m/` in **Account B / account_b / good**. Downloading to the Mac is not uploading to the notebook. Do not use account_a or the default adaptive-training session.

After checking the supplied archive checksum, extract only if `correspondence_matched_s1_v1/` does not already exist. Do not overwrite an existing installation.

```bash
cd ~/recurrent-graph/recurrent_graph_s1_124m
/opt/conda/bin/python correspondence_matched_s1_v1/study.py integrity --project "$PWD"
bash correspondence_matched_s1_v1/launch.sh --run
```

This single detached launch runs the new target-GPU validation gate and, only on passing it, the six parents, eighteen continuations and locked REPORT evaluations. A printed PID alone is not a passed CUDA gate. The decisive log line is `CMS1 FULL-SIZE ACTIVE-CONTROL GATE: PASS`, followed by a scientific seed-147 parent log. No other study auto-launches afterward.

## Progress
```bash
/opt/conda/bin/python correspondence_matched_s1_v1/study.py status --project "$PWD"
tail -f "$(cat logs/cms1_latest_log.txt)"
```

The outer log reports child starts/completions. Detailed update progress is in the active child's file:
```bash
tail -f "$(cat logs/cms1_current_child_log.txt)"
```
Reopen the child monitor when the queue moves to the next child, or use `study.py status` for the current tail. Ctrl-C stops tail, not the detached coordinator. Do not launch a second queue because the log is quiet.

## Evidence export
```bash
/opt/conda/bin/python correspondence_matched_s1_v1/study.py export --project "$PWD"
```
The timestamped ZIP under `exports/` includes this study's source, protocol, results, checksums and bounded log/trace tails. No full model checkpoints or token arrays are included. It is a per-file snapshot, not a globally atomic snapshot of running jobs.

## Validation-only and interruption
`launch.sh --validate-only` is an alternative first launch that stops before scientific seed training. After it completes, `launch.sh --run` verifies and reuses its completed gate. Do not start these concurrently.

For a real interruption after inspection, use the unchanged package and `launch.sh --run --resume`. A stored fatal invariant/control failure is NOT a routine interruption; do not clear it or replace manifests merely to force progress.
