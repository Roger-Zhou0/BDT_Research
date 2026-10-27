# Logs: preserve the record; publish an auditable view

The reviewer package is not a dump of the experiment filesystem. Its core is
source, configurations, data provenance, the registered evaluation contract,
complete endpoint records and deterministic analysis. Include additional log
excerpts when they explain an execution issue or are necessary to verify a
reported quantity. Keep original operational records in their existing archive.
No original logs are deleted or edited by the tools in this package.

## Three distinct cases

| Situation | Treatment |
|---|---|
| A terminal or chat view is shortened, while the source file is intact | Prefer the actual source file. Otherwise declare an excerpt; do not present the copied screen as a complete log. |
| The final JSONL record is incomplete or a record is malformed | Retain its bytes in the derived view and flag its line number. It is not a usable final metric. Never invent the missing suffix or value. |
| Training stopped and resumed | Keep attempt identifiers, interruption evidence and restart lineage. Do not count attempts as independent seeds or concatenate overlapping steps into an apparently uninterrupted trajectory. |

A valid JSON record without a final newline is not, by itself, proof of truncation.
A complete copy of one supplied file does not prove that the original capture or
the training run was complete. An exception in a log does not by itself show that
a subsequently resumed trajectory is invalid. Establish completion from the
study's endpoint/checkpoint/evaluation bindings and registered cohort barrier.

For a reported curve, stitch only from explicit committed-checkpoint lineage and
record the overlap rule in a separate derivation. Preserve discarded replay
segments in the original records. The tool here deliberately does not stitch,
deduplicate, smooth, interpolate, fill missing values or select endpoints.

## Bounded exporter

Create a private specification with an explicit file list. Paths are local to the
machine running the tool; this tool never downloads anything. Use neutral source,
run and attempt identifiers in the exported views. `run_state` and
`resumed_from_attempt` are declarations, not conclusions verified by the tool.

```json
{
  "inputs": [
    {
      "source_id": "source_01",
      "run_id": "study_seed_001_arm_a",
      "attempt_id": "attempt_02",
      "path": "existing_logs/attempt_02.jsonl",
      "format": "jsonl",
      "capture_scope": "file_snapshot",
      "run_state": "resumed",
      "resumed_from_attempt": "attempt_01"
    }
  ]
}
```

First inspect sizes, without reading log contents or writing files:

```bash
python scripts/export_logs.py --spec private_export_spec.json
```

An actual export requires the explicit `--export` flag and a new output directory:

```bash
python scripts/export_logs.py --spec private_export_spec.json \
  --export --out build/reviewer_log_views --redactions private_redactions.json
```

The output parent (`build` in this example) must already exist. Default ceilings:
8 MiB per input, 64 MiB of inputs, 100 MiB of output, with 10 GiB of free space
reserved. Larger files cause refusal, not automatic truncation. There is no
recursive collection, compressed-archive expansion, checkpoint loading or cleanup
of old files. UTF-8 decoding damage and a changing source file also cause refusal.

A private redaction map replaces explicitly approved identifying strings with
placeholders such as `<REDACTED_PATH>`. The map itself and source paths are not
exported. Review it before use; it must target identity/access metadata, not
scientific string-valued fields. Valid JSON records must retain their keys,
structure and all non-string values. Numeric changes are refused. Automatic
privacy checks are conservative heuristics, not a guarantee of anonymity.

The bundle contains derived text views and `INVENTORY.json`: original and exported
SHA-256 hashes, byte counts, declared capture scope, malformed-line indices,
duplicate/step-regression flags, and counts of each redaction/ANSI removal.
Original file bytes are not rewritten. All failures, warnings and restarts remain
visible unless identifying substrings were explicitly redacted. ANSI removal is
optional and recorded; it is not scientific-data cleaning.

Do not upload private redaction maps, access-helper scripts, machine/account
inventories, credentials, unreviewed raw logs or a private provenance handoff as
the anonymous supplementary ZIP. Keep required third-party attribution intact.


## v6 saved-record scope

V4 retains the saved original study contracts, cohort seals, gate records and small
execution metadata where they were recovered. These are not a complete raw
stdout/stderr export of every run. The new package records integrity/analysis
outcomes separately from original historical evidence. No original failure,
interruption or malformed trailing line was removed. The tool's export limits
remain unchanged; source fragments are never promoted to complete logs.
