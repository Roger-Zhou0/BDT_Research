# BCT2: fresh-parent confirmation records

BCT2 is the four-fresh-parent confirmation in manuscript Table 3 and Appendix AA.
It is analyzed separately from BCT1's four reused development parents. The
architecture, corpus realization, copying procedure and hybrid recipe are
unchanged; fresh initializations and pretraining streams define the new cohort.

## Inspect or recompute

From the extracted package root, with Python's standard library only:

```bash
python scripts/verify_bct2.py
python scripts/verify_bct2.py --out build/bct2_records
```

The second command requires a new output directory. It writes all eight endpoint
scores and four differences as CSV, a result summary, and the record-check report.
`python scripts/reproduce_all.py --out build/all_records` includes this same BCT2
check alongside the other studies; it does not pool the cohorts.

The paired full-minus-structured result is -0.0005939332768321037 nats/token,
with two-sided 95% Student-t interval [-0.0006332087165214267,
-0.0005546578371427807] and four negative signs. The standard error is
0.00001234127332301207. The verifier recomputes these from the original scores,
using four parents and three degrees of freedom. It does not infer precision
from the rounded manuscript table.

## What was recovered

Seventeen small files were read directly from the completed experiment's
authenticated notebook Contents API on September 26, 2026. The UTF-8 bytes were
hashed before redaction. No experiment file or training process was modified.
No model was loaded, and no training or REPORT evaluation was rerun.

Twelve files are included byte-for-byte:

* `run/REPORT/FINAL_REPORT.json` (2,506 bytes);
* all eight `run/REPORT/<parent>_<arm>.json` score receipts;
* `run/SUCCESS.json`;
* `SOURCE_MANIFEST.json` and `SOURCE_DIGEST.txt`.

The original final report's SHA-256, also stated in manuscript Appendix AA, is:

```text
529b846d764a05147859ab47cd210d07075934246d3ddd9bf15cca2e341a3fd9
```

Five files are explicitly labeled anonymous derivatives in `reference/bct2/ORIGIN.json`:

* `protocol.json`: one execution-device identifier is replaced;
* `run/GLOBAL_ENDPOINT_SEAL.json`: private absolute workspace/cache paths are replaced;
* both lane `LOCAL_COMPLETE.json` files: private workspace paths in ledger task labels are replaced;
* `run/lanes/good/status.json`: the execution hostname is replaced.

The generic labels `default` and `good` are the recorded lane names, not author
identities. All scalar scientific values, losses, seed identifiers, timestamps,
recipe settings, evaluation offsets, ledger counts, model/checkpoint digests and
original binding strings are preserved. Changed JSON string fields are listed
explicitly in ORIGIN.json. Synthetic path labels do not claim those large
external files are contained in the ZIP.

## How to interpret the hash chains

Every ORIGIN row distinguishes `original_sha256` from `public_sha256`. The public
verifier checks all shipped bytes. The twelve unchanged files additionally match
their original byte hashes. The final report hashes the eight unchanged score
receipts; each receipt binds a seed, arm, loss, model/checkpoint digest, runtime,
evaluation slice and original seal digest. The success record binds that final
report and original seal.

The seal, protocol and affected nested contracts have different public hashes
because their private strings were redacted. Original binding strings are kept,
not rewritten to suggest the anonymous copies were used in the historical run.
ORIGIN records each original/public nested-contract canonical digest. The release
process checked the original contracts against the original checkpoint metadata
before replacing paths. Offline verification checks the public contract hashes
and the recorded original bindings; it cannot reconstruct erased original values.

The protocol has two historically different hashes even before anonymization:
its raw-byte digest is the source manifest's entry, whereas the report's protocol
binding uses sorted, two-space-indented JSON plus a final newline. Both original
and public canonical digests are recorded. They are not interchangeable with
raw-byte digests.

The included source manifest binds thirteen original runner/protocol files.
Those original coordinator files are **not** included or newly source-audited in
this release. The manifest is provenance evidence, not a claim of complete BCT2
training-source coverage. The existing `native/` and portable BCT1 routes have
their own documented scope and do not recreate the exact BCT2 coordinator.

## Verification boundary

The new `repro/bct2_records.py` and `scripts/verify_bct2.py` are release utilities,
not code claimed to have generated the original training outcomes. They check
record integrity and cross-bindings, the fixed cohort, unique parent identities,
endpoint metadata, both completion ledgers and the paired arithmetic. The
recorded seal predates the recorded reporting files. This is timestamp/binding
evidence, not independent proof of every historical execution event.

Checkpoint bytes, full token arrays and original GPU execution are external and
were not reloaded or reproduced. Hash consistency supports the integrity of the
included evidence; it is not itself an end-to-end training reproduction.
