# How Parameter Correspondence Shapes Cross-Layer Optimization
## Supplementary code and records — v9, accompanying manuscript v39

Released September 26, 2026. This documentation-only update aligns the release
labels and copying terminology with manuscript v39. BCT2 fresh-parent evidence
introduced in v8 is retained; the historical v20 evidence export keeps its
original directory name. Scientific source and numerical records are unchanged.

This release contains original CMS1, GCT2 and BAC1 training implementations and
frozen dependencies, the BCT1 numerical implementation and portable reproduction
route, the historical numerical records supplied with manuscript v20, and the
completed BCT2 fresh-parent report and hash-bound evidence used in manuscript
v39. The fast route is offline: verify the archive, recompute results, and run
CPU engineering checks.
No verification command downloads a corpus, a checkpoint, or a model.

The release distinguishes original records, hash-linked score extractions and
manuscript-displayed precision. It does not claim that every historical run log
is redistributed, that every appendix experiment has a runnable driver, or that
full-sized GPU experiments were rerun during packaging. The exact external S1-B
cache is not included; its historical upstream dataset/tokenizer commits remain
unresolved. See `docs/COVERAGE.md` for the claim-by-claim boundary.

## Quick start: inspection and offline evidence

Run these commands from the extracted directory. These initial commands require only
Python's standard library and do not import PyTorch. Python 3.13.5 on Linux is
the tested packaging environment; native studies use POSIX file locking.

```bash
python scripts/check_package.py
python scripts/native_study.py --study cms1 --action plan
python scripts/verify_bct2.py
```

The complete record recomputation additionally needs SciPy. To run all tests,
install the explicitly separate dependencies in a fresh environment **on a
machine with sufficient storage**, not necessarily the laptop used to view this
archive. Package installation itself may consume substantial space. It is not
required simply to read the source or the saved validation records.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python scripts/reproduce_all.py --out build/all_records
python scripts/verify_bct2.py --out build/bct2_records
python -m unittest discover -s tests -v
python scripts/smoke_training.py --out build/bct1_cpu
python scripts/native_study.py --study cms1 --action cpu-test --out build/cms1_cpu
python scripts/native_study.py --study gct2 --action cpu-test --out build/gct2_cpu
python scripts/native_study.py --study bac1 --action cpu-test --out build/bac1_cpu
python scripts/plot_results.py --out build/plots
```

Choose a new output directory on each invocation. Analysis and test outputs are
not original scientific results. All CPU tests use synthetic tokens or recorded
numbers, not a fresh corpus sample. Timing these tests does not measure scientific
GPU throughput or an optimizer speed advantage.

## What is included

| Directory | Contents |
|---|---|
| `studies/cms1/` | Complete original CMS1 source and frozen shared/private S1 backend. |
| `studies/gct2/` | Complete original GCT2 release, growth backend, tests and protocols. Its inherited BCO1 profiling code is retained for source identity, not run during packaging. |
| `studies/bac1/` | Complete original stacked-copy BAC1 release and its frozen dependencies. |
| `native/` | Nineteen byte-identical BCT1/model/policy kernels and their subset manifest. |
| `repro/` | New packaging helpers, four exact trainer excerpts, portable BCT1 loop, and offline analyses. |
| `reference/cms1/` | All 18 byte-preserved original CMS1 REPORT records and their provenance. |
| `reference/native_cms1/` | Original source archive's contracts, gate/cohort records and saved metadata, with identity-only redactions where needed. |
| `reference/recovered_records/` | GCT2/FEV1 audited unrounded score projections and original GCT2 seal; BAC1 unrounded completed final analysis, exact 24-endpoint cohort lock, and executed-source registration. |
| `reference/manuscript_v20/` | All 42 table fragments and numerical evidence from the historical v20 export, conceptual figure source and inspectable historical figure scripts. |
| `reference/bct2/` | Completed original BCT2 report and receipt/protocol/seal bindings, with explicit original/public hashes for privacy derivatives. |
| `data/` | Compact final scores, displayed historical tables and 240 retained allocation observations. No text corpus or weights. |
| `scripts/` | Safe inspection, analysis, plotting, native-study dispatch, portable BCT1 training and bounded log export. |
| `validation/` | Retained v8 check summaries and earlier `previous_v7/` records; `CURRENT_CHECKS.md` distinguishes v9 verification from those historical checks. |

`SUPPLEMENT_RECORDS_INDEX.json` and `docs/COVERAGE.md` explain the evidence levels.
Presence of a table fragment is not presented as recovery of its original raw
training log. The BCT1 carrier is a fixed GS-derived **direction construction**
combined with own-state sector budgets and attention BDT, not standalone GS.

## Copying layouts

GCT2 and BAC1 use **stacked copying**. With zero-based target layer `r`, the
pretrained source block is `r % 6`:

```text
GCT2/BAC1: 0,1,2,3,4,5,0,1,2,3,4,5
```

The later adjacent-copy allocation studies instead place the two descendants of
each block next to one another, with source block `r // 2`:

```text
Adjacent:  0,0,1,1,2,2,3,3,4,4,5,5
```

These are different layouts. The code, mappings and historical records are
unchanged; see `docs/NATIVE_STUDIES.md` for implementation locations.

## Native training and full-sized reproduction

`docs/NATIVE_STUDIES.md` documents CMS1/GCT2/BAC1 entrypoints and the new serial
adapter. The original numerical code and tests are preserved. Account/path
redactions are recorded per file in `studies/ORIGIN.json`; the public source
manifests intentionally have different digests from original private run
contracts. Do not use a public manifest as authentication of historical original
bytes. Raw original hashes and anonymous derivative hashes remain distinct.

GPU actions require explicit flags, a NEW workspace, an exact existing external
cache, a passing full-sized engineering gate and complete cohort barriers before
REPORT. The templates have **not** been executed on CUDA during packaging. The
native CPU suites validate small-model numerical/restart paths, not full-sized
scientific efficacy. Original sources use `torch.load(..., weights_only=False)`
in places: load only trusted, locally generated checkpoints. Do not load
untrusted pickle-based checkpoints simply because a hash was provided.

`docs/FULL_SIZE_REPRODUCTION.md` separately explains the portable BCT1 route.
It regenerates parent training rather than loading the original reused parents;
it must not be pooled with the paper's four-parent historical cohort.

## Important record distinctions

CMS1 has 18 original REPORT files and six training-seed units. GCT2/FEV1 has six
trained parents, 18 endpoint models and two evaluation partitions on those same
models; the package contains audited unrounded score projections, not all 36
original report files. BAC1 now includes the completed unrounded six-row final analysis
and the exact original 24-endpoint cohort lock with endpoint-result hashes; every
unrounded row is cross-checked against the manuscript's nine-decimal display. The
24 individual raw per-arm REPORT files are still not redistributed. AGT1/FRT1/ART1/BCT1 provide 56 logical reporting cells over
40 unique checkpoint cells on four reused parents. The 240 allocation snapshots
are repeated observations, not 240 independent replications. BCT2 contributes four
fresh pretrained parents (946301–946304); its two-arm comparison is not pooled
with the four reused BCT1 families. See `docs/BCT2_RECORDS.md`.

## Storage, logs, rights and scope

Data/checkpoint storage remains external. No installation/import/test connects
to a cluster, notebook, scheduler or submission portal. The original preparation
script is inspectable text, not a downloader launched by the quick start. Its
source and exact-cache checks are described in `docs/DATA_AND_ENVIRONMENT.md`.

The log exporter defaults to metadata-only preflight; export is opt-in and bounded.
It preserves malformed tails, separate attempts, duplicates and interruption
markers, with documented derivative redactions. It never repairs missing results
or deletes originals. See `docs/LOG_HANDLING.md`.

No new repository-wide license or third-party rights grant is asserted by this
packaging pass. Framework binaries and fonts are not bundled. See
`THIRD_PARTY_NOTICES.md` and the recorded notice inventory. GSC1 remains outside
this release. The FEV1 builder/evaluator source is not shipped; its saved scores
and offline arithmetic are included. BCT2 coverage is described separately.


## Original appendix display replay

`python scripts/reproduce_appendix.py --out build/appendix` stages anonymous
copies in a new directory and runs four original manuscript display/check
scripts. It verifies fourteen regenerated table fragments byte-for-byte, checks
OSC1 receipts and OIP1 descriptive extraction, and separately reconstructs all
240 allocation observations and 48 displayed mean cells using a public-record
adapter. Original versus anonymous payload hashes remain distinct. The wrapper
sets only tight figure bounds to prevent clipping of long historical axis labels;
this is not a numerical or manuscript-source change. It does not regenerate all
42 tables or read original training logs. Failure outputs remain in the chosen
attempt directory rather than being silently overwritten.

## v7 provenance-label clarification

V7 corrects four `studies/ORIGIN.json` transformation labels for derivative source-manifest files. The public manifest bytes already differed intentionally because anonymized paths/hashes were rebound; v6 described those rows with an ambiguous `byte-identical; derivative ...` prefix. V7 removes that false byte-identity implication. No scientific source, result, numeric evidence, timestamp, or original/public hash value changed.

## v8 release corrections (retained history)

This release adds verified completed BCT2 records and a standalone arithmetic and
record-binding checker, removes author-specific identifiers from the public
privacy checker and its tests, and corrects FEV1 source-coverage wording. No
pre-existing scientific source or numerical record is changed. Public privacy
checks use generic patterns and synthetic test identities. The author-specific
release scan is deliberately outside this archive. See `docs/RELEASE_STATUS.md`.

## v9 documentation corrections

V9 updates the submitted title and manuscript/release labels, replaces ambiguous
copying terminology with the stacked and adjacent mappings above, and refreshes
the top-level manifest. Historical source/evidence labels and numerical payloads
are preserved. See `docs/RELEASE_STATUS.md` and `validation/CURRENT_CHECKS.md` for
the current verification scope and retained earlier checks.
