# Reviewer workflow

Start with `validation/RELEASE_CHECKS.json` and `SUPPLEMENT_RECORDS_INDEX.json`.
Then run `python scripts/check_package.py` to verify the delivered bytes. Read
`docs/COVERAGE.md` before interpreting a missing raw file as an available rerun.

For the main findings, `python scripts/reproduce_all.py --out build/records`
produces human-readable contrasts and complete machine-readable output without
loading weights, opening data caches or connecting to a service. Look at the
paired units and evidence-precision labels next to each result. The code reports
both the successful BCT1 ordering and its remaining gap to combined BDT.
BCT2 is a separate four-fresh-parent confirmation. Its original report and eight
score receipts can also be checked without SciPy or PyTorch using
`python scripts/verify_bct2.py`; see `docs/BCT2_RECORDS.md` for redaction lineage.

For implementation inspection, start with `repro/geometry.py`, then the original
policy source under each `studies/` tree. The new serial adapter only supplies
workspace/dispatch functions; it is not the origin of the numerical method.
The original model backends are included so S1 and ordinary Dense construction
are not accidentally substituted for one another.

For a low-cost implementation exercise, use the unit tests, the separate BCT1
synthetic test, and each study's original CPU suite. These outputs are labelled
engineering, with reduced models and synthetic data. They are not smaller seed
replications of the paper's scientific effect.

For a substantial reproduction, use external compute and the exact external
cache. Run a full-sized gate before scientific jobs, and let complete-cohort
checks precede REPORT evaluation. Never treat changed data, a failed gate, or a
partially completed cohort as an equivalent result. The templates' CUDA behavior
has not been certified by this packaging pass.

For logs, retain attempts and malformed tails; do not make a resumed run appear
uninterrupted. The safe exporter refuses oversized inputs rather than quietly
shortening them. No reviewer needs private account identifiers, access helpers,
credentials or a copy of the researchers' entire working directory to inspect
the included method and record analyses.
