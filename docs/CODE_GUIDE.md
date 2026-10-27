# Reading the code

This repository has two layers: convenient entry points at the root, and the
hash-verified research release in `reproducibility/`. The latter keeps original
file layouts because imports, source manifests, and provenance depend on them.

## Suggested reading order

1. Run [`examples/depth_update_demo.py`](../examples/depth_update_demo.py) with
   `make demo`. It transforms synthetic updates across six depths and checks
   that the mean update and centered norm are preserved.
2. Read [`tempering_reference.py`](../reproducibility/studies/cms1/fair_depth_lr_bracketing_v2/tempering_reference.py).
   `TemperConfig` defines the schedule and condition bound;
   `OnlineDepthTemper.transform` observes raw proposals, estimates their
   depthwise covariance, and applies the bounded transform.
3. Follow [`cms_policy.py`](../reproducibility/studies/cms1/correspondence_matched_s1_v1/cms_policy.py)
   for optimizer integration and correspondence controls, then
   [`cms_train.py`](../reproducibility/studies/cms1/correspondence_matched_s1_v1/cms_train.py)
   for the training driver.
4. Inspect [`repro/geometry.py`](../reproducibility/repro/geometry.py) for the
   exported numerical components. This module loads frozen implementations;
   it is not a second independent implementation of BDT.
5. Read the [GCT2](../reproducibility/studies/gct2/README.md) and
   [BAC1](../reproducibility/studies/bac1/README.md) study guides for pretrained
   growth and ancestry experiments. `make plans` prints their job plans along
   with CMS1 without starting training.

## From saved evidence to results

[`scripts/reproduce_all.py`](../reproducibility/scripts/reproduce_all.py) is the
unified offline analysis entry point. Its helpers live in
[`repro/`](../reproducibility/repro/), with compact inputs in
[`data/`](../reproducibility/data/) and archived records in
[`reference/`](../reproducibility/reference/).

Run `make analysis` once to write outputs to
`reproducibility/build/all_records/`. The scripts refuse to overwrite existing
outputs. For another run, choose a fresh destination:

```bash
cd reproducibility
../.venv/bin/python scripts/reproduce_all.py --out build/another_analysis
```

Read [coverage](../reproducibility/docs/COVERAGE.md) alongside the results:
recomputing saved scores is distinct from retraining the models.

## Verification layers

| Command | What it checks |
| --- | --- |
| `make verify` | Release hashes, source syntax, provenance bindings, and BCT2 receipts |
| `make test` | CPU numerical behavior, record validation, data handling, and portable entry points |
| `make demo` | Mean and centered-norm preservation on a small synthetic example |
| `make check` | All three checks above |

The original native-study CPU suites are separate from `make test`. Their
commands and the full training requirements are documented in the
[release README](../reproducibility/README.md). Historical results under
`validation/` describe earlier runs, not the current GitHub Actions run.
