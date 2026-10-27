# Cross-Layer Transformer Optimization

**[Quick start](#verification) · [Code guide](docs/CODE_GUIDE.md) · [Reproducibility coverage](reproducibility/docs/COVERAGE.md) · [Contributing](CONTRIBUTING.md)**

Research code and reproducibility records for **bounded depth tempering (BDT)**, a method for modifying optimizer updates across corresponding Transformer layers. The research question is whether cross-layer parameter correspondence contains useful information for optimization, beyond the magnitude of an update change.

## Method

For each corresponding feed-forward projection across layers, BDT collects the proposed AdamW updates into a depth-by-parameter matrix. It separates the update shared across depths from deviations between layers, then uses an exponential moving estimate of depthwise update statistics to attenuate high-energy directions. The transformed proposal preserves the mean update and the total norm of the centered updates. The network's forward architecture is unchanged.

The experiments distinguish **which parameters are matched** across layers, **how their updates are transformed**, and **how update magnitude is distributed**. Correspondence-scrambled controls hold selected properties of the intervention constant; pretrained-layer copying provides a known ancestry relationship for further comparisons.

## Start here

| Task | Source |
| --- | --- |
| BDT transformation used by the training implementation | [`reproducibility/studies/cms1/fair_depth_lr_bracketing_v2/tempering_reference.py`](reproducibility/studies/cms1/fair_depth_lr_bracketing_v2/tempering_reference.py) |
| CMS1 optimizer and correspondence-matched controls | [`reproducibility/studies/cms1/correspondence_matched_s1_v1/cms_policy.py`](reproducibility/studies/cms1/correspondence_matched_s1_v1/cms_policy.py) |
| CMS1 training driver | [`reproducibility/studies/cms1/correspondence_matched_s1_v1/cms_train.py`](reproducibility/studies/cms1/correspondence_matched_s1_v1/cms_train.py) |
| Pretrained growth experiments (GCT2) | [`reproducibility/studies/gct2/`](reproducibility/studies/gct2/) |
| Ancestry correspondence experiments (BAC1) | [`reproducibility/studies/bac1/`](reproducibility/studies/bac1/) |
| Offline evidence and further BCT1/BCT2 analyses | [`reproducibility/repro/`](reproducibility/repro/), [`reproducibility/reference/`](reproducibility/reference/) |
| Claim-by-claim reproducibility coverage | [`reproducibility/docs/COVERAGE.md`](reproducibility/docs/COVERAGE.md) |

The `reproducibility/` directory retains the research release, including its pinned scientific source, archived records, and integrity checks. Historical provenance files are deliberately retained rather than silently rewritten or pruned.

## Verification

Use Python 3.13 for the pinned CPU environment on macOS or Linux. The original
release was validated with Python 3.13.5 on Linux; native studies require POSIX
file locking.

The archive checks use the Python standard library. From the repository root:

```bash
make verify
```

To run the CPU-only test suite, install the dependencies and run:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
make install
make test
```

For an **illustrative**, small-tensor invocation of the archived BDT reference operator (not a language-model training run):

```bash
python examples/depth_update_demo.py
```

The offline analysis and native-study instructions are in [`reproducibility/README.md`](reproducibility/README.md) and [`reproducibility/docs/NATIVE_STUDIES.md`](reproducibility/docs/NATIVE_STUDIES.md). `make help` lists convenience commands. No default command downloads data, connects to a compute cluster, or launches GPU training.

## Repository layout

```text
examples/                 Small, runnable introduction to BDT
docs/                     Reading guide for the implementation
reproducibility/
├── studies/              Original CMS1, GCT2, and BAC1 implementations
├── native/               Frozen numerical kernels and dependencies
├── repro/                Analysis and portable training helpers
├── scripts/              Verification, analysis, and training entry points
├── tests/                CPU and record-integrity tests
├── data/                 Compact scores and diagnostics
├── reference/            Archived evidence and source provenance
├── validation/           Historical verification records
└── docs/                 Protocols, coverage, and reproduction instructions
```

Start with the [code guide](docs/CODE_GUIDE.md) for a short reading order through
the numerical implementation, optimizer integration, and experiment drivers.

## Evidence and limitations

This repository contains original training implementations, selected source-matched numerical components, saved results, and verification scripts. It is **not** a complete standalone reproduction of all experiments: the original language-model dataset cache, checkpoints, and some historical raw logs are not included. Offline tests and result recomputation validate the packaged software and records; they do not constitute a fresh full-scale training replication.

Training-loss improvements were observed in the studied settings. They are not claims of a generally superior optimizer or faster end-to-end training. Exact experimental coverage and limitations are recorded in [`reproducibility/docs/COVERAGE.md`](reproducibility/docs/COVERAGE.md).

## Source provenance and reuse

The release preserves original source hashes and redaction records. Consult [`reproducibility/THIRD_PARTY_NOTICES.md`](reproducibility/THIRD_PARTY_NOTICES.md) before redistribution. No blanket open-source license is asserted, and preserved historical records should not be edited merely for presentation.
