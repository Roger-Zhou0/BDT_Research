# Working on this repository

Use Python 3.13 on macOS or Linux and create an environment at the repository
root:

```bash
python3.13 -m venv .venv
make install
make check
```

The Makefile selects `.venv/bin/python` when present. To use a different
interpreter, pass `PYTHON=/absolute/path/to/python` to `make`.

## Preserve the research release

The files in `reproducibility/` are bound by
`reproducibility/MANIFEST_SHA256.json` and additional source manifests. Cosmetic
reformatting, moving files, or editing archived results breaks those identities.
Place navigation, examples, and developer tooling outside that directory.

If a scientific implementation needs a correction, make it an explicit,
documented derivative with appropriate tests and provenance. Do not silently
refresh hashes to make an unintended change pass verification.

Keep generated outputs in ignored `build/` directories. Do not commit virtual
environments, corpus caches, checkpoints, or credentials. Analysis commands
expect a new output directory for each invocation.

## Before submitting a change

Run `make check` and describe the behavior changed and the checks performed.
For changes to a study, also run its native CPU suite as documented in the
[release README](reproducibility/README.md). CPU checks do not establish a
full training replication.

Consult [source notices](reproducibility/THIRD_PARTY_NOTICES.md) before
redistributing source. This repository does not assert a blanket open source
license for the preserved release.
