# Data and environment: recovered source, explicit remaining gaps

## Data-preparation source recovered in candidate v2

`reference/data/prepare_s1_data.py.txt` is the byte-preserved Python source from a
saved file-response archive. Its SHA-256 is
`ae0538cd75e94a48640eeefcb3ba420a64ac484fb95120ad011a69b8f604097c`,
which agrees with a separately retained later preparation-source pin. The `.txt`
suffix is intentional: inspection and offline tests do not launch its downloader.
The original script is not modified to make its outputs look reproducible.

The recovered implementation uses the following procedure:

- Stream the training split of `HuggingFaceFW/fineweb-edu`, configuration
  `sample-10BT`, with a dataset `revision` argument whose default is `main`.
- Shuffle the iterable stream with seed 20260828 and a 10,000-document buffer;
  skip empty text values and tokenize text batches of 64 by default.
- Use the fast GPT-2 tokenizer without special-token insertion or truncation;
  append one EOS token to each document's token sequence.
- Fill validation before training. At the validation token boundary, discard the
  remainder of that document rather than putting its tail into training.
- Fill the remaining training quota, truncate its final document at the token
  quota, and write native NumPy `uint16` memmaps and a metadata JSON.

These statements describe the recovered source. They are not an independent
certification of document identities in the already produced scientific cache.
The offline tests use simulated dataset/tokenizer APIs and tiny temporary arrays
to exercise packing, EOS handling, empty records, the boundary-document tail and
incomplete-stream behavior. They do not exercise the real Hugging Face shuffle,
real GPT-2 tokenization or reproduce six gigabytes of study data.

## Exact S1-B contract and verification

The study contract is 3,000,000,000 training IDs and 5,000,000 validation IDs:
6,000,000,000 and 10,000,000 bytes, respectively. The recorded metadata is 346
bytes. Hashes are in `configs/bct1_historical_protocol.json` and independently
recovered CMS1 contracts in `reference/ARCHIVED_DATA_AND_RUNTIME.json`.

The preparation source's default training quota is only 100,000,000 IDs. The
later reconstruction command explicitly requested the 3,000,000,000-ID study
quota. Do not confuse the script default with the study recipe.

The new inspector never downloads or copies data:

```bash
# No token payload read: check sizes of an existing cache only.
python scripts/check_data_environment.py --cache /existing/cache

# Explicitly read the existing files in place for SHA-256 verification.
# This reads about 6.01 GB but creates no second token-cache copy.
python scripts/check_data_environment.py --cache /existing/cache --hash-cache
```

A size-only check is never labelled exact-cache verification. A mismatched hash
is a refusal, not permission to substitute a new FineWeb-Edu realization. The
original preparation script's metadata-based “already complete” shortcut is not
an integrity test; the study's checksum verifier remains necessary.

## Still unresolved: immutable upstream identities

The recovered preparation source accepts a mutable dataset revision (`main`) and
does not pass a tokenizer revision to `from_pretrained`. The package has not
recovered the originally resolved dataset commit, tokenizer snapshot, complete
original preparation-environment lock, or an independently validated byte-exact
rebuild from those identities. Later reconstruction-environment choices must not
be relabelled the original cache-creation environment.

Therefore source recovery closes a missing-code gap, but does not certify that
running the source against today's upstream data would regenerate S1-B. No
upstream revisions, download URLs for a private cache, missing document IDs or
metadata values have been invented. No real-corpus data-generation command is run during
package tests. Full paper reproduction still requires the exact external cache
or a subsequently verified reconstruction/distribution route.

## Historical GPU runtime recovered for CMS1

All 18 archived CMS1 continuation contracts record one identical runtime variant:
Python 3.13.14, PyTorch 2.11.0+cu128, NumPy 2.4.6, CUDA runtime 12.8, cuDNN 91900,
and NVIDIA H200 NVL with capability 9.0. They record four CPU threads,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, strict deterministic algorithms, disabled
matmul TF32 and cuDNN benchmarking, and enabled deterministic cuDNN. SDP backend
availability flags are preserved in the JSON, not converted into a claim that
one specific kernel handled every operation.

The exact fields and 18 contract-byte hashes are in
`reference/ARCHIVED_DATA_AND_RUNTIME.json`. These are historical observations,
not a new live GPU read. They are not a full pip/conda/container lock, a driver
inventory, an installation-availability check, or evidence that every later study
used the same runtime. In particular, do not apply them automatically to BCT1.

```bash
python scripts/check_data_environment.py
```

This compares installed package metadata with the recorded CMS1 versions without
initializing CUDA, installing packages, changing environment flags or training.
A CPU environment mismatch is reported rather than repaired automatically.

`requirements-cpu.txt` and `validation/ENVIRONMENT.json` describe the environment
used for this package's CPU tests, not the historical scientific training stack.


## Additional runtime evidence recovered in v3

`reference/GCT2_ARCHIVED_RUNTIME_EXAMPLE.json` preserves the runtime from one
actual GCT2 REPORT file. This is one recorded evaluation, not a newly inferred
all-cohort environment lock. Original BAC1 source registration was recovered,
but a complete original BAC1 runtime/dependency lock was not established. The
package does not fill that gap by copying CMS1 versions into BAC1 or BCT1.

The CPU environment is separately recorded in `validation/ENVIRONMENT.json`.
Hardware/backend-specific reproducibility and full dependency resolution require
actual compute-environment checks; source identity and CPU parity alone cannot
supply them.
