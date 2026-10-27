# Verification scope for v9 and retained v8 checks

V9 changes explanatory documentation and the top-level release/manuscript labels
only. Scientific code, test implementations and numerical records are unchanged.
The top-level manifest is refreshed for the edited files. Current release checks
are recorded in the separate v9 delivery report; they must not be confused with
the larger v8 validation campaign described below. Inherited validation JSON,
result summaries and `previous_v7/` files retain their original bytes and labels.

V9 reran the package checker, the full 135-test packaging unit suite, unified
offline record recomputation, and the 119-check BCT2 verifier. All passed.
The unchanged source-name mapping functions were also checked for both copying
layouts without constructing a model or running a training loop. These checks
do not re-execute the original native-study CPU campaigns listed below.

## Retained v8 verification summary

The v8 checks ran on the environment in `ENVIRONMENT.json`. All 15 bounded
command invocations completed successfully. The packaging unit suite passed 135
tests, including integrity failure cases for the new BCT2 records. Original
CMS1, GCT2 and BAC1 tiny-CPU suites and the separate portable BCT1 restart
exercise passed. These are synthetic implementation checks, not scientific
efficacy replications.

BCT2's 119 checks validate the shipped records, original/derivative hash
distinctions, score bindings, cohort and arithmetic. Unified recomputation covers
the packaged study records. An additional arithmetic script, not importing the
package's analysis modules, checked 15 principal or qualification-bearing
comparisons. It also checked the retained allocation endpoints. The original
appendix replay regenerated fourteen supplied table fragments byte-for-byte.

Earlier validation is preserved under `previous_v7/` with its original bytes.
Its labels and counts describe that earlier release, not this one. The current
checks do not load full checkpoint payloads or verify external corpus bytes, and
do not revalidate full-sized GPU templates. Final archive-level checks are
described in the separate delivery report rather than embedding a self-referential
ZIP hash inside this package.
