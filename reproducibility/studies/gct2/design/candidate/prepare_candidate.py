"""Build one isolated diagnostics-only candidate from an exact archived source.

This never edits the reference, a study manifest, or a running optimizer.
The candidate is NOT an approved replacement for GCT2's frozen treatment.
"""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json

REFERENCE_SHA = 'd1bd72d9ccd18019aea83551b838d5915f727a0a7b6a0a2fd1006fb1c47969f1'
OLD_SIGNATURE = 'def transform(self, update: torch.Tensor, step: int, *, active: bool = True\n'
NEW_SIGNATURE = 'def transform(self, update: torch.Tensor, step: int, *, active: bool = True, diagnostics: bool = True\n'
OLD_REPORT = '''        cos = (actual_c * uc).sum() / (acn * raw_norm)
        report.update(operator_condition=float(gains.max()/gains.min()),
                      raw_update_cosine=float(cos), centered_norm_ratio=float(acn/raw_norm),
                      common_max_abs_error=float((actual.mean(0)-mean[0]).abs().max()))
'''
NEW_REPORT = '''        # Both fields used by the production safety guard remain computed.
        report.update(operator_condition=float(gains.max()/gains.min()),
                      centered_norm_ratio=float(acn/raw_norm))
        if diagnostics:
            cos = (actual_c * uc).sum() / (acn * raw_norm)
            report.update(raw_update_cosine=float(cos),
                          common_max_abs_error=float((actual.mean(0)-mean[0]).abs().max()))
'''

def transform_source(raw: bytes) -> bytes:
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA:
        raise ValueError('Reference hash mismatch; refusing an unverified transformation')
    source = raw.decode('utf-8')
    if source.count(OLD_SIGNATURE) != 1 or source.count(OLD_REPORT) != 1:
        raise ValueError('Expected source anchors absent or ambiguous')
    candidate = source.replace(OLD_SIGNATURE, NEW_SIGNATURE, 1).replace(OLD_REPORT, NEW_REPORT, 1)
    candidate += '\n# BCO1 diagnostics_lite_v1. Frozen reference remains unchanged.\n'
    candidate += '# Call transform(..., diagnostics=False) only in the new candidate adapter.\n'
    return candidate.encode('utf-8')

def build(reference: Path, output: Path) -> dict:
    reference, output = reference.resolve(), output.resolve()
    if reference == output:
        raise ValueError('Never overwrite the reference')
    raw = reference.read_bytes(); candidate = transform_source(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Idempotent only for byte-identical candidate; never overwrite another version.
    if output.exists():
        if output.read_bytes() != candidate:
            raise ValueError('Output already exists with different bytes')
    else:
        with output.open('xb') as f: f.write(candidate)
    return dict(candidate='diagnostics_lite_v1', reference_sha256=REFERENCE_SHA,
                candidate_sha256=hashlib.sha256(candidate).hexdigest(),
                reference_unchanged=hashlib.sha256(reference.read_bytes()).hexdigest()==REFERENCE_SHA,
                removed_only=['optional raw_update_cosine','optional common_max_abs_error'],
                preserved=['update arithmetic','covariance/eigenstate','input/finite/order checks','condition guard inputs','centered norm guard inputs'],
                status='PROTOTYPE_REQUIRES_FULL_MODEL_CUDA_EQUIVALENCE_AND_TIMING')

if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();print(json.dumps(build(args.reference,args.output),indent=2))
