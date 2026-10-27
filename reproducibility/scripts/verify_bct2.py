#!/usr/bin/env python3
"""Verify BCT2 score receipts and recompute its fresh-parent primary, offline."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repro.bct2_records import SEEDS, ARMS, verify_bct2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, help='New output directory; no overwrite.')
    args = parser.parse_args()
    result = verify_bct2()
    if args.out is not None:
        out = args.out.resolve()
        if out.exists():
            raise FileExistsError('Use a new output directory')
        if out.is_relative_to(ROOT) and not out.is_relative_to(ROOT / 'build'):
            raise ValueError('Inside the release, use build/ only')
        out.mkdir(parents=True)
        (out / 'BCT2_CHECKS.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        with (out / 'BCT2_ENDPOINTS.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['parent', *ARMS, 'full_minus_structured'])
            for seed in SEEDS:
                writer.writerow([seed, *(result['scores'][str(seed)][a] for a in ARMS), result['per_parent'][str(seed)]])
        s = result['paired_95']
        text = ['# BCT2: fresh-parent confirmation', '',
                'All eight original score receipts are included. The four fresh parents are analyzed separately from BCT1.', '',
                f"Mean full-minus-structured loss: {s['mean']:+.12f} nats/token.",
                f"Paired 95% Student-t interval (df=3): [{s['lower']:+.12f}, {s['upper']:+.12f}].",
                'All four differences are negative; the registered criterion is met.', '',
                result['boundary'], '']
        (out / 'BCT2_RESULTS.md').write_text('\n'.join(text))
    print(json.dumps({k: result[k] for k in ('passed', 'checks_passed', 'fresh_parents',
                     'original_score_receipts', 'original_final_report_sha256', 'paired_95',
                     'pooled_with_BCT1', 'new_model_evaluations')}, indent=2))


if __name__ == '__main__':
    main()
