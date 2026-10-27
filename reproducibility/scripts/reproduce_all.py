#!/usr/bin/env python3
"""Verify and recompute the packaged record set. No model, network or cache I/O."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from repro.results import report_results,allocation_results
from repro.archived_cms1 import recompute
from repro.extended_records import gct2_fev1,bac1_records
from repro.bct2_records import verify_bct2

def manuscript_inventory():
    origin=json.loads((ROOT/'reference/manuscript_v20/ORIGIN.json').read_text())
    rows=origin['inputs']
    names=[r['member'] for r in rows]
    if len(names)!=len(set(names)):raise RuntimeError('Duplicate source members')
    for row in rows:
        p=ROOT/row['member']
        if p.is_symlink() or not p.resolve().is_relative_to(ROOT) or hashlib.sha256(p.read_bytes()).hexdigest()!=row['public_sha256']:
            raise RuntimeError('Changed manuscript input: '+row['member'])
    tables=[r['member'] for r in rows if '/tables/' in r['member']]
    if len(tables)!=42:raise RuntimeError('Incomplete v20 table-fragment export')
    return {'manuscript':'v20','table_fragments':tables,'source_members':len(rows),
            'boundary':'Hash checks of every shipped table fragment and numerical source. Not an assertion that every historical appendix experiment has a complete raw-record export or independent arithmetic reimplementation.'}

def recompute_all():
    return {'package':'supplementary code and records v8','new_model_evaluations':0,
            'CMS1':recompute(),'GCT2_FEV1':gct2_fev1(),'BAC1':bac1_records(),
            'AGT1_FRT1_ART1_BCT1':report_results(),'allocation':allocation_results(),
            'manuscript_inputs':manuscript_inventory(),'BCT2':verify_bct2()}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    out=a.out.resolve()
    if out.exists():raise FileExistsError('Use a new output directory')
    if out.is_relative_to(ROOT) and not out.is_relative_to(ROOT/'build'):raise ValueError('Inside release, use build/ only')
    result=recompute_all();out.mkdir(parents=True)
    (out/'ALL_RECORD_CHECKS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Supplementary record recomputation','',
           'Offline arithmetic; zero new model evaluations. Confidence intervals are paired across the stated training parents, not checkpoints or observations.','',
           '| Study | Comparison | Mean | 95% interval | Evidence |','|---|---|---:|---|---|']
    for label,section in [('CMS1',result['CMS1']),*[(f'GCT2 {part}',result['GCT2_FEV1']['results'][part]) for part in ('REPORT','FEV1')]]:
        for comparison,stats in section['contrasts'].items():
            r=stats['paired_95'];level='18 original REPORT files' if label=='CMS1' else 'Unrounded audited projection; same six trained parents'
            lines.append(f"| {label} | {comparison} | {r['mean']:+.12f} | [{r['lower']:+.12f}, {r['upper']:+.12f}] | {level} |")
    b=result['BAC1']['primary']['recomputed_paired_95']
    lines.append(f"| BAC1 | reassigned - preserved | {b['mean']:+.12f} | [{b['lower']:+.12f}, {b['upper']:+.12f}] | Unrounded final analysis + exact 24-endpoint cohort lock |")
    lines+=['','## Four reused adjacent-copy parents','',
            '| Study | Comparison | Mean | Interval level | Interval |','|---|---|---:|---:|---|']
    for study,contrasts in result['AGT1_FRT1_ART1_BCT1']['contrasts'].items():
        for comparison,r in contrasts.items():
            lines.append(f"| {study.upper()}1 | {comparison} | {r['mean']:+.12f} | {100*r['confidence']:g}% | [{r['lower']:+.12f}, {r['upper']:+.12f}] |")
    b2=result['BCT2']['paired_95']
    lines+=['','## Four fresh BCT2 parents (separate cohort)','',
            f"Full-BDT budgets minus structured budgets: {b2['mean']:+.12f} nats/token; paired 95% interval [{b2['lower']:+.12f}, {b2['upper']:+.12f}].",
            'All eight original score receipts and the exact final report are checked. Anonymous seal/protocol derivatives retain explicit original/public hash mappings. No pooling with BCT1.','']
    lines+=['','The JSON also preserves 97.5% CMS1/GCT2 intervals, all per-parent values, source precision, 240 repeated allocation observations, and hashes of the 42 table fragments. Display-level and raw-record evidence remain separate.','']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({'passed':True,'output':str(out),'manuscript_table_fragments':42,'CMS1_original_reports':18,'GCT2_trained_parents':6,'BAC1_unrounded_rows':6,'BAC1_endpoint_hashes':24,'BCT2_fresh_parents':4,'BCT2_original_reports':8,'new_model_evaluations':0},indent=2))
if __name__=='__main__':main()
