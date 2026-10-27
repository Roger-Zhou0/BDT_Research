"""Arithmetic from 18 byte-preserved, archived complete CMS1 REPORT records."""
from pathlib import Path
import hashlib,json
from repro.results import paired
ROOT=Path(__file__).resolve().parents[1]
SEEDS=tuple(range(147,153));ARMS=('adamw','bdt','scrambled_matched')


def recompute(root: Path | None = None) -> dict:
    root=ROOT/'reference/cms1' if root is None else Path(root)
    origin=json.loads((root/'ORIGIN.json').read_text());scores={s:{} for s in SEEDS}
    expected={f'seed_{s}/{a}_report.json' for s in SEEDS for a in ARMS}
    bindings=origin['bindings']
    if len(bindings)!=18 or {v['path'] for v in bindings}!=expected:
        raise RuntimeError('Incomplete or duplicated CMS1 REPORT cohort')
    for row in bindings:
        raw=(root/row['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise RuntimeError('Archived report bytes changed')
        r=json.loads(raw);seed=r['seed'];arm=r['method'];ev=r['report']
        if row['path']!=f'seed_{seed}/{arm}_report.json' or r['state']!='complete':
            raise RuntimeError('Report identity/state mismatch')
        if (ev['tokens'],ev['partition_start'],ev['partition_stop'])!=(1048576,2500000,3550624):
            raise RuntimeError('Report evaluation contract differs')
        scores[seed][arm]=ev['loss']
    out={}
    for other in ('adamw','scrambled_matched'):
        diffs={s:scores[s]['bdt']-scores[s][other] for s in SEEDS}
        out['bdt - '+other]={'per_seed':diffs,'paired_95':paired(diffs.values(),.95),
                           'paired_97_5':paired(diffs.values(),.975)}
    return {'study':'CMS1','training_seed_units':6,'archived_report_files':18,'model_evaluations_performed':0,
            'scores':scores,'contrasts':out,'archive_sha256':origin['archive_sha256'],
            'boundary':'Byte-checked saved REPORT records and offline arithmetic. No new checkpoint, GPU or data-cache verification.'}
