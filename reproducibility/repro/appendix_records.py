"""Public-record adaptation of the archived allocation verifier.

Numerical identities follow the original verify_allocation.py. Public and original
byte digests differ because record names are anonymous; no historical hash is
rewritten or treated as authentication of changed bytes.
"""
from pathlib import Path
import hashlib,json,math,statistics
ROOT=Path(__file__).resolve().parents[1]

def allocation_record_check(root=None):
    root=ROOT/'reference/manuscript_v20' if root is None else Path(root)
    origin=json.loads((root/'ORIGIN.json').read_text())
    mapped={r['original_source_member']:r for r in origin['inputs']}
    folder=root/'evidence/budget_interpretability'
    raw=(folder/'plot_data_compact.json').read_bytes();a=(folder/'SAVED_SNAPSHOT_AUDIT.json').read_bytes()
    for name,b in [('plot_data_compact.json',raw),('SAVED_SNAPSHOT_AUDIT.json',a)]:
        entry=mapped['evidence/budget_interpretability/'+name]
        if hashlib.sha256(b).hexdigest()!=entry['public_sha256']:raise RuntimeError('Public allocation record bytes changed')
    d=json.loads(raw);audit=json.loads(a);original=mapped['evidence/budget_interpretability/plot_data_compact.json']['original_sha256']
    if original!=audit['input_hashes']['plot_data_compact.json']:raise RuntimeError('Lost original allocation provenance')
    if audit['state']!='passed' or audit['records']!=240:raise RuntimeError('Wrong retained-record audit')
    if d['full_precision_source_sha256']!=audit['input_hashes']['BUDGET_DIAGNOSTICS.json']:raise RuntimeError('Source projection linkage differs')
    if d['native_source']!=audit['source'] or d['native_source']!='c63c877701dc3feb22b4b7e0bf01166beba3775e39cbc9603b487363e669f3fa':raise RuntimeError('Wrong native BCT1 source')
    if d['checked']['new_checkpoint_payload_audit'] is not False:raise RuntimeError('Do not promote retained-record check to checkpoint audit')
    if not all(v for k,v in d['checked'].items() if k!='new_checkpoint_payload_audit'):raise RuntimeError('Incomplete original checks')
    if any(audit['input_hashes'].get(k)!=v for k,v in d['snapshot_hashes'].items()):raise RuntimeError('Snapshot linkage differs')
    expected=([946201,946202,946203,946204],['bdt_budget_gs','structured_budget_gs'],[201,210,250,300,310,400,420,1000,2000,3000],['gate','up','down'])
    if tuple(d[k] for k in ('seeds','branches','phases','projections'))!=expected:raise RuntimeError('Grid differs')
    rows=[];maxerr=0.
    for si,s in enumerate(d['seeds']):
        if len(d['values'][si])!=2:raise RuntimeError('Branch grid differs')
        for bi,b in enumerate(d['branches']):
            if len(d['values'][si][bi])!=10:raise RuntimeError('Phase grid differs')
            for ti,phase in enumerate(d['phases']):
                rho=2**min(1.,max(0.,(phase-200)/200))
                if len(d['values'][si][bi][ti])!=3:raise RuntimeError('Projection grid differs')
                for qi,q in enumerate(d['projections']):
                    qf,qs,saved=d['values'][si][bi][ti][qi]
                    if not all(math.isfinite(x) for x in (qf,qs,saved)) or not(0<qf<1 and 0<qs<1):raise RuntimeError('Invalid retained geometry')
                    qr=qs/(rho*rho*(1-qs)+qs);effective=rho*math.sqrt(qf*(1-qs)/((1-qf)*qs))
                    other=math.sqrt(qf/(1-qf)*(1-qr)/qr)
                    if abs(effective-other)>=2e-14 or abs(effective*effective*qr/(1-qr+effective*effective*qr)-qf)>=5e-15:raise RuntimeError('Allocation identity differs')
                    maxerr=max(maxerr,abs(effective-saved));rows.append(dict(seed=s,branch=b,phase=phase,projection=q,qD_full=qf,qD_structured=qs,qD_raw_inferred=qr,rho_effective_full=effective))
    if len(rows)!=240 or maxerr>=1e-8:raise RuntimeError('Retained-value reconstruction failed')
    if not all(r['qD_full']>r['qD_structured'] for r in rows if r['phase']==400):raise RuntimeError('Early ordering differs')
    if not all(r['qD_full']<r['qD_structured'] for r in rows if r['phase']==3000):raise RuntimeError('Late ordering differs')
    if not all(r['rho_effective_full']>1 for r in rows):raise RuntimeError('Effective amplification differs')
    text=(root/'tables/budget_profile_means.tex').read_text();matched=0
    for b in d['branches']:
        for phase in (400,3000):
            for q in d['projections']:
                rr=[r for r in rows if (r['branch'],r['phase'],r['projection'])==(b,phase,q)]
                for k in ('qD_full','qD_structured','qD_raw_inferred','rho_effective_full'):
                    if f'{statistics.mean(r[k] for r in rr):.6f}' not in text:raise RuntimeError('Displayed allocation mean differs')
                    matched+=1
    return {'state':'passed','records':240,'parents':4,'original_payload_sha256':original,'public_payload_sha256':hashlib.sha256(raw).hexdigest(),'displayed_mean_cells_checked':matched,'max_effective_scale_error':maxerr,'new_model_evaluations':0,'new_checkpoint_audit':False,'original_hashes_modified':False,'scope':'Anonymous public-byte integrity plus original-provenance links and numerical reconstruction from retained nine-decimal values; not a new original-snapshot/tensor audit.'}
