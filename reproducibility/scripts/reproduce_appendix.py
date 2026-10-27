#!/usr/bin/env python3
"""Regenerate selected original appendix tables/figures in a NEW local workspace.

Uses shipped numerical records only. Historical source is copied unchanged from
inspectable text into the new workspace. Original source/evidence stays read-only.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from reproduce_all import manuscript_inventory
sys.path.insert(0,str(ROOT))
from repro.appendix_records import allocation_record_check
SCRIPTS=('build_new_evidence.py','build_transfer_evidence.py','build_osc_evidence.py','check_osc_geometry.py')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();out=a.out.resolve()
    if out.exists():raise FileExistsError('Choose a NEW output directory')
    if out.is_relative_to(ROOT) and not out.is_relative_to(ROOT/'build'):raise ValueError('Inside the release, write only under build/')
    manuscript_inventory();out.mkdir(parents=True)
    source=ROOT/'reference/manuscript_v20';work=out/'source';work.mkdir();(work/'scripts').mkdir();(work/'figures').mkdir()
    for folder in ('evidence','tables'):shutil.copytree(source/folder,work/folder)
    script_hashes={}
    for name in SCRIPTS:
        raw=(source/'figure_scripts'/(name+'.txt')).read_bytes();(work/'scripts'/name).write_bytes(raw);script_hashes[name]=hashlib.sha256(raw).hexdigest()
    # These are explicit new editorial-input manifests over anonymous copied
    # records, not recreated historical scientific manifests or endpoint seals.
    for folder in ('final_report_20260923','oip1'):
        p=work/'evidence'/folder
        m={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.iterdir()) if f.is_file() and f.name!='EDITORIAL_INPUT_MANIFEST.json'}
        (p/'EDITORIAL_INPUT_MANIFEST.json').write_text(json.dumps(m,sort_keys=True,indent=2)+'\n')
    # Presentation-only setting keeps long original axis labels inside the exported figure.
    # The archived plotting source, numerical inputs and table bytes stay unchanged.
    (out/'plot_cache').mkdir()
    (out/'plot_cache/matplotlibrc').write_text('savefig.bbox: tight\nsavefig.pad_inches: 0.3\n')
    rows=[];env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',MPLCONFIGDIR=str(out/'plot_cache'))
    for name in SCRIPTS:
        with (out/(name+'.console.txt')).open('x') as log:r=subprocess.run([sys.executable,str(work/'scripts'/name)],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        rows.append({'script':name,'source_sha256':script_hashes[name],'returncode':r.returncode})
        if r.returncode:raise RuntimeError('Original manuscript script failed; retained console: '+name)
    names=['eac1_v15.tex','dnt_v15.tex','cta1_v15.tex','toc1_v15.tex','adjacent_report_scores.tex','art_report_scores.tex','bct_report_scores.tex','adjacent_report_contrasts.tex','art_report_contrasts.tex','bct_report_contrasts.tex','oip_all_state_metrics.tex','osc1_scores.tex','osc1_primary.tex','osc1_manipulation.tex']
    tables=[]
    for name in names:
        raw=(work/'tables'/name).read_bytes();same=raw==(source/'tables'/name).read_bytes();tables.append({'name':name,'matches_shipped_table_bytes':same,'sha256':hashlib.sha256(raw).hexdigest()})
        if not same:raise RuntimeError('Regenerated table differs from v20: '+name)
    allocation=allocation_record_check()
    figure_names=[str(f.relative_to(out)) for f in sorted((work/'figures').rglob('*.pdf'))]
    result={'state':'passed','new_model_evaluations':0,'new_training_updates':0,'original_scripts':rows,'byte_matching_regenerated_tables':tables,'allocation_public_record_verification':allocation,'pdf_figures_generated':figure_names,'plotting_adjustment':'New wrapper sets savefig.bbox=tight and0.3-inch padding to prevent clipping; no data/style-color changes or edits to archived figure scripts.','editorial_input_manifests':'New anonymous local input manifests; original native source and endpoint hashes are not forged or rewritten.','scope':'Four selected original manuscript display/check scripts plus a new anonymous-record allocation verifier, fourteen regenerated table fragments and retained-record geometry. Not regeneration of all42 tables or all historical raw logs. The extra AOT1 author-only DEV plot remains labelled descriptive and is not promoted to a new REPORT result.'}
    (out/'APPENDIX_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
