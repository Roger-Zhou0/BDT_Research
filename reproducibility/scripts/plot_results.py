"""Recreate data views for v20 figures. Layout is not a pixel-exact manuscript build."""
from pathlib import Path
import argparse,csv,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repro.results import ROOT,report_results,allocation_results

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path('build/plots'));a=p.parse_args()
    if a.out.exists():raise FileExistsError('Choose a new output directory')
    if a.out.resolve().is_relative_to(ROOT/'data') or a.out.resolve().is_relative_to(ROOT/'native'):raise ValueError('Read-only inputs')
    report_results();allocation_results()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    a.out.mkdir(parents=True)
    manifest={}
    # Figure 2: display precision only, each parent shown separately.
    rows=json.loads((ROOT/'data/display_tables.json').read_text())['tables']['bac1_all_endpoints']
    fig,ax=plt.subplots(figsize=(7.2,3.6))
    for col,name,shift in [(2,'Canonical BDT',-.18),(3,'Ancestry-preserving',0),(4,'Reassigned',.18)]:
        ax.scatter(np.arange(len(rows))+shift,[float(r[col])-float(r[1]) for r in rows],label=name)
    ax.axhline(0,linestyle='--',linewidth=.8);ax.set_xticks(range(len(rows)),[r[0] for r in rows]);ax.set_xlabel('Parent');ax.set_ylabel('Loss minus parent AdamW (nats/token)');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(a.out/'figure2_ancestry.png',dpi=180);plt.close(fig)
    manifest['figure2_ancestry.png']='v20 Figure2 numerical content from display-precision BAC1 inputs; one point per parent/policy.'
    # Figure 3 is a mean over TWO reused parents, not a new population interval.
    d=json.loads((ROOT/'data/asq1_display_inputs.json').read_text());fig,ax=plt.subplots(figsize=(6,3.6))
    for kind,label in [('true','True ancestry'),('false','False pairing')]:
        yy=[np.mean(np.array(d[kind+'_'+op])-np.array(d['adamw'])) for op in ('suppress','amplify')]
        ax.plot([0,1],yy,marker='o',label=label)
    ax.axhline(0,linestyle='--',linewidth=.8);ax.set_xticks([0,1],['Suppress','Amplify']);ax.set_ylabel('Mean loss minus AdamW (nats/token)');ax.legend();fig.tight_layout();fig.savefig(a.out/'figure3_operation.png',dpi=180);plt.close(fig)
    manifest['figure3_operation.png']='v20 Figure3 numerical content; two reused parents; no confidence intervals.'
    with (ROOT/'data/allocation_diagnostics.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    phases=sorted({int(r['phase']) for r in rows})
    for branch in ('bdt_budget_gs','structured_budget_gs'):
        for proj in ('gate','up','down'):
            fig,ax=plt.subplots(figsize=(6.4,3.6))
            for col,label in [('qD_full','Full-BDT candidate'),('qD_structured','Structured candidate')]:
                v=[np.array([float(r[col]) for r in rows if r['branch_state']==branch and r['projection']==proj and int(r['phase'])==phase]) for phase in phases]
                means=np.array([q.mean() for q in v]);lo=means-np.array([q.min() for q in v]);hi=np.array([q.max() for q in v])-means
                ax.errorbar(phases,means,yerr=np.stack([lo,hi]),marker='o',markersize=3,linewidth=1,capsize=2,label=label)
            ax.set_xlabel('Retained phase');ax.set_ylabel('Difference-sector energy fraction');ax.set_title(f'{proj}: {branch}');ax.legend(fontsize=8);fig.tight_layout()
            name=f'allocation_{branch}_{proj}.png';fig.savefig(a.out/name,dpi=180);plt.close(fig)
            manifest[name]='v20 Figure4 / Appendix allocation: four-parent means and parent ranges, not CIs; sparse observations, lines are guides.'
    (a.out/'PLOT_PROVENANCE.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Saved',len(manifest),'separate plots to',a.out)

if __name__=='__main__':main()
