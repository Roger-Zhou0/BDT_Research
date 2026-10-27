"""Recompute final REPORT and allocation diagnostics offline (standard library)."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repro.results import write_results

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('build/report'))
    p.add_argument('--include-inherited',action='store_true',help='Also use SciPy to recompute older display-precision summaries')
    a=p.parse_args();r=write_results(a.out,a.include_inherited)
    print(json.dumps({'output_directory':str(a.out),'BCT1_primary':r['contrasts']['bct']['bdt_budget_gs - structured_budget_gs'],
                      'allocation':r['allocation_diagnostics']},indent=2))

if __name__=='__main__':main()
