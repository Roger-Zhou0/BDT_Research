"""Recompute CMS1 from unrounded archived REPORT records; requires SciPy."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repro.archived_cms1 import recompute

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.parse_args()
    print(json.dumps(recompute(),indent=2))
