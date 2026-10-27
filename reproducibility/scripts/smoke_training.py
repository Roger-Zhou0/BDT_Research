"""Exercise the real policy on a small synthetic six-to-twelve-layer model."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # This entrypoint is deliberately CPU-only.
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repro.training import pipeline

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('build/cpu_smoke'))
    args=p.parse_args()
    print(json.dumps(pipeline(args.out,smoke=True,resume_check=True),indent=2))

if __name__=='__main__':main()
