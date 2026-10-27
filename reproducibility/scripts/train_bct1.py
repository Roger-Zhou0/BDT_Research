"""Explicit full-size reproduction template. Not GPU-parity certified for submission."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True,help='Exact S1-B token cache, supplied separately')
    p.add_argument('--out',type=Path,required=True,help='A NEW local output directory')
    p.add_argument('--seed',type=int,choices=[946201,946202,946203,946204],required=True)
    p.add_argument('--allow-training',action='store_true',help='Authorize thousands of GPU updates in this invocation')
    args=p.parse_args()
    if not args.allow_training:p.error('No training started: explicit --allow-training is required')
    from repro.training import pipeline
    print(json.dumps(pipeline(args.out,data=args.data,seed=args.seed,smoke=False,resume_check=False),indent=2))

if __name__=='__main__':main()
