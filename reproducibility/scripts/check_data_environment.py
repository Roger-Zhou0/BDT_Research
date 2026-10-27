"""Inspect archived runtime pins and optional external data, with no downloads."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repro.data_audit import inspect_cache,recorded_environment_comparison


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,help='An existing cache on this machine; never downloaded')
    p.add_argument('--hash-cache',action='store_true',help='Read the existing 6.01 GB token cache in place; writes nothing')
    a=p.parse_args()
    if a.hash_cache and a.cache is None:p.error('--cache is required with --hash-cache')
    r={'environment':recorded_environment_comparison()}
    if a.cache is not None:r['cache']=inspect_cache(a.cache,hash_bytes=a.hash_cache)
    print(json.dumps(r,indent=2))
    if a.hash_cache and not r['cache']['exact_cache_verified']:sys.exit(2)

if __name__=='__main__':main()
