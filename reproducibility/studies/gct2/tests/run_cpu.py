#!/usr/bin/env python3
"""Isolated CPU admission tests. All test outputs use temporary directories."""
import argparse,os,subprocess,sys,tempfile,json
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path);ap.add_argument('--study',choices=['gct2','bco1'],required=True);ap.add_argument('--output',type=Path);a=ap.parse_args()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with tempfile.TemporaryDirectory(prefix='confirmation_cpu_tests_') as td:
        commands=[['-m','unittest','discover','-s',str(HERE/'design/tests'),'-v'],[str(HERE/'tests/test_runtime.py')],
                  [str(HERE/'design/candidate/test_candidate.py'),'--reference',str(HERE/'backend/tempering_reference.py')],
                  [str(HERE/'tests/integration.py'),'--study',a.study,'--output',str(Path(td)/'integration.json')]]
        for command in commands:
            print('CPU TEST COMMAND:',*command,flush=True)
            r=subprocess.run([sys.executable,*command],env=env,cwd=td)
            if r.returncode:raise SystemExit(r.returncode)
        result=json.loads((Path(td)/'integration.json').read_text())
        if a.output:
            a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
        print('GCT2/BCO1 CPU RELEASE SUITE: PASS',a.study,flush=True)
if __name__=='__main__':main()
