"""Internal, cohort-barriered confirmation evaluator."""
import sys
from pathlib import Path
import common as io
if __name__ == '__main__':
    if len(sys.argv)!=4 or io.SPEC['kind']!='ancestry_confirmation':
        raise SystemExit('Internal confirmation evaluator only')
    import experiment
    experiment.evaluate_report(Path(sys.argv[1]),int(sys.argv[2]),sys.argv[3])
