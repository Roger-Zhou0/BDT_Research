"""Inspect an explicit log list; export only with --export. Never downloads files."""
from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repro.log_export import Limits, MiB, GiB, plan, export


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--export', action='store_true')
    p.add_argument('--redactions', type=Path, help='Private literal-to-placeholder map; not included in output')
    p.add_argument('--strip-ansi', action='store_true')
    p.add_argument('--max-file-mib', type=int, default=8)
    p.add_argument('--max-input-mib', type=int, default=64)
    p.add_argument('--max-output-mib', type=int, default=100)
    p.add_argument('--min-free-gib', type=int, default=10)
    a = p.parse_args()
    limits = Limits(a.max_file_mib*MiB, a.max_input_mib*MiB, a.max_output_mib*MiB, a.min_free_gib*GiB)
    spec = json.loads(a.spec.read_text())
    if not a.export:
        result = plan(spec, limits)
    else:
        if a.out is None:
            p.error('--out is required with --export')
        redactions = json.loads(a.redactions.read_text()) if a.redactions else {}
        result = export(spec, a.out, redactions=redactions, strip_ansi=a.strip_ansi, limits=limits)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
