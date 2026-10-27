"""Bounded, non-destructive log views; never selects scientific endpoints.

Only explicit input files are read. A public view is a derivative, not a repaired
original. All malformed lines, duplicate records, exceptions and restart markers
remain visible. Source paths and literal redaction maps are not put in the export.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from typing import Any

MiB = 1024 ** 2
GiB = 1024 ** 3
SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,80}\Z")
ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


@dataclass(frozen=True)
class Limits:
    max_file_bytes: int = 8 * MiB
    max_input_bytes: int = 64 * MiB
    max_output_bytes: int = 100 * MiB
    minimum_free_bytes: int = 10 * GiB

    def validate(self) -> None:
        if min(self.max_file_bytes, self.max_input_bytes, self.max_output_bytes) <= 0:
            raise ValueError('All byte ceilings must be positive')
        if self.minimum_free_bytes < 0:
            raise ValueError('Free-space reserve cannot be negative')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json(text: str) -> Any:
    def invalid(value: str) -> None:
        raise ValueError('Nonfinite JSON number: ' + value)
    value = json.loads(text, parse_constant=invalid)
    # Reject overflow such as 1e999 as well as explicit NaN/Infinity tokens.
    json.dumps(value, allow_nan=False)
    return value


def _stamp(s: os.stat_result) -> tuple:
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def plan(spec: dict, limits: Limits = Limits()) -> dict:
    """Metadata-only preflight. Does not read log contents or make directories."""
    limits.validate()
    rows = spec.get('inputs')
    if not isinstance(rows, list) or not 1 <= len(rows) <= 256:
        raise ValueError('Supply an explicit list of 1 to 256 input files')
    ids = set(); result = []; total = 0
    for row in rows:
        for key in ('source_id', 'run_id', 'attempt_id'):
            if not isinstance(row.get(key), str) or not SAFE_ID.fullmatch(row[key]):
                raise ValueError('Use neutral identifiers for ' + key)
        if row['source_id'] in ids:
            raise ValueError('Duplicate source_id')
        ids.add(row['source_id'])
        if row.get('format') not in ('text', 'json', 'jsonl'):
            raise ValueError('format must be text, json or jsonl')
        if row.get('capture_scope', 'unknown') not in ('unknown', 'file_snapshot', 'excerpt'):
            raise ValueError('Unrecognized capture_scope')
        if row.get('run_state', 'unknown') not in ('unknown', 'interrupted', 'resumed', 'failed', 'complete'):
            raise ValueError('Unrecognized declared run_state')
        path = Path(row['path']).expanduser()
        if path.is_symlink():
            raise ValueError('Symlink input is refused')
        s = path.stat()
        if not stat.S_ISREG(s.st_mode):
            raise ValueError('Only regular files are accepted')
        if path.suffix.lower() in ('.bin', '.pt', '.pth', '.safetensors', '.zip', '.gz', '.xz', '.db', '.sqlite'):
            raise ValueError('This tool does not export datasets, checkpoints, archives or databases')
        if s.st_size > limits.max_file_bytes:
            raise ValueError('Per-file size ceiling exceeded; no automatic truncation')
        total += s.st_size
        if total > limits.max_input_bytes:
            raise ValueError('Total input size ceiling exceeded')
        result.append({'source_id': row['source_id'], 'input_bytes': s.st_size})
    return {'mode': 'METADATA_ONLY_PLAN', 'files': result, 'total_input_bytes': total,
            'maximum_output_bytes': limits.max_output_bytes,
            'minimum_free_bytes': limits.minimum_free_bytes, 'content_read': False,
            'files_written': 0, 'no_recursive_collection': True}


def validate_redactions(redactions: dict[str, str]) -> None:
    if not isinstance(redactions, dict):
        raise ValueError('Redactions must be a literal-string map')
    for old, new in redactions.items():
        if not isinstance(old, str) or len(old) < 4 or not any(c.isalpha() for c in old):
            raise ValueError('Redaction keys need at least four characters and a letter')
        if not isinstance(new, str) or not re.fullmatch(r'<REDACTED_[A-Z0-9_]+>', new):
            raise ValueError('Redaction values must be explicit neutral placeholders')
        if '\n' in old or '\r' in old:
            raise ValueError('Redactions must not remove line boundaries')


def _redact(text: str, redactions: dict[str, str]) -> tuple[str, dict[str, int]]:
    counts = {}
    for old in sorted(redactions, key=len, reverse=True):
        n = text.count(old)
        if n:
            counts[redactions[old]] = counts.get(redactions[old], 0) + n
            text = text.replace(old, redactions[old])
    return text, counts


def _structure_and_numbers(value: Any) -> Any:
    """String values may be redacted; keys, shapes and all other values must match."""
    if isinstance(value, dict):
        return ('dict', tuple((key, _structure_and_numbers(v)) for key, v in sorted(value.items())))
    if isinstance(value, list):
        return ('list', tuple(_structure_and_numbers(v) for v in value))
    if isinstance(value, str):
        return ('string',)
    return (type(value).__name__, value)


def _check_numerical_redaction(original: str, view: str, format_name: str) -> None:
    if original.count('\n') != view.count('\n') or original.count('\r') != view.count('\r'):
        raise ValueError('A transformation would remove line boundaries')
    if format_name == 'text':
        return
    before = original.splitlines() if format_name == 'jsonl' else [original]
    after = view.splitlines() if format_name == 'jsonl' else [view]
    if len(before) != len(after):
        raise ValueError('A transformation would remove source lines')
    for left, right in zip(before, after):
        try:
            a = strict_json(left)
        except (ValueError, TypeError):
            continue  # Malformed originals remain in the derivative, with diagnostics.
        try:
            b = strict_json(right)
        except (ValueError, TypeError) as exc:
            raise ValueError('Redaction would corrupt a valid JSON record') from exc
        if _structure_and_numbers(a) != _structure_and_numbers(b):
            raise ValueError('Redaction would change JSON keys, structure or numeric values')


def privacy_flags(text: str) -> list[str]:
    """Conservative gate, not a complete secret/anonymity scanner."""
    checks = {
        'absolute_home_path': r'/(?:Us' + r'ers|ho' + r'me)/[^\s/]+/',
        'email_like_string': r'[A-Za-z0-9._%+-]+@' + r'[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
        'private_key_header': r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE' + r' KEY-----',
        'private_notebook_url': r'https?://[^\s]+/' + r'user/[^\s]+',
        'credential_assignment': r'(?i)(?:api_key|access_token|password)\s*[:=]\s*["\x27][^"\x27\s]{8,}',
    }
    return [name for name, pattern in checks.items() if re.search(pattern, text)]


def inspect_text(text: str, format_name: str) -> dict:
    """Diagnostic parsing only. Invalid/duplicate lines are never removed."""
    lines = text.splitlines(keepends=True)
    issues = []; duplicates = []; repeated_steps = []; regressions = []
    first_record = {}; first_step = {}; previous_step = None; valid = 0; blank = 0
    if format_name == 'json':
        try:
            strict_json(text); valid = 1
        except (ValueError, TypeError) as exc:
            issues.append({'kind': 'invalid_json_document', 'error_type': type(exc).__name__})
    elif format_name == 'jsonl':
        for number, line in enumerate(lines, 1):
            if not line.strip():
                blank += 1; continue
            try:
                value = strict_json(line)
            except (ValueError, TypeError) as exc:
                final_unterminated = number == len(lines) and not line.endswith(('\n', '\r'))
                issues.append({'line': number,
                               'kind': 'invalid_unterminated_final_line' if final_unterminated else 'invalid_json_line',
                               'error_type': type(exc).__name__})
                continue
            valid += 1
            key = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
            if key in first_record:
                duplicates.append({'line': number, 'same_record_as_line': first_record[key]})
            else:
                first_record[key] = number
            step = value.get('phase', value.get('step')) if isinstance(value, dict) else None
            if isinstance(step, int) and not isinstance(step, bool):
                if step in first_step:
                    repeated_steps.append({'line': number, 'step': step, 'earlier_line': first_step[step]})
                else:
                    first_step[step] = number
                if previous_step is not None and step < previous_step:
                    regressions.append({'line': number, 'previous_step': previous_step, 'step': step})
                previous_step = step
    return {'physical_lines': len(lines), 'valid_json_records': valid, 'blank_lines': blank,
            'ends_with_newline': text.endswith(('\n', '\r')) if text else None,
            'parse_issues': issues, 'duplicate_records': duplicates,
            'repeated_steps': repeated_steps, 'step_regressions': regressions,
            'interpretation': 'Signals requiring provenance review; not proof of run failure, completion or independent replication.'}


def export(spec: dict, out: Path, *, redactions: dict[str, str] | None = None,
           strip_ansi: bool = False, limits: Limits = Limits()) -> dict:
    """Produce a new derivative bundle after a full size/privacy/stability preflight."""
    plan(spec, limits)
    redactions = {} if redactions is None else redactions
    validate_redactions(redactions)
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError('Choose a new output directory; existing outputs are never overwritten')
    if not out.parent.is_dir():
        raise ValueError('Output parent must already exist')
    free = shutil.disk_usage(out.parent).free
    # Reserve the entire permitted output ceiling, not an optimistic compression estimate.
    if free - limits.max_output_bytes < limits.minimum_free_bytes:
        raise OSError('Insufficient free space after the configured reserve')
    payloads = {}; inventory = []; stamps = []; total = 0
    for item in spec['inputs']:
        path = Path(item['path']).expanduser()
        before = path.stat()
        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, 'rb') as handle:
            opened = os.fstat(handle.fileno())
            if _stamp(opened) != _stamp(before):
                raise RuntimeError('Input changed between preflight and open')
            if opened.st_size > limits.max_file_bytes:
                raise ValueError('Input grew beyond the per-file limit')
            raw = handle.read(limits.max_file_bytes + 1)
            after = os.fstat(handle.fileno())
        if len(raw) != opened.st_size or _stamp(opened) != _stamp(after) or _stamp(path.stat()) != _stamp(after):
            raise RuntimeError('Input changed during capture; no partial snapshot exported')
        total += len(raw)
        if total > limits.max_input_bytes:
            raise ValueError('Total input ceiling exceeded during capture')
        text = raw.decode('utf-8')  # Refuse damaged encoding rather than silently replacing bytes.
        diagnostics = inspect_text(text, item['format'])
        view, counts = _redact(text, redactions)
        ansi_count = len(ANSI.findall(view)) if strip_ansi else 0
        if strip_ansi:
            view = ANSI.sub('', view)
        _check_numerical_redaction(text, view, item['format'])
        if privacy_flags(view):
            raise ValueError('Unredacted privacy indicators: ' + ', '.join(privacy_flags(view)))
        target = 'views/' + item['source_id'] + '.txt'
        encoded = view.encode('utf-8'); payloads[target] = encoded
        entry = {'source_id': item['source_id'], 'run_id': item['run_id'], 'attempt_id': item['attempt_id'],
                 'format': item['format'], 'declared_capture_scope': item.get('capture_scope', 'unknown'),
                 'declared_run_state': item.get('run_state', 'unknown'),
                 'source_sha256': sha(raw), 'source_bytes': len(raw), 'view': target,
                 'export_sha256': sha(encoded), 'export_bytes': len(encoded), 'diagnostics': diagnostics,
                 'redaction_counts_by_placeholder': counts, 'ansi_sequences_removed': ansi_count,
                 'run_completion_verified': False, 'source_modified': False,
                 'records_removed_or_imputed': 0}
        if 'resumed_from_attempt' in item:
            if not SAFE_ID.fullmatch(item['resumed_from_attempt']):
                raise ValueError('Unsafe resumed_from_attempt identifier')
            entry['declared_resumed_from_attempt'] = item['resumed_from_attempt']
        inventory.append(entry); stamps.append((path, _stamp(after)))
    manifest = {'format': 'non_destructive_log_views_v1', 'inputs': inventory,
                'total_source_bytes': total, 'source_paths_exported': False,
                'scientific_endpoint_selection': 'none', 'raw_originals_remain_at_source': True,
                'resume_chains_verified': False, 'automatic_deduplication': False,
                'privacy_boundary': 'Literal redactions and heuristic checks; manual anonymity review still required.'}
    payloads['INVENTORY.json'] = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
    payloads['README.txt'] = (
        'These are derived views of explicitly selected inputs, not cleaned originals.\n'
        'All lines, failures, restarts, duplicates and incomplete final records remain.\n'
        'INVENTORY.json records source/view hashes, declared scope and transformations.\n'
        'Complete capture of a supplied file does not establish a complete original run.\n'
        'No endpoint was selected, no missing metric filled, no restart deduplicated.\n'
        'Interrupted attempts sharing a parent remain one dependent training history.\n'
    ).encode()
    size = sum(len(b) for b in payloads.values())
    if size > limits.max_output_bytes:
        raise ValueError('Output ceiling exceeded; originals left untouched')
    for path, expected in stamps:
        if _stamp(path.stat()) != expected:
            raise RuntimeError('Source changed before export commit')
    # Temporary files belong solely to this new export. No source or prior output is deleted.
    with tempfile.TemporaryDirectory(prefix='.log_export_', dir=out.parent) as stage_name:
        stage = Path(stage_name)
        for rel, data in payloads.items():
            dest = stage / rel; dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as handle:
                handle.write(data)
        # mkdir fails even for a racing empty destination; never replace a previous bundle.
        out.mkdir()
        for child in stage.iterdir():
            os.rename(child, out / child.name)
    return {'state': 'DERIVATIVE_EXPORT_COMPLETE', 'input_files': len(inventory),
            'total_source_bytes': total, 'total_export_bytes': size,
            'source_modified': False, 'run_completion_verified': False}
