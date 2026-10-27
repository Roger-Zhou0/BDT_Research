"""Offline checks of recovered BCT2 records and their anonymous derivatives.

This release utility is not the original training coordinator. It uses no model,
network, cache, or checkpoint payload. Original binding digests are not replaced
by hashes of anonymized files; ORIGIN.json distinguishes the two byte streams.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (946301, 946302, 946303, 946304)
ARMS = ('bdt_budget_gs', 'structured_budget_gs')
REPORT_SHA256 = '529b846d764a05147859ab47cd210d07075934246d3ddd9bf15cca2e341a3fd9'
SOURCE_SHA256 = '27a8786f2c0173f4812b46014179d94fd54179f93aa55909874ee262f6f2e80f'
PROTOCOL_SHA256 = '11c36c6ecb55474754d90138c0d1c29e62deef119ba4675a898e2d9d4956db18'
ORIGIN_SHA256 = '29471424f5ec05528fc128b78cfc924db2418a122cf20bad28ab6245b5f30e9d'
# Two-sided 95% Student-t critical value for exactly four paired observations.
T_DF3_975 = 3.182446305284263


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError('Non-finite JSON constant: ' + value)


def _json(raw: bytes) -> Any:
    return json.loads(raw, object_pairs_hook=_no_duplicates, parse_constant=_invalid_constant)


def _path(root: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or not rel.parts or '..' in rel.parts or '\\' in relative:
        raise ValueError('Unsafe record path: ' + relative)
    p = root.joinpath(*rel.parts)
    if not p.resolve().is_relative_to(root.resolve()):
        raise ValueError('Record escapes its directory: ' + relative)
    for q in (p, *p.parents):
        if q == root.parent:
            break
        if q.is_symlink():
            raise ValueError('Symlink not permitted: ' + relative)
    return p


def paired_summary(differences: list[float]) -> dict[str, float | int]:
    if len(differences) != 4 or any(type(x) not in (float, int) or not math.isfinite(x)
                                    for x in differences):
        raise ValueError('BCT2 requires four finite paired differences')
    mean = statistics.mean(differences)
    se = statistics.stdev(differences) / math.sqrt(4)
    return {'confidence': 0.95, 'df': 3, 'mean': mean, 'standard_error': se,
            'lower': mean - T_DF3_975 * se, 'upper': mean + T_DF3_975 * se}


def verify_bct2(directory: Path | None = None) -> dict[str, Any]:
    root = (directory or ROOT / 'reference/bct2').resolve()
    checks: dict[str, bool] = {}

    def check(label: str, condition: bool) -> None:
        checks[label] = bool(condition)
        if not condition:
            raise ValueError('BCT2 verification failed: ' + label)

    origin_raw = _path(root, 'ORIGIN.json').read_bytes()
    check('release_origin_digest', sha256(origin_raw) == ORIGIN_SHA256)
    origin = _json(origin_raw)
    rows = origin['records']
    bindings = {row['path']: row for row in rows}
    expected_receipts = {f'{s}_{a}.json' for s in SEEDS for a in ARMS}
    expected_paths = {'SOURCE_MANIFEST.json', 'SOURCE_DIGEST.txt', 'protocol.json',
                      'run/REPORT/FINAL_REPORT.json', 'run/SUCCESS.json',
                      'run/GLOBAL_ENDPOINT_SEAL.json', 'run/lanes/good/status.json',
                      'run/lanes/default/LOCAL_COMPLETE.json', 'run/lanes/good/LOCAL_COMPLETE.json'}
    expected_paths |= {'run/REPORT/' + name for name in expected_receipts}
    check('seventeen_unique_records', len(rows) == len(bindings) == 17 and set(bindings) == expected_paths)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    check('no_missing_or_unlisted_records', actual == expected_paths | {'ORIGIN.json'})
    raw_records: dict[str, bytes] = {}
    records: dict[str, Any] = {}
    for rel, row in bindings.items():
        raw = _path(root, rel).read_bytes()
        check('public_bytes:' + rel, len(raw) == row['public_bytes'] and sha256(raw) == row['public_sha256'])
        if row['representation'] == 'original_bytes':
            check('original_bytes:' + rel, row['original_sha256'] == sha256(raw)
                  and row['original_bytes'] == len(raw) and not row['changed_string_fields'])
        else:
            check('derivative_label:' + rel, row['representation'] == 'anonymous_derivative'
                  and row['original_sha256'] != sha256(raw) and bool(row['changed_string_fields']))
        raw_records[rel] = raw
        if rel.endswith('.json'):
            records[rel] = _json(raw)

    report = records['run/REPORT/FINAL_REPORT.json']
    success = records['run/SUCCESS.json']
    seal = records['run/GLOBAL_ENDPOINT_SEAL.json']
    protocol = records['protocol.json']
    source_manifest = records['SOURCE_MANIFEST.json']
    check('exact_final_report', len(raw_records['run/REPORT/FINAL_REPORT.json']) == 2506
          and sha256(raw_records['run/REPORT/FINAL_REPORT.json']) == REPORT_SHA256)
    check('completion_state', report['state'] == success['state'] == 'complete'
          and seal['state'] == 'global_eight_endpoint_seal')
    check('success_binds_report', success['final_report_sha256'] == REPORT_SHA256)
    check('original_seal_binding', success['seal_sha256'] == report['seal_sha256']
          == bindings['run/GLOBAL_ENDPOINT_SEAL.json']['original_sha256'])
    check('source_manifest_binding', sha256(canonical_json(source_manifest)) == SOURCE_SHA256
          == report['source'] == success['source'] == seal['source']
          == raw_records['SOURCE_DIGEST.txt'].decode().strip())
    check('protocol_original_binding', bindings['protocol.json']['original_canonical_json_sha256']
          == report['protocol'] == seal['protocol'] == PROTOCOL_SHA256
          and source_manifest['protocol.json'] == bindings['protocol.json']['original_sha256'])
    check('protocol_public_binding', sha256(canonical_json(protocol))
          == bindings['protocol.json']['public_canonical_json_sha256'])
    check('native_source_binding', protocol['native_bct1_source'] == report['native_source'] == seal['native_source'])
    check('registered_cohort', protocol['seeds'] == list(SEEDS) and protocol['arms'] == list(ARMS)
          and set(report['scores']) == set(report['per_parent']) == {str(s) for s in SEEDS})
    check('registered_recipe', protocol['parent_updates'] == 7630 and protocol['prefix_updates'] == 200
          and protocol['endpoint'] == 3000 and protocol['ancestor_map'] == [j for j in range(6) for _ in range(2)]
          and protocol['batch'] == 64 and protocol['microbatch'] * protocol['accumulation'] == 64
          and protocol['context'] == 512 and protocol['post_lr'] == 0.0006)
    check('registered_evaluation', protocol['evaluation'] == seal['report_partition']
          and protocol['evaluation']['tokens'] == 1048576 and protocol['evaluation']['offset'] == 2500000
          and protocol['evaluation']['all_eight_endpoints_before_any_scientific_evaluation'] is True)
    check('four_distinct_parent_identities', set(seal['parents']) == {str(s) for s in SEEDS}
          and len(set(seal['parents'].values())) == 4)
    endpoints = {(e['seed'], e['arm']): e for e in seal['endpoints']}
    check('eight_unique_endpoints', len(seal['endpoints']) == len(endpoints) == 8
          and set(endpoints) == {(s, a) for s in SEEDS for a in ARMS})
    nested = {(e['seed'], e['arm']): e for e in bindings['run/GLOBAL_ENDPOINT_SEAL.json']['nested_contract_bindings']}
    check('eight_contract_origin_bindings', len(nested) == 8 and set(nested) == set(endpoints))
    check('eight_original_receipts', set(report['report_receipts']) == expected_receipts)
    times = {rel: datetime.fromisoformat(row['server_last_modified_utc'].replace('Z', '+00:00')) for rel, row in bindings.items()}
    check('recorded_reporting_order', all(times['run/GLOBAL_ENDPOINT_SEAL.json'] <= times['run/REPORT/' + name]
          <= times['run/REPORT/FINAL_REPORT.json'] <= times['run/SUCCESS.json'] for name in expected_receipts))
    for seed, arm in sorted(endpoints):
        endpoint = endpoints[seed, arm]
        cp, contract = endpoint['checkpoint'], endpoint['contract']
        rel = f'run/REPORT/{seed}_{arm}.json'
        receipt = records[rel]
        check(f'original_receipt:{seed}:{arm}', sha256(raw_records[rel]) == report['report_receipts'][f'{seed}_{arm}.json'])
        check(f'receipt_identity:{seed}:{arm}', receipt['seed'] == seed and receipt['arm'] == arm
              and receipt['source'] == report['source'] and receipt['protocol'] == report['protocol']
              and receipt['seal_sha256'] == report['seal_sha256'])
        check(f'checkpoint_metadata:{seed}:{arm}', receipt['checkpoint_sha256'] == cp['sha256']
              and receipt['model'] == cp['identity']['model'] and cp['step'] == cp['identity']['phase'] == 3000
              and cp['identity']['offset'] == 7630 and cp['identity']['age'] == 10630)
        check(f'contract_metadata:{seed}:{arm}', contract['seed'] == seed and contract['mode'] == arm
              and contract['start'] == 200 and contract['end'] == 3000 and contract['offset'] == 7630
              and contract['source'] == report['source'] and contract['protocol'] == report['protocol']
              and contract['native_source'] == report['native_source']
              and contract['cache'] == protocol['cache_sha256'])
        check(f'contract_derivative:{seed}:{arm}', cp['contract_sha256'] == nested[seed, arm]['original_canonical_sha256']
              and sha256(canonical_json(contract)) == nested[seed, arm]['public_canonical_sha256'])
        check(f'evaluation_and_runtime:{seed}:{arm}', receipt['runtime'] == seal['runtime'] == contract['runtime']
              and receipt['offset'] == protocol['evaluation']['offset']
              and receipt['tokens'] == protocol['evaluation']['tokens'])
        check(f'score_binding:{seed}:{arm}', set(report['scores'][str(seed)]) == set(ARMS)
              and math.isfinite(receipt['loss']) and receipt['loss'] == report['scores'][str(seed)][arm])

    for lane in ('default', 'good'):
        local = records[f'run/lanes/{lane}/LOCAL_COMPLETE.json']
        check('lane_completion:' + lane, local['state'] == 'local_complete' and local['lane'] == lane
              and local['seeds'] == protocol['lanes'][lane] and local['source'] == report['source']
              and local['science_proofs'] == local['ledger']['counters']['science'] == 26860)
        tasks = local['ledger']['tasks']
        scientific = [t for t in tasks if t['task'].startswith('bct2_workspace/run/families/')]
        check('lane_science_accounting:' + lane, len(scientific) == 8
              and sum(t['proved'] for t in scientific) == 26860
              and len({t['task'] for t in scientific}) == 8)
    check('science_total', protocol['budgets']['science_total'] == 2 * 26860)
    check('completion_status', records['run/lanes/good/status.json']['state'] == 'complete'
          and records['run/lanes/good/status.json']['holding_cuda'] is False)
    differences = [report['scores'][str(s)][ARMS[0]] - report['scores'][str(s)][ARMS[1]] for s in SEEDS]
    summary = paired_summary(differences)
    check('per_parent_arithmetic', all(d == report['per_parent'][str(s)] for s, d in zip(SEEDS, differences)))
    check('paired_interval', all(math.isclose(float(v), float(report['interval95'][k]), rel_tol=1e-12, abs_tol=1e-15)
                                  for k, v in summary.items()))
    check('registered_primary', report['primary'] == protocol['primary'] == 'bdt_budget_gs - structured_budget_gs')
    check('primary_criterion', report['negative_signs'] == sum(d < 0 for d in differences) == 4
          and summary['upper'] < 0 and summary['standard_error'] > 0
          and report['primary_criterion_met'] is True and report['zero_variance_requires_review'] is False)
    return {'passed': True, 'study': 'BCT2', 'fresh_parents': 4, 'parent_seeds': list(SEEDS),
            'original_score_receipts': 8, 'recovered_records': 17, 'byte_identical_records': 12,
            'anonymous_derivatives': 5, 'checks_passed': len(checks), 'checks': checks,
            'scores': report['scores'], 'per_parent': report['per_parent'], 'paired_95': summary,
            'negative_signs': 4, 'primary_criterion_met': True, 'pooled_with_BCT1': False,
            'original_final_report_sha256': REPORT_SHA256, 'new_model_evaluations': 0,
            'boundary': origin['boundary']}
