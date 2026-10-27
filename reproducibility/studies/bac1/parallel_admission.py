"""Outcome-independent admission for the authorized parallel BAC1 release."""
from pathlib import Path
import common as io


def prerequisites(project, lane_override=None):
    import followup_queue as q
    name = 'balanced_ancestry_assignment_v1'
    d = Path(project) / 'runs' / name
    if not (d / 'PROTOCOL.json').is_file() or not (d / 'source_registration.json').is_file():
        return False, 'Waiting for predecessor registration'
    expected = io.SPEC['prerequisite_sources'][name]
    protocol = io.read(d / 'PROTOCOL.json')
    registered = io.read(d / 'source_registration.json')
    if any(registered.get(k) != v for k, v in expected.items()) or io.digest(protocol) != expected['protocol']:
        raise RuntimeError('Predecessor source/protocol mismatch')
    lane = lane_override or q.lane()
    seeds = io.SPEC['local_predecessor_seeds'][lane]
    status = d / 'lanes' / lane / 'status.json'
    if not status.is_file() or io.read(status).get('state') not in ('training_complete', 'finished'):
        return False, 'Waiting for local BAA1 lane completion, not its efficacy outcome'
    for seed in seeds:
        for arm in protocol['arms']:
            f = d / ('seed_' + str(seed)) / 'arms' / arm / 'result.json'
            if not f.is_file():
                return False, 'Local BAA1 endpoint still missing'
            v = io.read(f)
            contract = v.get('contract', {})
            if (v.get('state') != 'complete' or v.get('step') != protocol['post_updates']
                    or contract.get('seed') != seed or contract.get('mode') != arm
                    or any(contract.get(k) != h for k, h in expected.items())):
                raise RuntimeError('Local BAA1 endpoint identity/integrity failure')
    return True, 'Local predecessor complete; global efficacy is not an admission condition'
