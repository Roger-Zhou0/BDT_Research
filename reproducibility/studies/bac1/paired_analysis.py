"""Fixed six-family primary analysis and descriptive window-policy readouts."""
from __future__ import annotations
import math
import statistics

T5_975 = 2.5705818356363146

def six_parent_contrast(rows, seeds, left, right):
    if len(seeds) != 6 or len(set(seeds)) != 6:
        raise ValueError('Six distinct registered families required')
    if len(rows) != 6 or {r['seed'] for r in rows} != set(seeds):
        raise ValueError('Incomplete, duplicate, or unregistered cohort')
    ordered = sorted(rows, key=lambda x: seeds.index(x['seed']))
    values = [r[left] - r[right] for r in ordered]
    if any(not math.isfinite(v) for v in values):
        raise ValueError('Nonfinite endpoint')
    mean, sd = statistics.mean(values), statistics.stdev(values)
    margin = T5_975 * sd / math.sqrt(6)
    return dict(n=6, values=values, mean=mean, sd=sd,
                ci95=[mean-margin, mean+margin], positive_families=sum(v > 0 for v in values),
                zero_variance_review=sd == 0)

def first_crossing_bracket(curve, target):
    """No interpolation, best-checkpoint selection, or invented threshold crossing."""
    if not math.isfinite(target):
        raise ValueError('Finite development-fixed target required')
    points = sorted(curve, key=lambda p: p['step'])
    if len({p['step'] for p in points}) != len(points):
        raise ValueError('Duplicate evaluation steps')
    previous = None
    for point in points:
        if any(not math.isfinite(point[k]) for k in ('loss','training_seconds')):
            raise ValueError('Nonfinite curve point')
        if previous and point['training_seconds'] < previous['training_seconds']:
            raise ValueError('Nonmonotone training-time ledger')
        if point['loss'] <= target:
            return dict(reached=True, step_upper=point['step'],
                        step_lower=previous['step'] if previous else 0,
                        seconds_upper=point['training_seconds'],
                        seconds_lower=previous['training_seconds'] if previous else 0.,
                        interpolated=False)
        previous = point
    return dict(reached=False, observed_through=previous['step'] if previous else 0,
                last_training_seconds=previous['training_seconds'] if previous else 0.,
                interpolated=False)
