"""Frozen GWP1 schedules; post-growth phase, never global Adam age."""
from __future__ import annotations

MODES = ('adamw', 'bdt', 'early_bdt', 'late_bdt')
HORIZON = 15000
PREFIX = 200
RAMP = 200
WINDOWS = {'adamw': None, 'bdt': (201, 15000),
           'early_bdt': (201, 3000), 'late_bdt': (6201, 9000)}

def strength(mode: str, phase: int) -> float:
    if mode not in MODES or not isinstance(phase, int) or phase < 1:
        raise ValueError('Invalid schedule or post-growth phase')
    window = WINDOWS[mode]
    if window is None or not window[0] <= phase <= window[1]:
        return 0.0
    return min(1.0, (phase - window[0] + 1) / RAMP)

def collect_history(mode: str, phase: int) -> bool:
    if mode not in MODES or phase < 1:
        raise ValueError('Invalid history request')
    # The shared prefix collects canonical history identically for all branches.
    # The late window uses history from phase one; this cost is counted.
    return phase <= PREFIX or (mode != 'adamw' and phase <= WINDOWS[mode][1])

def last_history_step(mode: str, phase: int) -> int:
    if mode not in MODES or phase < 0:
        raise ValueError('Invalid history state')
    return min(phase, PREFIX if mode == 'adamw' else WINDOWS[mode][1])
