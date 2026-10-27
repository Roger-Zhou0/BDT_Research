"""A fixed window-smoothed direction carrier, not a standalone GS optimizer."""
from __future__ import annotations
import math
import torch

@torch.no_grad()
def smooth_window(raw: torch.Tensor, phase: int, alpha: float = 0.1,
                  onset: int = 200, ramp: int = 200) -> torch.Tensor:
    """S_alpha U, where S_alpha = I - alpha/2 L_path (endpoint self-neighbors).

    Applied independently to each gate/up/down projection. The strength is
    ramped to the frozen alpha, not selected using loss or observed allocation.
    No claim is made that this necessarily reduces relative difference energy.
    """
    if raw.ndim < 2 or raw.shape[0] != 12 or raw.dtype != torch.float32:
        raise ValueError('BCT1 requires a twelve-row FP32 proposal')
    if not isinstance(phase, int) or phase < 0 or ramp <= 0:
        raise ValueError('Invalid phase or ramp')
    if not math.isfinite(alpha) or not 0 <= alpha < 0.5:
        raise ValueError('BCT1 uses an invertible, positive window operator')
    if not bool(torch.isfinite(raw).all()):
        raise FloatingPointError('Nonfinite GS carrier input')
    amount = alpha * min(1.0, max(0.0, (phase - onset) / ramp))
    if amount == 0 or torch.equal(raw, raw[0:1].expand_as(raw)):
        return raw.clone()
    left = torch.cat((raw[0:1], raw[:-1]), dim=0)
    right = torch.cat((raw[1:], raw[-1:]), dim=0)
    return (1.0 - amount) * raw + (amount / 2.0) * (left + right)
