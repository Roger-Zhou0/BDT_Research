"""Reference transforms for proposed E6; NOT a language-model training runner.

Input: stacked, signed, data-only AdamW proposals [role, ...], prior to
separate decoupled weight decay. Keep all Adam moments in parameter coordinates.
One OnlineDepthTemper instance per homologous FFN projection group.
Computes on the input device with FP32/FP64 proposals; this is a correctness
reference, not a fused implementation or a GPU-memory/performance claim.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any
import torch
from torch.profiler import record_function as _bco_region

def helmert_basis(roles: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Deterministic orthonormal basis of the subspace orthogonal to all-ones."""
    if roles < 2:
        raise ValueError('At least two roles are required')
    q = torch.zeros((roles, roles - 1), device=device, dtype=dtype)
    for j in range(roles - 1):
        denom = math.sqrt((j + 1) * (j + 2))
        q[:j + 1, j] = 1.0 / denom
        q[j + 1, j] = -(j + 1) / denom
    return q

def check_update(update: torch.Tensor) -> None:
    if update.ndim < 2 or update.shape[0] < 2:
        raise ValueError('Expected [roles, ...] with at least two roles')
    if update.dtype not in (torch.float32, torch.float64):
        raise TypeError('Construct the proposal in FP32 or FP64, outside autocast')
    if not bool(torch.isfinite(update).all()):
        raise FloatingPointError('Non-finite Adam proposal')

@torch.no_grad()
def prefix_transform(update: torch.Tensor, task_basis: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
    """E4-like initial task frame, but norm target comes from this own proposal."""
    check_update(update)
    roles = update.shape[0]
    if task_basis.shape != (roles, roles - 1) or spectrum.shape != (roles - 1,):
        raise ValueError('Incorrect initial-frame dimensions')
    if not bool(torch.isfinite(spectrum).all()) or not bool((spectrum > 0).all()):
        raise ValueError('Positive finite spectrum required')
    flat = update.reshape(roles, -1)
    mean = flat.mean(0, keepdim=True)
    centered = flat - mean
    norm = torch.linalg.vector_norm(centered)
    if float(norm) == 0.0:
        return update.clone()
    q = task_basis.to(device=update.device, dtype=update.dtype)
    lam = spectrum.to(device=update.device, dtype=update.dtype)
    z = q @ (lam[:, None] * (q.T @ centered))
    z = z - z.mean(0, keepdim=True)
    nz = torch.linalg.vector_norm(z)
    if not bool(torch.isfinite(nz)) or float(nz) <= 0:
        raise FloatingPointError('Invalid transformed prefix proposal')
    return (mean + z * (norm / nz)).reshape_as(update)

@dataclass(frozen=True)
class TemperConfig:
    warmup_steps: int = 200
    ramp_steps: int = 200
    observation_every: int = 10
    covariance_beta: float = 0.9
    max_operator_condition: float = 2.0

    def validate(self) -> None:
        if min(self.warmup_steps, self.ramp_steps, self.observation_every) < 1:
            raise ValueError('Step counts must be positive')
        if not 0 <= self.covariance_beta < 1:
            raise ValueError('covariance_beta must lie in [0, 1)')
        if not math.isfinite(self.max_operator_condition) or self.max_operator_condition < 1:
            raise ValueError('max_operator_condition must be finite and >= 1')

class OnlineDepthTemper:
    """Normalized raw-update covariance EMA and bounded smooth late transform.

    Observe RAW proposals, including during the prefix, before any transform.
    At t <= 200, return raw unchanged. At t=201..400, ramp to condition <=2.
    For a release control call active=False, which is an exact identity on U.
    Save/load state at a fork *after* its last complete optimizer step.
    """

    def __init__(self, roles: int, device: torch.device | str='cpu', dtype: torch.dtype=torch.float32, config: TemperConfig=TemperConfig()) -> None:
        config.validate()
        if dtype not in (torch.float32, torch.float64):
            raise TypeError('Use FP32/FP64')
        self.config, self.roles = (config, roles)
        self.basis = helmert_basis(roles, device=torch.device(device), dtype=dtype)
        k = roles - 1
        self.covariance = torch.zeros((k, k), device=device, dtype=dtype)
        self.eigenvalues = torch.zeros(k, device=device, dtype=dtype)
        self.eigenvectors = torch.eye(k, device=device, dtype=dtype)
        self.last_step = 0
        self.observations = 0

    @torch.no_grad()
    def transform(self, update: torch.Tensor, step: int, *, active: bool=True) -> tuple[torch.Tensor, dict[str, Any]]:
        with _bco_region('BCO/input_safety'):
            check_update(update)
            if update.shape[0] != self.roles or update.device != self.basis.device or update.dtype != self.basis.dtype:
                raise ValueError('Update must match the state role count, device, and dtype')
            if step != self.last_step + 1:
                raise ValueError(f'Expected step {self.last_step + 1}, got {step}; restore saved state when resuming')
        with _bco_region('BCO/centering_projection'):
            flat = update.reshape(self.roles, -1)
            mean = flat.mean(0, keepdim=True)
            uc = flat - mean
            z = self.basis.T @ uc
            nz = torch.linalg.vector_norm(z)
        with _bco_region('BCO/covariance_refresh'):
            if step % self.config.observation_every == 0 and float(nz) > 0:
                with _bco_region('BCO/covariance_formation_ema'):
                    zn = z / nz
                    observation = zn @ zn.T
                    beta = self.config.covariance_beta
                    self.covariance.mul_(beta).add_(observation, alpha=1 - beta)
                    self.covariance.copy_((self.covariance + self.covariance.T) / 2)
                with _bco_region('BCO/eigendecomposition'):
                    ev, q = torch.linalg.eigh(self.covariance.double())
                    scale = max(float(ev.abs().max()), 1e-30)
                    if float(ev.min()) < -1e-05 * scale:
                        raise FloatingPointError('Materially non-PSD covariance')
                    self.eigenvalues.copy_(ev.clamp_min(0).to(self.basis.dtype))
                    self.eigenvectors.copy_(q.to(self.basis.dtype))
                with _bco_region('BCO/covariance_formation_ema'):
                    self.observations += 1
        with _bco_region('BCO/control_gains'):
            self.last_step = step
            a = max(0.0, min(1.0, (step - self.config.warmup_steps) / self.config.ramp_steps)) if active else 0.0
            report: dict[str, Any] = dict(step=step, strength=a, observations=self.observations)
            lmax = float(self.eigenvalues.max())
            if a == 0 or float(nz) == 0 or lmax == 0:
                report.update(operator_condition=1.0, raw_update_cosine=1.0, centered_norm_ratio=1.0)
                return (update.clone(), report)
            c = self.config.max_operator_condition
            gains = (1.0 + (c * c - 1.0) * a * self.eigenvalues / lmax).rsqrt()
            q = self.eigenvectors
        with _bco_region('BCO/operator_application'):
            transformed = self.basis @ (q @ (gains[:, None] * (q.T @ z)))
            transformed -= transformed.mean(0, keepdim=True)
        with _bco_region('BCO/norm_restoration'):
            raw_norm = torch.linalg.vector_norm(uc)
            transformed_norm = torch.linalg.vector_norm(transformed)
            if not bool(torch.isfinite(transformed_norm)) or float(transformed_norm) <= 0:
                raise FloatingPointError('Invalid tempered proposal')
            transformed *= raw_norm / transformed_norm
            out = (mean + transformed).reshape_as(update)
        with _bco_region('BCO/report_reductions'):
            actual = out.reshape(self.roles, -1)
            actual_c = actual - actual.mean(0, keepdim=True)
            acn = torch.linalg.vector_norm(actual_c)
            cos = (actual_c * uc).sum() / (acn * raw_norm)
            report.update(operator_condition=float(gains.max() / gains.min()), raw_update_cosine=float(cos), centered_norm_ratio=float(acn / raw_norm), common_max_abs_error=float((actual.mean(0) - mean[0]).abs().max()))
        with _bco_region('BCO/control_gains'):
            return (out, report)

    def state_dict(self) -> dict[str, Any]:
        return dict(config=self.config.__dict__.copy(), roles=self.roles, last_step=self.last_step, observations=self.observations, covariance=self.covariance.detach().cpu().clone(), eigenvalues=self.eigenvalues.detach().cpu().clone(), eigenvectors=self.eigenvectors.detach().cpu().clone())

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state['config'] != self.config.__dict__ or state['roles'] != self.roles:
            raise ValueError('Tempering configuration mismatch; do not bypass')
        for field in ('covariance', 'eigenvalues', 'eigenvectors'):
            value = state[field]
            destination = getattr(self, field)
            if value.shape != destination.shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f'Invalid state: {field}')
            destination.copy_(value.to(destination))
        self.last_step, self.observations = (int(state['last_step']), int(state['observations']))
