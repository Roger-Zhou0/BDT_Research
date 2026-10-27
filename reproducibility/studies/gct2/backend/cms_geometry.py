"""Proposal-space correspondence control; never permutes the trained model.

The ordinary BDT proposal remains the frozen FP32 result. Matching geometry
is computed in FP64 in a single canonical chart, then cast to proposal dtype.
"""
from __future__ import annotations
import hashlib
import math
import torch
import cms_protocol as p


class DegenerateDirection(RuntimeError):
    """The registered control is undefined; stop rather than invent a fallback."""


def make_permutations(roles, hidden, seed):
    text = f'{p.VERSION}:hidden-permutations:{int(seed)}'
    value = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little') % (2**63-1)
    gen = torch.Generator(device='cpu').manual_seed(value)
    return torch.stack([torch.randperm(hidden, generator=gen) for _ in range(roles)])


def validate_permutations(permutations):
    if permutations.ndim != 2 or permutations.dtype != torch.int64:
        raise ValueError('Expected [roles,hidden] int64 permutations')
    target = torch.arange(permutations.shape[1], device=permutations.device).expand_as(permutations)
    if not torch.equal(permutations.sort(dim=1).values, target):
        raise ValueError('Invalid hidden-unit permutation')


@torch.no_grad()
def transport(stack, permutations, projection, inverse=False):
    if stack.ndim != 3 or stack.shape[0] != permutations.shape[0]:
        raise ValueError('Homologous [roles,rows,columns] projection stack required')
    if projection not in ('gate', 'up', 'down'): raise ValueError(projection)
    idx = permutations.to(stack.device)
    if inverse: idx = torch.argsort(idx, dim=1)
    if projection in ('gate', 'up'):
        if stack.shape[1] != idx.shape[1]: raise ValueError('Gate/up hidden size mismatch')
        return torch.gather(stack, 1, idx[:, :, None].expand(-1, -1, stack.shape[2]))
    if stack.shape[2] != idx.shape[1]: raise ValueError('Down hidden size mismatch')
    return torch.gather(stack, 2, idx[:, None, :].expand(-1, stack.shape[1], -1))


def _norm(x):
    return torch.linalg.vector_norm(x)


def _angle(a, b):
    """Stable angle without subtracting near-one squared cosines."""
    an = _norm(a); bn = _norm(b)
    if float(an) == 0 or float(bn) == 0: return 0.
    along = (a*b).sum()/an
    perp = b - (along/an)*a
    return math.atan2(float(_norm(perp)), float(along))


@torch.no_grad()
def match_direction(raw, canonical, scrambled_mapped, *, tolerances=None):
    """Return norm/angle-matched scrambled direction and auditable invariants.

    Each input is a signed, data-only projection proposal at THIS branch state.
    There is no reference-trajectory norm/angle and no validation feedback.
    """
    tol = p.GEOMETRY if tolerances is None else tolerances
    if raw.shape != canonical.shape or raw.shape != scrambled_mapped.shape or raw.ndim < 2:
        raise ValueError('Mismatched proposal geometry')
    if raw.dtype not in (torch.float32, torch.float64): raise TypeError('Use FP32/FP64 proposals')
    if any(x.device != raw.device or x.dtype != raw.dtype for x in (canonical, scrambled_mapped)):
        raise ValueError('Proposal device/dtype mismatch')
    if not all(bool(torch.isfinite(x).all()) for x in (raw, canonical, scrambled_mapped)):
        raise FloatingPointError('Nonfinite proposal supplied to matching')
    # Frozen warmup and exact-zero cases must be exact identities, not recast reconstructions.
    if torch.equal(raw, canonical):
        return raw.clone(), dict(kind='exact_identity', target_angle=0., actual_angle=0.,
                                 angle_error=0., norm_error=0., common_error=0.,
                                 displacement_error=0., candidate_relative_norm=None,
                                 amplification=None, zero_angle=True)
    R = raw.shape[0]
    u = raw.reshape(R,-1).double()
    v = canonical.reshape(R,-1).double()
    w = scrambled_mapped.reshape(R,-1).double()
    mean = u.mean(0,keepdim=True)
    uc = u-mean; vc = v-v.mean(0,keepdim=True)
    nr = _norm(uc); nt = _norm(vc)
    nr_f, nt_f = float(nr), float(nt)
    if nr_f == 0:
        if nt_f != 0: raise DegenerateDirection('Zero raw-centered proposal has nonzero canonical target')
        return raw.clone(), dict(kind='zero_centered', zero_angle=True, target_angle=0.,actual_angle=0.)
    if nt_f == 0: raise DegenerateDirection('Nonzero raw-centered proposal has zero canonical target')
    canonical_norm_error = abs(nt_f/nr_f-1.)
    if canonical_norm_error > tol['norm_relative_tolerance']:
        raise RuntimeError('Canonical BDT did not preserve centered norm')
    along = (uc*vc).sum()/nr
    perpendicular = vc - (along/nr)*uc
    pn = _norm(perpendicular)
    target_sine = float(pn/nt)
    theta = math.atan2(float(pn), float(along))
    if float(along) <= 0: raise RuntimeError('Canonical bounded BDT rotated beyond 90 degrees')
    if target_sine <= tol['zero_angle_sine']:
        return raw.clone(), dict(kind='numerical_zero', zero_angle=True,
            target_angle=theta, actual_angle=0., angle_error=theta,
            canonical_norm_error=canonical_norm_error,
            candidate_relative_norm=None, amplification=None)
    # Difference-first evaluation reduces cancellation when scrambled BDT is near identity.
    delta = w-u
    dc = delta-delta.mean(0,keepdim=True)
    dc = dc - ((dc*uc).sum()/nr.square())*uc
    # Project again to suppress finite-precision residual common/parallel components.
    dc = dc-dc.mean(0,keepdim=True)
    dc = dc-((dc*uc).sum()/nr.square())*uc
    nd = _norm(dc)
    candidate_rel = float(nd/nr)
    if not math.isfinite(candidate_rel) or candidate_rel <= tol['minimum_candidate_sine']:
        raise DegenerateDirection(
            f'Canonical sine={target_sine:.9g}, scrambled orthogonal/raw norm={candidate_rel:.9g}; '
            'registered direction undefined; no fallback')
    direction = dc/nd
    matched_c = math.cos(theta)*uc + (nr*math.sin(theta))*direction
    out = (mean+matched_c).reshape_as(raw).to(raw.dtype)
    # Verify ACTUAL applied values after casting; mathematical construction alone is insufficient.
    a = out.reshape(R,-1).double(); am = a.mean(0,keepdim=True); ac = a-am
    na = float(_norm(ac)); actual_theta = _angle(uc,ac)
    norm_error = abs(na/nr_f-1.)
    common_error = float(_norm(am-mean))*math.sqrt(R)/max(float(_norm(u)),1e-300)
    angle_error = abs(actual_theta-theta)
    target_displacement = float(_norm(vc-uc)/nr)
    actual_displacement = float(_norm(ac-uc)/nr)
    displacement_error = abs(actual_displacement-target_displacement)
    if norm_error > tol['norm_relative_tolerance'] or common_error > tol['common_relative_tolerance']:
        raise RuntimeError(f'Matched norm/common invariant failed: {norm_error}, {common_error}')
    if angle_error > tol['angle_absolute_tolerance']:
        raise RuntimeError(f'Matched angle invariant failed: {angle_error}')
    allowed = tol['displacement_absolute_tolerance'] + tol['displacement_relative_tolerance']*target_displacement
    if displacement_error > allowed:
        raise RuntimeError(f'Matched displacement invariant failed: {displacement_error} > {allowed}')
    return out, dict(kind='matched', zero_angle=False, target_angle=theta,
        actual_angle=actual_theta, angle_error=angle_error,
        norm_error=norm_error, common_error=common_error,
        canonical_norm_error=canonical_norm_error,
        target_displacement=target_displacement, actual_displacement=actual_displacement,
        displacement_error=displacement_error, candidate_relative_norm=candidate_rel,
        amplification=math.sin(theta)/candidate_rel,
        direction_cosine_with_canonical=float((direction*perpendicular).sum()/pn))
