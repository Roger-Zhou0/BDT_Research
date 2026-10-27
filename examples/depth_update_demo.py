"""See how BDT changes updates across layers while preserving mean and norm.

Input tensors represent signed AdamW data proposals at corresponding depths.
This demonstration does not train a neural network or reproduce paper results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Import original source without copying or modifying the recorded implementation.
REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "reproducibility/studies/cms1/fair_depth_lr_bracketing_v2"
)
sys.path.insert(0, str(REFERENCE))
from tempering_reference import OnlineDepthTemper, TemperConfig  # noqa: E402


def main() -> None:
    generator = torch.Generator().manual_seed(7)
    roles, width = 6, 16
    config = TemperConfig(
        warmup_steps=2,
        ramp_steps=2,
        observation_every=1,
        covariance_beta=0.9,
        max_operator_condition=3.0,
    )
    transform = OnlineDepthTemper(roles=roles, config=config)

    print("BDT: six layers, sixteen update coordinates per layer")
    print("Steps 1–2 observe raw updates; steps 3–4 ramp up tempering.")
    print("Cosine measures direction agreement with the raw centered update.")
    print("Mean error should stay near zero; norm ratio should stay near one.\n")
    print("step  strength  condition  cosine    mean error  norm ratio")

    for step in range(1, 7):
        # Nonuniform layer update scales produce depth-dependent proposal energy.
        update = torch.randn(roles, width, generator=generator)
        update = update * torch.linspace(0.5, 1.8, roles)[:, None]
        updated, stats = transform.transform(update, step)

        # BDT changes relative directions while retaining these two invariants.
        mean_before = update.mean(dim=0)
        mean_after = updated.mean(dim=0)
        common_error = (mean_after - mean_before).abs().max()
        before = torch.linalg.vector_norm(update - mean_before)
        after = torch.linalg.vector_norm(updated - mean_after)
        if not torch.allclose(before, after, atol=1e-5, rtol=1e-5):
            raise AssertionError("Centered update norm changed")
        if float(common_error) > 1e-5:
            raise AssertionError("Mean update changed")
        print(
            f"{step:>4}  {stats['strength']:>8.2f}  "
            f"{stats['operator_condition']:>9.3f}  "
            f"{stats['raw_update_cosine']:>8.5f}  "
            f"{float(common_error):>10.2e}  "
            f"{stats['centered_norm_ratio']:>10.6f}"
        )

    print("\nAll mean and centered-norm checks passed.")
    print("The synthetic example illustrates update geometry, not training quality.")


if __name__ == "__main__":
    main()
