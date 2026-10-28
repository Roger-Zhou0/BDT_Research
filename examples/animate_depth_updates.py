"""Generate the README animation from the BDT reference operator.

Run from the repository root with:
    .venv/bin/python examples/animate_depth_updates.py

Each arrow is a complete two-coordinate synthetic update, not a projection of
higher-dimensional data. Tempering frames are actual sequential operator calls.
The split animation only fades in the mean and difference decomposition.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bdt-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproducibility/studies/cms1/fair_depth_lr_bracketing_v2"))
from tempering_reference import OnlineDepthTemper, TemperConfig  # noqa: E402

BACKGROUND = "#f5f7fb"
INK = "#18283e"
MUTED = "#596b81"
BLUE = "#2879b7"
ORANGE = "#c86927"
GHOST = "#b3becb"


def proposals() -> tuple[np.ndarray, list[np.ndarray]]:
    """Return raw proposals and verified tempering frames, including warmup."""
    differences = torch.tensor(
        [[2.0, 0.1], [1.2, 0.8], [-1.6, 0.4],
         [-1.0, -0.9], [0.2, -0.7], [-0.8, 0.3]],
        dtype=torch.float64,
    )
    differences -= differences.mean(dim=0)
    raw = differences + torch.tensor([0.55, 0.35], dtype=torch.float64)
    operator = OnlineDepthTemper(
        roles=6,
        dtype=torch.float64,
        config=TemperConfig(
            warmup_steps=1, ramp_steps=40, observation_every=1,
            covariance_beta=0.9, max_operator_condition=3.0,
        ),
    )
    frames = []
    for step in range(1, 42):
        updated, _ = operator.transform(raw, step)
        torch.testing.assert_close(updated.mean(0), raw.mean(0), atol=1e-12, rtol=0)
        torch.testing.assert_close(
            (updated - updated.mean(0)).norm(), differences.norm(),
            atol=1e-12, rtol=0,
        )
        frames.append(updated.numpy().copy())
    assert not np.allclose(frames[-1], raw.numpy())
    return raw.numpy(), frames


def arrow(ax, start, end, color, *, alpha=1.0, width=2.2, dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=12,
        linewidth=width, color=color, alpha=alpha,
        linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0,
    ))


def draw(raw: np.ndarray, updated: np.ndarray, stage: int, reveal: float = 1.0) -> Image.Image:
    """Draw six layer views using identical coordinate scales."""
    mean = raw.mean(0)
    norm_before = np.linalg.norm(raw - mean)
    norm_after = np.linalg.norm(updated - updated.mean(0))
    mean_error = np.max(np.abs(updated.mean(0) - mean))
    fig = plt.figure(figsize=(10, 6.5), dpi=110, facecolor=BACKGROUND)
    fig.text(.045, .928, "What BDT changes", fontsize=24, weight="bold", color=INK)
    subtitles = [
        "Six layers propose six updates. Each arrow has two parameter coordinates.",
        "Each update is a shared mean plus a difference specific to that layer.",
        "BDT changes the differences. Their total norm and shared mean stay fixed.",
    ]
    fig.text(.045, .876, subtitles[stage], fontsize=10.5, color=MUTED)
    for i, label in enumerate(["1  Raw updates", "2  Separate the mean", "3  Temper and recombine"]):
        x = .045 + i * .315
        fig.add_artist(FancyBboxPatch(
            (x, .795), .28, .043, boxstyle="round,pad=0.009,rounding_size=0.009",
            transform=fig.transFigure, facecolor=INK if stage == i else "#e6ebf2",
            edgecolor="none", zorder=0,
        ))
        fig.text(x + .012, .807, label, fontsize=10,
                 color="white" if stage == i else MUTED, weight="bold")
    for layer in range(6):
        x = .045 + (layer % 3) * .315
        y = .475 - (layer // 3) * .25
        fig.add_artist(FancyBboxPatch(
            (x, y), .28, .223, boxstyle="round,pad=0.009,rounding_size=0.012",
            transform=fig.transFigure, facecolor="white", edgecolor="#dee5ee", zorder=0,
        ))
        fig.text(x + .012, y + .19, f"Layer {layer + 1}", fontsize=9, color=MUTED)
        ax = fig.add_axes([x + .012, y + .014, .256, .165])
        ax.set_xlim(-2.0, 3.0)
        ax.set_ylim(-1.25, 1.65)
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0, color="#e7ecf3", linewidth=.8)
        ax.axvline(0, color="#e7ecf3", linewidth=.8)
        ax.plot(0, 0, ".", color=INK, markersize=4)
        ax.axis("off")
        if stage == 0:
            arrow(ax, (0, 0), raw[layer], BLUE)
        else:
            arrow(ax, (0, 0), raw[layer], GHOST, width=1.3, dashed=True)
            arrow(ax, (0, 0), mean, ORANGE, alpha=reveal, width=2.8)
            arrow(ax, mean, updated[layer], BLUE, alpha=reveal)
            ax.plot(*updated[layer], "o", color=BLUE, markersize=4, alpha=reveal)
    fig.text(.045, .735, "BLUE  update / difference", fontsize=9, color=BLUE)
    fig.text(.365, .735, "ORANGE  shared mean", fontsize=9, color=ORANGE)
    fig.text(.68, .735, "DASHED  original update", fontsize=9, color=MUTED)
    fig.text(.045, .167, "Mean across layers", fontsize=9, color=MUTED)
    fig.text(.045, .127, f"[{mean[0]:.2f}, {mean[1]:.2f}]  unchanged", fontsize=13, color=INK)
    fig.text(.405, .167, "Total difference norm", fontsize=9, color=MUTED)
    fig.text(.405, .127, f"{norm_before:.3f}  →  {norm_after:.3f}", fontsize=13, color=INK)
    fig.text(.76, .167, "Numerical check", fontsize=9, color=MUTED)
    fig.text(.76, .127, "Mean error < 10⁻¹²" if mean_error < 1e-12 else "FAILED", fontsize=11, color=INK)
    fig.text(.045, .065, "Actual BDT output on synthetic updates. Each panel uses the same scale.", fontsize=9, color=MUTED)
    fig.text(.045, .035, "The orange and blue arrows add to the applied update. This is not a training result.", fontsize=9, color=MUTED)
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)
    return image


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    raw, outputs = proposals()
    destination = ROOT / "docs/assets"
    destination.mkdir(parents=True, exist_ok=True)
    frames = [draw(raw, raw, 0)]
    durations = [2200]
    for alpha in np.linspace(.1, 1, 10):
        frames.append(draw(raw, raw, 1, float(alpha)))
        durations.append(70)
    durations[-1] = 2200
    for updated in outputs:
        frames.append(draw(raw, updated, 2))
        durations.append(90)
    durations[-1] = 3200
    frames[-1].save(destination / "bdt-updates-static.png", optimize=True)
    # Use one palette throughout so colors remain stable between GIF frames.
    palette = frames[-1].quantize(colors=128)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    indexed[0].save(
        destination / "bdt-updates.gif", save_all=True, append_images=indexed[1:],
        duration=durations, loop=0, optimize=True, disposal=2,
    )
    print(f"Saved {len(frames)} animation frames and a static comparison to {destination}")
    print("All 41 operator outputs preserve mean and centered norm within 1e-12.")


if __name__ == "__main__":
    main()
