"""Build three explanatory GIFs and still images for the README.

Run: .venv/bin/python examples/animate_depth_updates.py

The first animation illustrates exact mean decomposition. The second shows the
spectral steps of the reference operator for a fixed observed covariance. The
third illustrates coordinate permutations, not the full CMS1 matched control.
Intermediate fades and tile movements are presentation effects, not training.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproducibility/studies/cms1/fair_depth_lr_bracketing_v2"))
from tempering_reference import OnlineDepthTemper, TemperConfig  # noqa: E402

BG, INK, MUTED = "#f6f7fb", "#192b43", "#53677f"
BLUE, ORANGE, TEAL = "#276eb0", "#bf651d", "#168078"
COLORS = [BLUE, ORANGE, TEAL]
ASSETS = ROOT / "docs/assets"


def canvas(title, subtitle, stage, labels):
    fig = plt.figure(figsize=(11, 6.4), dpi=100, facecolor=BG)
    fig.text(.05, .925, title, fontsize=25, weight="bold", color=INK)
    fig.text(.05, .868, subtitle, fontsize=12, color=MUTED)
    for index, label in enumerate(labels):
        x = .05 + index * .31
        fig.add_artist(FancyBboxPatch(
            (x, .775), .28, .048, boxstyle="round,pad=0.01,rounding_size=0.01",
            transform=fig.transFigure, facecolor=INK if index == stage else "#e7ecf3",
            edgecolor="none", zorder=0,
        ))
        fig.text(x + .012, .79, label, fontsize=11, weight="bold",
                 color="white" if index == stage else MUTED)
    return fig


def finish(fig, caption, note):
    fig.text(.05, .112, caption, fontsize=15, color=INK, weight="bold")
    fig.text(.05, .059, note, fontsize=10.5, color=MUTED)
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)
    return image


def arrow(ax, start, end, color, alpha=1):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=19,
        linewidth=4, color=color, alpha=alpha, shrinkA=0, shrinkB=0,
    ))


def save(name, frames, durations):
    ASSETS.mkdir(parents=True, exist_ok=True)
    # Build a palette from every stage, not only from the final image.
    palette_source = Image.new("RGB", (220, 128 * len(frames)))
    for index, frame in enumerate(frames):
        palette_source.paste(frame.resize((220, 128)), (0, 128 * index))
    palette = palette_source.quantize(colors=160)
    indexed = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    indexed[0].save(
        ASSETS / f"{name}.gif", save_all=True, append_images=indexed[1:],
        duration=durations, loop=0, disposal=2, optimize=True,
    )
    frames[-1].save(ASSETS / f"{name}-static.png", optimize=True)
    with Image.open(ASSETS / f"{name}.gif") as gif:
        assert gif.info["loop"] == 0 and gif.n_frames > 10
        for index in range(gif.n_frames):
            gif.seek(index)
            gif.load()
    print(f"{name}: {len(frames)} frames, {sum(durations) / 1000:.1f} seconds")


def draw_mean(stage, reveal=1):
    raw = np.array([1., 3., 5.])
    mean = raw.mean()
    diff = raw - mean
    assert diff.sum() == 0 and np.allclose(mean + diff, raw)
    subtitles = [
        "One parameter coordinate, viewed across three layers.",
        "The average update is 3. Every layer shares this same component.",
        "Subtract 3 from each update to isolate the differences between layers.",
    ]
    fig = canvas("1. Shared movement and layer differences", subtitles[stage], stage,
                 ["1  Start with the updates", "2  Find their mean", "3  Separate the differences"])
    ax = fig.add_axes([.13, .24, .70, .44])
    ax.set_xlim(-.4, 5.7)
    ax.set_ylim(-.6, 2.6)
    ax.set_xticks(range(6))
    ax.set_yticks([2, 1, 0], ["Layer 1", "Layer 2", "Layer 3"])
    ax.tick_params(length=0, labelsize=12, colors=MUTED, pad=10)
    ax.set_xlabel("Proposed update", fontsize=12, color=MUTED, labelpad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(BG)
    for x in range(6):
        ax.axvline(x, color="#e1e6ee", linewidth=.8, zorder=0)
    if stage:
        ax.axvline(mean, color=ORANGE, linewidth=2, linestyle="--", alpha=reveal)
        ax.text(mean, 2.48, "shared mean = 3", ha="center", color=ORANGE,
                fontsize=12, weight="bold", alpha=reveal)
    for index, value in enumerate(raw):
        y = 2 - index
        arrow(ax, (0, y), (value, y), BLUE, alpha=.2 if stage == 2 else 1)
        ax.plot(value, y, "o", color=BLUE, markersize=7)
        if stage == 2:
            # Parallel tracks separate the two components without hiding their signs.
            arrow(ax, (0, y-.13), (mean, y-.13), ORANGE, reveal)
            if diff[index]:
                arrow(ax, (mean, y+.13), (value, y+.13), TEAL, reveal)
            ax.text((mean+value)/2, y+.26, f"difference {diff[index]:+.0f}",
                    ha="center", color=TEAL, fontsize=11, alpha=reveal)
        ax.text(5.5, y, f"{value:.0f}", va="center", fontsize=17, color=BLUE, weight="bold")
    captions = ["Updates: 1, 3, 5", "Mean: (1 + 3 + 5) / 3 = 3", "1 = 3 − 2       3 = 3 + 0       5 = 3 + 2"]
    note = "BDT keeps the mean and acts on the differences. This decomposition has not changed any update."
    return finish(fig, captions[stage], note)


def mean_animation():
    frames, durations = [draw_mean(0)], [2200]
    for stage in [1, 2]:
        for reveal in np.linspace(.05, 1, 16):
            frames.append(draw_mean(stage, float(reveal)))
            durations.append(65)
        durations[-1] = 3000
    save("bdt-mean", frames, durations)


def spectral_example():
    # Two orthogonal, zero-mean layer patterns with different energies.
    first = torch.tensor([1., -1., 1., -1.], dtype=torch.float64) / 2
    second = torch.tensor([1., 1., -1., -1.], dtype=torch.float64) / 2
    centered = torch.stack([4 * first, second], dim=1)
    raw = centered + torch.tensor([.4, .2], dtype=torch.float64)
    operator = OnlineDepthTemper(4, dtype=torch.float64, config=TemperConfig(
        warmup_steps=1, ramp_steps=1, observation_every=1,
        covariance_beta=.9, max_operator_condition=3,
    ))
    operator.transform(raw, 1)
    expected, _ = operator.transform(raw, 2)
    basis, vectors, values = operator.basis, operator.eigenvectors, operator.eigenvalues
    z = basis.T @ centered
    coefficients = vectors.T @ z
    gains = (1 + 8 * values / values.max()).rsqrt()
    tempered = basis @ (vectors @ (gains[:, None] * coefficients))
    scaled = tempered * centered.norm() / tempered.norm()
    torch.testing.assert_close(scaled + raw.mean(0), expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(scaled.norm(), centered.norm(), atol=1e-12, rtol=0)
    torch.testing.assert_close(expected.mean(0), raw.mean(0), atol=1e-12, rtol=0)
    order = torch.argsort(values, descending=True)[:2]
    energies = (coefficients**2).sum(1)[order].numpy()
    selected_gains = gains[order].numpy()
    np.testing.assert_allclose(energies, [16, 1], atol=1e-12)
    return energies, selected_gains


def draw_tempering(stage, amount, energies, gains):
    # Intermediate frames illustrate continuous gain application, then normalization.
    if stage == 0:
        current = energies.copy()
    elif stage == 1:
        current = energies * (1 + amount * (gains - 1))**2
    else:
        attenuated = energies * gains**2
        scale = np.sqrt(energies.sum() / attenuated.sum())
        current = attenuated * (1 + amount * (scale - 1))**2
    titles = [
        "The covariance identifies patterns of differences across layers.",
        "Give the dominant pattern a smaller gain than the weaker pattern.",
        "Scale both patterns together to restore the original total norm.",
    ]
    fig = canvas("2. Temper the dominant pattern", titles[stage], stage,
                 ["1  Observe the patterns", "2  Apply unequal gains", "3  Restore the total norm"])
    ax = fig.add_axes([.23, .38, .68, .28])
    ax.set_xlim(0, 17)
    ax.set_ylim(-.6, 1.6)
    ax.set_yticks([1, 0], ["Dominant pattern", "Weaker pattern"])
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.tick_params(length=0, labelsize=12, colors=MUTED)
    ax.set_xlabel("Energy = squared update magnitude in this pattern", fontsize=11, color=MUTED, labelpad=12)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for x in [0, 4, 8, 12, 16]:
        ax.axvline(x, color="#e2e7ef", zorder=0)
    for index, (before, now, color) in enumerate(zip(energies, current, [BLUE, TEAL])):
        y = 1-index
        ax.barh(y, before, height=.40, facecolor="none", edgecolor="#a1adbd", linestyle="--", linewidth=1.5)
        ax.barh(y, now, height=.34, color=color)
        ax.text(now+.18, y, f"{now:.2f}", va="center", fontsize=13, color=INK)
    fig.text(.23, .706, "Dashed outlines show the original energy of each pattern.", fontsize=11, color=MUTED)
    norm = np.sqrt(current.sum())
    fig.text(.23, .215, f"Total difference norm:  {norm:.3f}    /    original: {np.sqrt(17):.3f}", fontsize=14, color=INK)
    captions = ["One pattern accounts for 16 of the 17 energy units.",
                "Both shrink, but the dominant pattern shrinks more.",
                "Same total norm. Less concentrated in the dominant pattern."]
    return finish(fig, captions[stage], "Computed from the reference operator on synthetic updates. The shared mean is unchanged.")


def tempering_animation():
    energies, gains = spectral_example()
    frames, durations = [draw_tempering(0, 0, energies, gains)], [2600]
    for stage in [1, 2]:
        for amount in np.linspace(0, 1, 26):
            frames.append(draw_tempering(stage, float(amount), energies, gains))
            durations.append(70)
        durations[-1] = 2800
    save("bdt-tempering", frames, durations)


RAW = np.array([[3., 1., -1.], [2., -.5, -1.5], [4., 1.5, .5]])
PERMUTATIONS = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])


def covariance(matrix):
    centered = matrix - matrix.mean(0)
    return centered @ centered.T / np.sum(centered**2)


def draw_pairing(stage, amount=1):
    subtitles = [
        "Columns determine which coordinates are compared across layers.",
        "Reorder coordinates independently within each layer. Keep every value.",
        "Different pairings produce different covariance, even with the same row norms.",
    ]
    fig = canvas("3. Why the pairing matters", subtitles[stage], stage,
                 ["1  Match coordinates", "2  Change the pairing", "3  Compare the statistics"])
    scrambled = np.take_along_axis(RAW, PERMUTATIONS, axis=1)
    np.testing.assert_allclose(np.linalg.norm(RAW, axis=1), np.linalg.norm(scrambled, axis=1))
    assert not np.allclose(covariance(RAW), covariance(scrambled))
    for panel, left in enumerate([.06, .54]):
        fig.text(left, .705, "Original pairing" if panel == 0 else "Reassigned pairing", fontsize=15, weight="bold", color=INK)
        ax = fig.add_axes([left, .38, .40, .28])
        ax.set_xlim(-.75, 2.65)
        ax.set_ylim(-.55, 2.55)
        ax.axis("off")
        for column in range(3):
            ax.add_patch(Rectangle((column-.42, -.45), .84, 2.9, color="#e7ecf3", zorder=0))
        for row in range(3):
            ax.text(-.58, 2-row, f"L{row+1}", ha="right", va="center", fontsize=11, color=MUTED)
            for original in range(3):
                target = int(np.where(PERMUTATIONS[row] == original)[0][0])
                t = (amount if stage == 1 else float(stage == 2)) if panel else 0
                x = original + t * (target-original)
                y = 2-row
                # Curved travel keeps moving cards distinguishable during a swap.
                y += .23*np.sin(np.pi*t)*(original-1)
                ax.add_patch(FancyBboxPatch((x-.35,y-.27),.70,.54,
                    boxstyle="round,pad=0.015,rounding_size=0.055",facecolor=COLORS[original],edgecolor="white",linewidth=1.5))
                ax.text(x,y,f"{'ABC'[original]}  {RAW[row,original]:g}",ha="center",va="center",fontsize=11,color="white",weight="bold")
        if stage == 2:
            cov = covariance(RAW if panel == 0 else scrambled)
            ca = fig.add_axes([left+.06, .18, .12, .15])
            ca.imshow(cov, cmap="RdBu_r", vmin=-.7,vmax=.7)
            ca.set_xticks([]); ca.set_yticks([])
            for (i,j),value in np.ndenumerate(cov):
                ca.text(j,i,f"{value:.2f}",ha="center",va="center",fontsize=8,color="white" if abs(value)>.4 else INK)
            for spine in ca.spines.values(): spine.set_visible(False)
            fig.text(left+.20,.27,"Covariance",fontsize=12,color=INK)
            fig.text(left+.20,.235,"across layers",fontsize=11,color=MUTED)
        else:
            fig.text(left+.01,.25,"Compare values within each shaded column.",fontsize=11,color=MUTED)
    captions = ["The original columns pair A with A, B with B, and C with C.",
                "No values are removed. Each layer keeps its own update norm.",
                "BDT sees different relationships when the pairing changes."]
    return finish(fig,captions[stage],"Synthetic pairing example. Covariance is centered and normalized. This is not the full matched control.")


def pairing_animation():
    frames,durations=[draw_pairing(0)],[2800]
    for amount in np.linspace(0,1,30):
        frames.append(draw_pairing(1,float(amount)))
        durations.append(80)
    durations[-1]=1800
    frames.append(draw_pairing(2));durations.append(4000)
    save("bdt-correspondence",frames,durations)


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans"})
    mean_animation()
    tempering_animation()
    pairing_animation()
    print("Checked decomposition, reference operator agreement, norm preservation, and permutation invariants.")


if __name__ == "__main__":
    main()
