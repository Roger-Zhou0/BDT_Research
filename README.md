# Bounded Depth Tempering

**Using relationships between Transformer layers to shape optimizer updates.**

A Transformer repeats similar building blocks at different depths. Those blocks
have corresponding weight matrices, but AdamW computes each parameter's update
from its own gradient history. This project studies whether relationships
between layers can help determine those updates.

**Bounded depth tempering (BDT)** takes the updates AdamW is about to apply
and uses their covariance across layers to change how they differ across depth. It preserves the average update
and the total size of the differences. The model's forward architecture stays
the same.

This project contains an animated explanation, the implementation, and experiments that
ask **whether the correspondence between parameters matters**, beyond
changing the size of an optimizer step.

[Watch the animation](#see-the-updates-change) · [How it works](#how-it-works) · [The experiments](#experiments) · [Inside the code](docs/CODE_GUIDE.md)

## See the updates change

![Animation of six layer updates: separate the shared mean, temper the differences, and recombine. The original arrows remain visible, while the mean and total difference norm stay fixed.](docs/assets/bdt-updates.gif)

Each panel is one layer, with two parameter coordinates. Orange shows the shared
mean; blue shows the update's difference from that mean. The dashed arrow keeps
the original update visible for comparison. Follow the blue tips as BDT changes
the updates, and the two measurements below as they stay fixed.

<details>
<summary>View the still image</summary>

![Final BDT updates compared with their original arrows. The shared mean is [0.55, 0.35] and the total difference norm remains 3.447.](docs/assets/bdt-updates-static.png)

</details>

The animation uses the actual reference operator on synthetic updates. The same
input is repeated while tempering ramps up, so you can see the operator's effect
without changes from a training run. It illustrates the geometry, not a change
in model quality. [Animation source](examples/animate_depth_updates.py) ·
[Run the numerical demo](#try-it)

## The intuition

Consider six layers, each proposing an update to the same kind of feedforward
weight matrix. Their updates can be split into a shared mean and a difference
from that mean for each layer.

BDT keeps the shared movement. It tracks recurring patterns in the differences
and reduces the relative influence of directions with more update energy.
Afterward, it restores the total size of those differences. The result changes
how layers move relative to one another while preserving the total update norm.

A direction has more energy when it accounts for more squared update magnitude.
That alone does not make it harmful. The experiments test whether changing the
relative sizes of these directions helps learning.

## How it works

For one group of corresponding projections:

1. **Collect the proposals.** Flatten each layer's proposed AdamW update into a
   row of a matrix. Each column represents a parameter coordinate matched across
   layers.
2. **Separate shared movement from differences.** Subtract the mean row. Keep
   that mean unchanged.
3. **Track patterns across depth.** Maintain an exponential moving average of
   normalized, centered update covariance. This describes which combinations of
   layers tend to move together or apart.
4. **Temper the differences.** Attenuate directions with larger covariance
   eigenvalues. A condition bound limits the relative scaling, and a warmup and
   ramp introduce the transformation gradually.
5. **Restore the size and mean.** Rescale the transformed differences to their
   original norm, then add back the shared update.

```mermaid
flowchart TD
    A[AdamW update proposals] --> B[Group corresponding layers]
    B --> C[Shared mean]
    B --> D[Differences across depth]
    D --> E[Estimate covariance<br/>and temper]
    E --> F[Restore difference norm]
    C --> G[Recombine and apply]
    F --> G
```

The mean update and the total centered norm are preserved, up to numerical
precision. Individual layers can receive different update magnitudes and
directions. Adam's moment estimates remain in their original parameter
coordinates, and decoupled weight decay is handled separately.

See [the method in detail](docs/METHOD.md) for the equations, correspondence
assumptions, and a small numerical example.

## Why correspondence matters

The same position in two weight matrices is an obvious way to pair parameters.
It is not proof that those parameters have learned the same function. Hidden
features can be represented in different orders, and layers serve different
roles at different depths.

BDT's covariance estimate depends on which coordinates are paired. The
experiments compare three choices:

- **Use the original pairing:** compare matching coordinates across layers.
- **Scramble the pairing:** change which coordinates are compared, then map the
  proposed update back. Matched controls also constrain the intervention's
  geometry to separate the effect of pairing from the size of the change.
- **Use known ancestry:** copy pretrained layers, giving parameters a known
  origin, then compare preserved and reassigned correspondence.

These comparisons test whether the original pairing helps when the size of the
intervention is controlled.

## Try it

Use Python 3.13 on macOS or Linux. From the repository root:

```bash
python3.13 -m venv .venv
make install
make demo
```

The [demo](examples/depth_update_demo.py) calls the BDT reference
operator with synthetic updates from six layers. It prints the transition from
warmup to active tempering, the change in update direction, and checks on the
preserved mean and norm. It runs on CPU without downloading a model or dataset.

The short warmup makes the mechanism visible in six steps. These are synthetic
updates, so the demo illustrates the transformation rather than measuring
language model quality.

## Experiments

The experiments separate three questions: which parameters are paired, how their
updates are transformed, and how update magnitude is distributed.

| Question | Experiment | Where to explore |
| --- | --- | --- |
| Does the pairing matter when the intervention's geometry is matched? | CMS1 compares AdamW, BDT, and a matched control with scrambled correspondence. | [Optimizer policy](reproducibility/studies/cms1/correspondence_matched_s1_v1/cms_policy.py) |
| What happens when a pretrained model grows by copying layers? | GCT2 studies continuation after stacked copying; FEV1 evaluates the same trained models on another partition. | [Growth study](reproducibility/studies/gct2/README.md) |
| Does preserving known ancestry matter? | BAC1 compares correspondence choices in a model with copied pretrained layers. | [Ancestry study](reproducibility/studies/bac1/README.md) |
| How much comes from distributing update magnitude? | BCT1 and BCT2 compare budget choices with a shared direction construction; BCT2 uses four fresh parents. | [Evidence coverage](reproducibility/docs/COVERAGE.md), [BCT2 records](reproducibility/docs/BCT2_RECORDS.md) |

Saved results include improvements in the studied settings. Their interpretation
depends on the control, training setup, and independent parent models. They do
not establish that BDT is universally better than AdamW or faster in elapsed training
time. Transforming the full update matrices still has a computational cost, even
though the covariance matrix is small.

You can inspect and recompute the saved comparisons with `make analysis`.
Outputs go into a new `reproducibility/build/all_records/` directory. The
[coverage guide](reproducibility/docs/COVERAGE.md) records which results have
original receipts, compact score records, or values transcribed at their displayed precision.

## Implementation

| Start with | What you will find |
| --- | --- |
| [Animation](examples/animate_depth_updates.py) | Source for the animation and its numerical checks |
| [Demo](examples/depth_update_demo.py) | A short, executable introduction to the transformation |
| [Method guide](docs/METHOD.md) | Equations and the choices behind the operator |
| [Code guide](docs/CODE_GUIDE.md) | A path from the operator to the optimizer and training drivers |
| [Reference operator](reproducibility/studies/cms1/fair_depth_lr_bracketing_v2/tempering_reference.py) | Covariance tracking, bounded scaling, and state handling |
| [Research release](reproducibility/README.md) | Study protocols, saved evidence, and reproduction commands |

Original source files and experiment records are stored under
`reproducibility/`. Their paths and hashes are retained so the code can be
checked against the recorded experiments.

## Check the project

```bash
make verify   # Source hashes and saved BCT2 receipts; standard library only
make test     # CPU numerical and record handling tests
make check    # Verification, tests, and the demo
```

The repository includes training implementations and saved results. Full training additionally requires external datasets, checkpoints, and compute;
those assets are not bundled. Passing the CPU tests or recomputing recorded
scores is distinct from rerunning the training experiments.

For development, see [Contributing](CONTRIBUTING.md). For source provenance and
reuse terms, see [Source notices](reproducibility/THIRD_PARTY_NOTICES.md).
No blanket open source license is asserted for the preserved research release.
