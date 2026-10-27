# How bounded depth tempering works

BDT transforms the updates AdamW proposes before they are applied to the
weights. It operates on corresponding weight matrices across layers. Weight
decay is applied separately.

## Separate the mean from the differences

Let $`U`$ contain one flattened update per row. With $`L`$ layers and $`P`$
parameters per layer, its shape is $`L \times P`$. The mean update is:

```math
\large \mu = \frac{1}{L}\sum_{\ell=1}^{L} U_{\ell,:}
```

Subtract it from every row to get the differences between layers:

```math
\large C = U - \mathbf{1}\mu
```

Here, $`\mathbf{1}`$ is a column of ones. Multiplying it by $`\mu`$ repeats the
mean across all rows. For three layers with two update coordinates each:

```text
Raw updates U      Shared mean       Differences C
   [3, 1]             [1, 1]             [ 2,  0]
   [1, 2]      =      [1, 1]      +      [ 0,  1]
   [-1, 0]            [1, 1]             [-2, -1]
```

The differences sum to zero across layers. BDT changes these differences while
preserving their total squared magnitude. Each layer's individual update norm
can change.

## Estimate covariance across layers

A fixed orthonormal Helmert basis $`H`$ represents the directions perpendicular
to the vector of ones. In this basis, the centered update is $`Z = H^T C`$,
with $`L-1`$ rows.

At each observation step, normalize the nonzero update:

```math
\large \widehat{Z} = \frac{Z}{\lVert Z\rVert_F}
```

Then update the moving covariance estimate:

```math
\large S_t = \beta S_{t-1} + (1-\beta)\widehat{Z}\widehat{Z}^{T}
```

The subscript $`F`$ denotes the Frobenius norm: the square root of the sum of
squared matrix entries. Normalizing each observation prevents larger updates
from dominating the estimate solely because of their size.

The covariance has shape $`(L-1) \times (L-1)`$. Its size depends on the number
of layers, not the number of parameters. The averaging factor $`\beta`$ applies
per observation; observations need not occur at every optimizer step.

The estimate uses raw AdamW proposals, including during warmup. Tempered updates
are not fed back into the estimate at the same step.

## Scale the covariance directions

Decompose the covariance into eigenvectors $`V`$ and eigenvalues $`\lambda`$:

```math
\large S = V\mathrm{diag}(\lambda)V^T
```

Here, $`\mathrm{diag}(\lambda)`$ is the diagonal matrix of eigenvalues. For a
condition cap $`\kappa \geq 1`$ and ramp strength $`a \in [0,1]`$, compute:

```math
\large g_i = \left(1 + (\kappa^2-1)a\frac{\lambda_i}{\lambda_{\max}}\right)^{-1/2}
```

Directions with larger eigenvalues receive smaller gains. The ratio between
the largest and smallest gain is bounded by $`\kappa`$. This bound applies to
the operator, not to each layer's individual update magnitude.

## Reconstruct the update

**Apply the gains.** Transform the centered update in the covariance basis,
then return it to layer coordinates:

```math
\large \widetilde{C} = H V\mathrm{diag}(g)V^T Z
```

**Restore the norm.** Rescale the result to match the original centered norm:

```math
\large C' = \widetilde{C}\,\frac{\lVert C\rVert_F}{\lVert\widetilde{C}\rVert_F}
```

**Add the mean back.** Combine the new differences with the unchanged mean:

```math
\large U' = \mathbf{1}\mu + C'
```

The implementation also recenters the transformed differences to reduce
numerical drift. It returns the original proposal during warmup, when the
centered update is zero, or when no nonzero covariance estimate is available.

After normalization, some components can be larger than before. Tempering
changes their relative sizes while keeping the total centered norm fixed.

## What stays fixed

| Preserved within numerical precision | Can change |
| --- | --- |
| Mean update across layers | Each layer's update direction |
| Total centered Frobenius norm | Each layer's update norm |
| Total proposal Frobenius norm | Energy in each direction across layers |

The last property follows because the mean and centered parts are orthogonal.
These properties apply to the grouped update proposal, before weight decay.

Adam's moment estimates stay in parameter coordinates. Each projection group
has its own tempering state. Resuming training requires restoring that state
along with the optimizer. The reference implementation checks that step numbers
and configurations agree.

## Why the pairing matters

The column pairing in $`U`$ determines the inner products used to estimate
covariance. Permuting coordinates independently in each layer generally changes
those inner products, even when each layer's norm stays fixed. Applying the
same permutation to every layer preserves them.

The correspondence experiments change the pairing, compute an alternative
proposal, and map it back into the model's coordinates. The CMS1 control also
matches the centered norm and angle to the raw proposal. See the
[policy](../reproducibility/studies/cms1/correspondence_matched_s1_v1/cms_policy.py)
and [matching geometry](../reproducibility/studies/cms1/correspondence_matched_s1_v1/cms_geometry.py)
for the matching procedure and its handling of degenerate cases.

Copied pretrained layers offer a pairing with a known origin. Their shared
ancestry makes the comparison easier to define, although the copied features
can take on different roles during training.

## Run the operator

The [reference implementation](../reproducibility/studies/cms1/fair_depth_lr_bracketing_v2/tempering_reference.py)
contains `TemperConfig`, `OnlineDepthTemper.transform`, and methods for saving
and restoring state. The [demo](../examples/depth_update_demo.py) calls it with
six synthetic layers. After installing dependencies, run `make demo` from the
repository root.

The [code guide](CODE_GUIDE.md) shows how the study drivers apply these proposals
to model weights.
