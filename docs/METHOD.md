# How bounded depth tempering works

BDT acts between an optimizer proposing an update and that update being applied
to the weights. Its input is a group of corresponding, signed AdamW data-update
proposals, before separate decoupled weight decay.

## Shared movement and differences

Let $`U`$ be an $`L \times P`$ matrix: $`L`$ layers, with $`P`$ flattened parameter
coordinates per layer. The mean row is $`\mu`$, and the centered updates are $`C`$:

```math
\begin{aligned}
\mu &= \frac{1}{L}\sum_{\ell=1}^{L} U_{\ell,:} \\
C &= U - \mathbf{1}\mu
\end{aligned}
```

For example, take three layers with two update coordinates each:

```text
Raw updates U      Shared mean       Differences C
   [3, 1]             [1, 1]             [ 2,  0]
   [1, 2]      =      [1, 1]      +      [ 0,  1]
   [-1, 0]            [1, 1]             [-2, -1]
```

The mean describes the motion all layers share. The differences sum to zero
across layers. BDT changes the pattern in those differences while preserving
their total squared magnitude. It does not require every layer's individual
update norm to stay fixed.

## A small covariance across layers

The implementation uses a fixed orthonormal Helmert basis $`H`$ for the subspace
orthogonal to the all-ones vector. It represents the centered proposal as
$`Z = H^T C`$, with $`L-1`$ rows. At an observation step with nonzero $`Z`$, it updates:

```math
\begin{aligned}
\widehat{Z} &= \frac{Z}{\lVert Z\rVert_F} \\
S_t &= \beta S_{t-1} + (1-\beta)\widehat{Z}\widehat{Z}^{T}
\end{aligned}
```

This covariance is $`(L-1) \times (L-1)`$, rather than a covariance over all $`P`$
parameters. Normalizing each observation separates its cross-layer pattern from
its overall magnitude. The averaging factor $`\beta`$ applies per observation,
which need not occur at every optimizer step.

The estimator observes raw proposals before tempering, including during warmup.
Thus it tracks what AdamW proposes rather than feeding transformed proposals
back into the covariance estimate at the same step.

## Bounded directional scaling

Write $`S = V\mathrm{diag}(\lambda)V^T`$. For condition cap $`\kappa \geq 1`$
and ramp strength $`a \in [0,1]`$, the reference operator uses gains

```math
g_i = \left(1 + (\kappa^2-1)a\frac{\lambda_i}{\lambda_{\max}}\right)^{-1/2}
```

Larger-eigenvalue directions receive smaller gains. Their relative scaling is
bounded by $`\kappa`$. This bounds the operator's gain ratio, not each layer's
individual update ratio or the final training effect.

The transformed differences and final proposal are

```math
\begin{aligned}
\widetilde{C} &= H V\mathrm{diag}(g)V^T Z \\
C' &= \widetilde{C}\frac{\lVert C\rVert_F}{\lVert\widetilde{C}\rVert_F} \\
U' &= \mathbf{1}\mu + C'
\end{aligned}
```

The implementation recenters the transformed differences to control numerical
drift. During warmup, for zero centered updates, or before a nonzero covariance
estimate is available, it returns an unchanged copy of the proposal.

After rescaling, “attenuation” is relative: some components can grow compared
with their original magnitude. The final centered norm stays fixed.

## What is preserved—and what can change

| Preserved, up to floating-point error | Allowed to change |
| --- | --- |
| Mean update across layers | Each layer's update direction |
| Total centered Frobenius norm | Each layer's update norm |
| Total proposal Frobenius norm, by orthogonality of mean and centered parts | Distribution of energy among depthwise directions |

These statements concern the grouped data-update proposal. They are not claims
about the norm of the weights after an optimizer step including weight decay.

Adam's moment estimates remain in parameter coordinates. Each corresponding
projection group gets its own tempering state. Resuming training requires
restoring that state alongside the optimizer; the reference implementation
checks sequential step numbers and configuration compatibility.

## Why matching coordinates is an assumption

The column pairing in $`U`$ determines the inner products used to estimate
cross-layer covariance. Independently permuting feature coordinates in each
layer generally changes those statistics, even when each layer's own norm is
unchanged. Applying the same permutation to every layer would preserve those
inner products.

The correspondence experiments exploit this distinction. They alter how
coordinates are paired across layers, compute the alternative proposal, and
map it back into the model's original coordinates. The CMS1 matched control
additionally matches geometric properties such as centered norm and angle to
the raw proposal. Its implementation includes explicit handling of degenerate
cases; see [the policy](../reproducibility/studies/cms1/correspondence_matched_s1_v1/cms_policy.py)
and [matching geometry](../reproducibility/studies/cms1/correspondence_matched_s1_v1/cms_geometry.py).

Copying pretrained layers provides another way to investigate correspondence:
parameters have a known common origin. That ancestry is useful experimental
structure, but it does not guarantee copied features retain the same role
throughout subsequent training.

## Reading and running the actual operator

The [reference implementation](../reproducibility/studies/cms1/fair_depth_lr_bracketing_v2/tempering_reference.py)
contains `TemperConfig`, `OnlineDepthTemper.transform`, and state save/load
methods. The [demo](../examples/depth_update_demo.py) calls this implementation
directly with six synthetic layers. Run it with `make demo` from the repository
root after installing dependencies.

Continue with the [code guide](CODE_GUIDE.md) to see how those proposals become
weight updates in the study drivers.
