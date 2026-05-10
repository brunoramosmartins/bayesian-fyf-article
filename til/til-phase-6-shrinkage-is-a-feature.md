# TIL — Shrinkage Is a Feature, Not a Bug

**Phase:** 6 · **Topic:** shrinkage · **Domain:** forecasting

## Hook

The word "shrinkage" sounds like your estimate is being punished. It is
actually being improved.

## Insight

When the posterior mean "shrinks" away from the MLE toward the prior, it
is trading a tiny bit of bias for a large reduction in variance. In
budget terms: the raw average of 3 months of actuals is noisy (high
variance). Blending it with the plan (which is biased but stable)
produces a forecast with **lower mean squared error** than either alone.

This is not Bayesian mysticism. It is the **James–Stein** phenomenon: in
three or more dimensions, the MLE is *inadmissible* — there always
exists an estimator with lower expected squared error than the MLE
uniformly over the parameter space. Shrinkage is that estimator.

## Example

After 3 months of actuals
$\{R\$ 1{,}120K,\;R\$ 1{,}095K,\;R\$ 1{,}080K\}$:

- MLE (sample mean): $\bar x_3 \approx \text{R\$ 1{,}098K}$.
- Plan (prior mean): $\mu_0 = \text{R\$ 1{,}050K}$.
- Posterior mean: $\mu_3 \approx \text{R\$ 1{,}094K}$ — shrunk toward the
  plan by $w_0(3) \approx 9$ %.

If the truth is $\theta_\star = \text{R\$ 1{,}080K}$:

- MLE error: $|1{,}098K - 1{,}080K| = \text{R\$ 18K}$.
- Posterior-mean error: $|1{,}094K - 1{,}080K| = \text{R\$ 14K}$.

The Bayesian estimate is closer to the truth in this realisation, and on
average it dominates in MSE.

## Takeaway

Shrinkage = borrowing strength from the plan when data is sparse. As
data accumulates, the shrinkage vanishes ($w_0(n) \to 0$). Early months:
trust the blend. Late months: trust the data.
