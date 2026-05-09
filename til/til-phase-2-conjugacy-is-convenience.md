# TIL — Conjugacy Is a Convenience, Not a Requirement

**Phase:** 2 · **Topic:** conjugate priors · **Domain:** Bayesian inference

## Hook

Conjugate priors have a reputation as "the easy way out." That
reputation is half-right and half-wrong.

## Insight

A conjugate prior is one where the posterior belongs to the same
family as the prior. Normal prior + Normal likelihood ⟶ Normal
posterior. The math closes in one line. The alternative — a
non-conjugate prior — requires numerical integration or MCMC to
compute the posterior. That is not wrong; it is just harder.

But here is what people miss: for the Normal-Normal case, the
posterior mean is a *precision-weighted average* of the prior mean and
the data mean. **Precision $= 1/\text{variance}$**, so the more
confident source gets more weight. That structure is so clean and
interpretable that it would be worth using even if we had to compute
it numerically.

## Example

Prior precision: $\tau_0 = 1/150{,}000^2 \approx 4.4 \times 10^{-11}$.
Data precision (3 months): $3\tau = 3/80{,}000^2 \approx 4.7 \times 10^{-10}$.
After 3 months the data has roughly $10\times$ more precision than the
prior. The posterior mean is ≈ 91 % data, ≈ 9 % prior. The plan has
been almost entirely overridden — as it should be.

## Takeaway

Conjugacy is not about laziness. It is about getting a formula you can
explain to your CFO in one sentence: "the forecast is a weighted
average of the plan and the actuals, where the weights are how
confident we are in each."
