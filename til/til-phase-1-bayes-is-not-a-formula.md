# TIL — Bayes' Theorem Is Not a Formula. It's a Machine.

**Phase:** 1 · **Topic:** Bayesian foundations · **Domain:** budget forecasting

## Hook

Most people learn Bayes' theorem as a formula for flipping conditional
probabilities. That undersells it catastrophically.

## Insight

Bayes' theorem is a *learning machine*. You feed it two inputs — what
you believed before (the prior) and what you observed (the data) — and
it outputs a calibrated, updated belief (the posterior). The formula is
just the mechanism; the power is in the *sequential* application: each
posterior becomes the next prior, and the machine runs again.

In budget terms: the original plan is a prior. January's actuals are
data. The updated forecast is a posterior. Then February arrives, and
the posterior becomes the prior for the next update. By Q3, the
forecast has absorbed nine months of evidence and is dramatically
sharper than the original plan.

## Example

Prior: "We expect to spend R$ 1.05M / month, ±R$ 150K."
After January (actual = R$ 1.12M): posterior shifts to ≈ R$ 1.10M,
±R$ 71K.
The uncertainty was cut in half — not because we got lucky, but because
that is what the precision-additivity identity guarantees.

## Takeaway

If you are revising a forecast based on new data, you are doing
Bayesian updating — whether you call it that or not. The theorem just
makes it precise, optimal, and auditable.
