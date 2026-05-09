# TIL — A Prior Is Just an Opinion, Written in Math

**Phase:** 1 · **Topic:** prior elicitation · **Domain:** budget planning

## Hook

"Where does the prior come from?" is the question that makes
frequentists uncomfortable and Bayesians smile.

## Insight

A prior is your pre-data belief about a parameter, expressed as a
probability distribution. In budgeting it is the most natural thing in
the world: the budget plan IS a prior. "We expect to spend around
R$ 12.6M this year, give or take R$ 1.8M" translates directly into a
Normal prior with $\mu_0 = 12{,}600{,}000$ and
$\sigma_0 = 1{,}094{,}221$ — chosen so that
$\mu_0 \pm 1.645\sigma_0$ covers the stated range at 90 % confidence.

The objection — "but that's subjective!" — misses the point. The
budget plan was always subjective. The Bayesian framework just makes
the subjectivity *explicit and updateable*, instead of hiding it in a
spreadsheet cell.

## Example

Two analysts with different priors: Analyst A is confident
($\sigma_0 = 500{,}000$), Analyst B is uncertain ($\sigma_0 = 2{,}000{,}000$).
After 6 months of the same data their posteriors are nearly identical.
The data overwhelms the prior. That is the beauty: even a "wrong"
prior gets corrected.

## Takeaway

You already have a prior — it is called "the budget." Bayesian
inference just gives you a principled way to update it.
