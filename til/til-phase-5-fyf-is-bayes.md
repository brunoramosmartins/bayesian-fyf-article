# TIL — Every FYF Is a Bayesian Update (Whether You Know It or Not)

**Phase:** 5 · **Topic:** FYF as Bayesian updating · **Domain:** budget management

## Hook

If you have ever revised a budget forecast based on actuals, you have
done Bayesian inference. You just did not have the formula.

## Insight

The FYF cycle maps perfectly to Bayesian updating:

- **December budget plan** = prior distribution (what we believe
  before data).
- **Q1 actuals** = data (evidence from the first 3 months).
- **April FYF revision** = posterior distribution (updated belief).
- **The April posterior becomes the prior for the Q2 update.**

The magic: the Bayesian framework tells you **exactly** how much
weight to put on the plan vs the actuals. Not 50/50. Not gut feeling.
The weight is the *precision ratio* — and it shifts automatically as
more data arrives.

It also tells you when to *stop trusting* the formula. If the surprise
z-score $z_t = (x_t - \mu_{t-1})/\sqrt{\sigma_{t-1}^2 + \sigma^2}$
clears $\pm 3$, the conjugate update will still produce a number, but
that number is mechanical rather than meaningful. **The model does the
math; the analyst owns the interpretation.**

## Example

Reference scenario, R$ 1.05M monthly plan with R$ 150K confidence,
R$ 80K observation noise:

| Quarter | Data weight | Forecast (R$/month) |
|---------|-------------|---------------------|
| Q1 (3 months)  | 91 % | ≈ 1{,}094K |
| Q2 (6 months)  | 95 % | ≈ 1{,}083K |
| Q3 (9 months)  | 97 % | ≈ 1{,}080K |
| Year-end (12 months) | 98 % | ≈ 1{,}079K |

By Q3 the plan is a ghost. The data has taken over — as it should.

## Takeaway

Stop treating FYF revisions as "corrections to the plan." They are
**learning**. The Bayesian framework makes that learning optimal,
auditable, and — through the surprise diagnostic — falsifiable.
