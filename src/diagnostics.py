"""Diagnostic layer for the Bayesian FYF model.

Operates on the trajectory produced by :class:`src.fyf_model.FYFModel`
(or any equivalent sequence of Normal-Normal posteriors plus the
matching observations). Provides the three diagnostics from
``notes/phase5-fyf-model.md`` §3:

- :func:`surprise_score` — predictive z-score for one observation.
- :func:`surprise_trajectory` — z-scores across an entire history.
- :func:`posterior_predictive_pvalues` — two-sided predictive p-values.
- :func:`calibration_score` — fraction of actuals inside a predictive
  interval.
- :func:`calibration_binomial_test` — binomial test of calibration vs
  the nominal level.
- :func:`cumulative_surprise` — running sum of standardised
  innovations.

The diagnostics are deliberately model-agnostic: they accept ordinary
``NormalPosterior`` objects (the inference layer's frozen dataclass)
plus the next observation, so they can be reused outside the FYF
context — e.g. with a plain SequentialUpdater.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from src.conjugate import NormalPosterior

# =============================================================================
# Surprise z-score
# =============================================================================


def surprise_score(
    prior_posterior: NormalPosterior, actual: float, sigma_sq: float
) -> float:
    """Predictive z-score :math:`z = (x - \\mu_n)/\\sqrt{\\sigma_n^2 + \\sigma^2}`.

    The score uses the posterior *before* the observation is consumed;
    thus it is a one-step-ahead prediction error on the predictive
    scale.
    """
    if sigma_sq <= 0:
        raise ValueError(f"sigma_sq must be > 0; got {sigma_sq}")
    pred_var = prior_posterior.variance() + sigma_sq
    return float((actual - prior_posterior.mean()) / math.sqrt(pred_var))


def surprise_trajectory(
    posteriors: list[NormalPosterior],
    actuals: list[float],
    sigma_sq: float,
) -> NDArray[np.float64]:
    """Vector of surprise z-scores for a full history.

    ``posteriors[t-1]`` is the posterior **before** observation ``t``
    (i.e. the prior posterior of step ``t``). For the first step pass
    the prior in the leading position.
    """
    if len(posteriors) != len(actuals):
        raise ValueError(
            "posteriors and actuals must have the same length; "
            f"got {len(posteriors)} and {len(actuals)}."
        )
    return np.array(
        [
            surprise_score(p, x, sigma_sq)
            for p, x in zip(posteriors, actuals, strict=True)
        ]
    )


# =============================================================================
# Posterior predictive p-values
# =============================================================================


def posterior_predictive_pvalues(
    posteriors: list[NormalPosterior],
    actuals: list[float],
    sigma_sq: float,
) -> NDArray[np.float64]:
    """Two-sided predictive p-values for each step.

    For each step :math:`t`,
    :math:`p_t = 2 \\min(\\Phi(z_t), 1 - \\Phi(z_t))`.
    Under correct specification the empirical distribution of
    :math:`\\{p_t\\}` is approximately Uniform ``(0, 1)``.
    """
    z = surprise_trajectory(posteriors, actuals, sigma_sq)
    cdf = stats.norm.cdf(z)
    return np.minimum(cdf, 1.0 - cdf) * 2.0


# =============================================================================
# Calibration
# =============================================================================


def calibration_score(
    posteriors: list[NormalPosterior],
    actuals: list[float],
    sigma_sq: float,
    level: float = 0.95,
) -> float:
    """Fraction of actuals inside the ``level`` predictive interval.

    A perfectly calibrated model returns ``level`` here in expectation.
    """
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1); got {level}")
    z = surprise_trajectory(posteriors, actuals, sigma_sq)
    z_crit = float(stats.norm.ppf(0.5 + level / 2.0))
    inside = np.abs(z) <= z_crit
    return float(inside.mean())


@dataclass(frozen=True)
class CalibrationTest:
    """Result of a binomial test of predictive-interval coverage."""

    n_observed: int
    n_inside: int
    nominal_coverage: float
    empirical_coverage: float
    p_value_two_sided: float

    def is_calibrated(self, alpha: float = 0.05) -> bool:
        return self.p_value_two_sided > alpha


def calibration_binomial_test(
    posteriors: list[NormalPosterior],
    actuals: list[float],
    sigma_sq: float,
    level: float = 0.95,
) -> CalibrationTest:
    """Two-sided binomial test of empirical vs nominal coverage."""
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1); got {level}")
    z = surprise_trajectory(posteriors, actuals, sigma_sq)
    z_crit = float(stats.norm.ppf(0.5 + level / 2.0))
    inside = (np.abs(z) <= z_crit).astype(int)
    n = int(inside.size)
    k = int(inside.sum())

    res = stats.binomtest(k, n, p=level, alternative="two-sided")
    return CalibrationTest(
        n_observed=n,
        n_inside=k,
        nominal_coverage=float(level),
        empirical_coverage=k / n if n > 0 else float("nan"),
        p_value_two_sided=float(res.pvalue),
    )


# =============================================================================
# Cumulative surprise
# =============================================================================


def cumulative_surprise(
    posteriors: list[NormalPosterior],
    actuals: list[float],
    sigma_sq: float,
) -> NDArray[np.float64]:
    """Running cumulative sum of surprise z-scores.

    Under correct specification this is approximately a random walk
    with mean 0 and variance growing linearly in :math:`n`. Persistent
    drift indicates structural mis-specification.
    """
    z = surprise_trajectory(posteriors, actuals, sigma_sq)
    return np.cumsum(z)


# =============================================================================
# Convenience: extract priors-before-step from an FYFModel
# =============================================================================


def priors_before_each_step(
    model_reviews: list, prior: NormalPosterior
) -> list[NormalPosterior]:
    """Reconstruct the list of posteriors-as-priors for a model's history.

    For step 1, the prior is the engine's initial prior. For step
    :math:`t > 1`, the prior is the posterior produced at step
    :math:`t - 1`. Pass the model's ``reviews()`` and the prior used at
    construction.
    """
    priors: list[NormalPosterior] = [prior]
    for review in model_reviews[:-1]:
        post = review.posterior
        if not isinstance(post, NormalPosterior):
            raise TypeError(
                f"Expected NormalPosterior; got {type(post).__name__}."
            )
        priors.append(post)
    return priors


__all__ = [
    "CalibrationTest",
    "calibration_binomial_test",
    "calibration_score",
    "cumulative_surprise",
    "posterior_predictive_pvalues",
    "priors_before_each_step",
    "surprise_score",
    "surprise_trajectory",
]
