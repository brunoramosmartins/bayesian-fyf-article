"""Prior specification helpers.

Translates plain-language budget-planning statements into mathematical
prior distributions for the four conjugate families used in the article:

- Normal prior on a mean (cost, salary, etc.) — `normal_prior_from_budget`.
- Gamma prior on a rate (incidents per month) — `gamma_prior_from_rate`.
- Beta prior on a proportion (overtime fraction) — `beta_prior_from_proportion`.

Each helper returns a `Prior` dataclass; `prior_summary` produces a
dictionary with mean, variance, standard deviation and a central
credible interval suitable for printing or logging.

The module is deliberately library-light: it depends only on
``scipy.stats`` for the inverse-CDFs needed to translate confidence
statements into hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import stats

PriorFamily = Literal["normal", "gamma", "beta"]


@dataclass(frozen=True)
class Prior:
    """A specified prior distribution.

    Attributes
    ----------
    family
        The distribution family. One of ``"normal"``, ``"gamma"``, ``"beta"``.
    params
        A mapping of hyperparameter names to values. Keys depend on the
        family:

        - ``"normal"``: ``{"mu": float, "sigma": float}``
        - ``"gamma"``: ``{"alpha": float, "beta": float}`` (rate parameterisation)
        - ``"beta"``:  ``{"alpha": float, "beta": float}``

    notes
        Free-form context describing how the prior was elicited. Useful
        for audit trails and reproducibility.
    """

    family: PriorFamily
    params: dict[str, float]
    notes: str = field(default="")

    def frozen(self) -> stats.rv_continuous:
        """Return the corresponding ``scipy.stats`` frozen distribution."""
        if self.family == "normal":
            return stats.norm(loc=self.params["mu"], scale=self.params["sigma"])
        if self.family == "gamma":
            # scipy uses scale = 1 / rate
            return stats.gamma(
                a=self.params["alpha"], scale=1.0 / self.params["beta"]
            )
        if self.family == "beta":
            return stats.beta(a=self.params["alpha"], b=self.params["beta"])
        raise ValueError(f"Unknown family: {self.family!r}")


def normal_prior_from_budget(
    plan_value: float,
    confidence_pct: float,
    interval_width: float,
) -> Prior:
    r"""Translate a budget plan into a Normal prior on the unknown mean.

    The planner states a target value :math:`\mu_0` and a *symmetric
    confidence band*: "I am ``confidence_pct`` confident that the truth
    lies within ``±interval_width × \mu_0`` of the plan." The width is
    expressed as a fraction of the plan (e.g. ``0.15`` for ±15 %).

    Solves the equation

    .. math::

        \Pr\!\big(\theta \in [\mu_0(1-w),\, \mu_0(1+w)]\big) = \gamma,

    where :math:`w` is ``interval_width`` and :math:`\gamma` is
    ``confidence_pct``, for the standard deviation
    :math:`\sigma_0 = w\,\mu_0 / z_{(1+\gamma)/2}`.

    Parameters
    ----------
    plan_value
        The point estimate from the budget plan, :math:`\mu_0`. Must be
        finite and non-zero (the prior is symmetric around it).
    confidence_pct
        The confidence level :math:`\gamma \in (0, 1)`. Use ``0.90`` for
        a 90 % statement.
    interval_width
        The half-width as a fraction of ``plan_value``. Must be positive.
        E.g. ``0.15`` means "the truth is within ±15 %".

    Returns
    -------
    Prior
        Normal prior with hyperparameters ``mu`` and ``sigma``.

    Raises
    ------
    ValueError
        If ``confidence_pct`` is not in ``(0, 1)`` or ``interval_width``
        is not positive.

    Examples
    --------
    A budget plan of R$ 1,050,000 / month, planner is 90 % confident the
    truth is within ±15 %:

    >>> p = normal_prior_from_budget(1_050_000, 0.90, 0.15)
    >>> round(p.params["sigma"], 0)
    95753.0
    """
    if not 0 < confidence_pct < 1:
        raise ValueError(
            f"confidence_pct must be in (0, 1); got {confidence_pct}"
        )
    if interval_width <= 0:
        raise ValueError(
            f"interval_width must be positive; got {interval_width}"
        )
    if plan_value == 0:
        raise ValueError("plan_value must be non-zero for a relative band.")

    z = stats.norm.ppf(0.5 + confidence_pct / 2.0)
    sigma = abs(interval_width * plan_value) / z
    return Prior(
        family="normal",
        params={"mu": float(plan_value), "sigma": float(sigma)},
        notes=(
            f"Elicited from plan={plan_value} with "
            f"{confidence_pct:.0%} confidence in ±{interval_width:.0%}."
        ),
    )


def gamma_prior_from_rate(
    expected_rate: float,
    confidence: float,
) -> Prior:
    r"""Build a Gamma prior on a positive rate from an expected value
    and a "confidence" pseudo-sample-size.

    Uses the *pseudo-observation* parameterisation: the planner's belief
    is encoded as if it had been formed from ``confidence`` prior
    observations whose mean rate was ``expected_rate``. Concretely

    .. math::

        \alpha_0 = \text{confidence} \cdot \text{expected\_rate},
        \qquad
        \beta_0 = \text{confidence}.

    Then :math:`\mathbb E[\lambda] = \alpha_0 / \beta_0 =
    \text{expected\_rate}` and the prior variance scales as
    :math:`\text{expected\_rate} / \text{confidence}`. Larger
    ``confidence`` produces a tighter prior.

    Parameters
    ----------
    expected_rate
        Prior expected rate, e.g. expected incidents per month. Must be
        positive.
    confidence
        A pseudo-sample-size: the number of months of "imaginary prior
        data" backing the belief. Must be positive. ``1.0`` is a weakly
        informative prior; ``10.0`` is a strongly informative one.

    Returns
    -------
    Prior
        Gamma prior with hyperparameters ``alpha`` and ``beta`` (rate).

    Raises
    ------
    ValueError
        If either argument is not strictly positive.

    Examples
    --------
    Expected 3 incidents/month with the equivalent of one month of
    prior data:

    >>> p = gamma_prior_from_rate(expected_rate=3.0, confidence=1.0)
    >>> p.params
    {'alpha': 3.0, 'beta': 1.0}
    """
    if expected_rate <= 0:
        raise ValueError(
            f"expected_rate must be positive; got {expected_rate}"
        )
    if confidence <= 0:
        raise ValueError(f"confidence must be positive; got {confidence}")

    alpha = float(confidence * expected_rate)
    beta = float(confidence)
    return Prior(
        family="gamma",
        params={"alpha": alpha, "beta": beta},
        notes=(
            f"Elicited as Gamma(alpha={alpha}, beta={beta}) with "
            f"expected_rate={expected_rate} and pseudo-N={confidence}."
        ),
    )


def beta_prior_from_proportion(
    expected_prop: float,
    sample_size_equiv: float,
) -> Prior:
    r"""Build a Beta prior on a proportion from a target mean and a
    pseudo-sample-size.

    Translates "I expect ``expected_prop`` of the team to do overtime
    in any given month, with the conviction equivalent to having
    observed ``sample_size_equiv`` trials" into a Beta distribution
    matching that mean and effective sample size:

    .. math::

        \alpha_0 = \text{expected\_prop} \cdot \text{sample\_size\_equiv},
        \qquad
        \beta_0 = (1 - \text{expected\_prop}) \cdot \text{sample\_size\_equiv}.

    The resulting prior has mean ``expected_prop`` and variance that
    decreases as ``sample_size_equiv`` grows.

    Parameters
    ----------
    expected_prop
        Prior expected proportion, in ``(0, 1)``. E.g. ``0.20`` for
        "20 % of the team does overtime".
    sample_size_equiv
        Pseudo-sample-size :math:`n_0 = \alpha_0 + \beta_0` representing
        how strongly the prior is held. Must be positive. ``2``–``5`` is
        weakly informative; ``20+`` is strongly informative.

    Returns
    -------
    Prior
        Beta prior with hyperparameters ``alpha`` and ``beta``.

    Raises
    ------
    ValueError
        If ``expected_prop`` is not in ``(0, 1)`` or
        ``sample_size_equiv`` is not positive.

    Examples
    --------
    20 % overtime proportion with effective sample size 10:

    >>> p = beta_prior_from_proportion(0.2, 10)
    >>> p.params
    {'alpha': 2.0, 'beta': 8.0}
    """
    if not 0 < expected_prop < 1:
        raise ValueError(
            f"expected_prop must be in (0, 1); got {expected_prop}"
        )
    if sample_size_equiv <= 0:
        raise ValueError(
            f"sample_size_equiv must be positive; got {sample_size_equiv}"
        )

    alpha = float(expected_prop * sample_size_equiv)
    beta = float((1.0 - expected_prop) * sample_size_equiv)
    return Prior(
        family="beta",
        params={"alpha": alpha, "beta": beta},
        notes=(
            f"Elicited as Beta(alpha={alpha}, beta={beta}) with "
            f"expected_prop={expected_prop} and pseudo-N={sample_size_equiv}."
        ),
    )


def prior_summary(
    prior: Prior,
    credible_level: float = 0.95,
) -> dict[str, float | tuple[float, float]]:
    """Compute mean, variance, std and a central credible interval.

    The interval is *equal-tailed*: lower and upper quantiles are
    ``(1 - credible_level) / 2`` and ``(1 + credible_level) / 2``.

    Parameters
    ----------
    prior
        The prior to summarise.
    credible_level
        The total probability mass between the interval endpoints.
        Default 0.95.

    Returns
    -------
    dict
        Keys: ``"mean"``, ``"variance"``, ``"std"``,
        ``"credible_interval"`` (a 2-tuple ``(lo, hi)``),
        ``"credible_level"``.

    Examples
    --------
    >>> p = normal_prior_from_budget(1_050_000, 0.90, 0.15)
    >>> s = prior_summary(p, credible_level=0.95)
    >>> sorted(s.keys())
    ['credible_interval', 'credible_level', 'mean', 'std', 'variance']
    """
    if not 0 < credible_level < 1:
        raise ValueError(
            f"credible_level must be in (0, 1); got {credible_level}"
        )

    rv = prior.frozen()
    alpha_tail = (1.0 - credible_level) / 2.0
    lo = float(rv.ppf(alpha_tail))
    hi = float(rv.ppf(1.0 - alpha_tail))
    mean = float(rv.mean())
    var = float(rv.var())
    return {
        "mean": mean,
        "variance": var,
        "std": float(np.sqrt(var)),
        "credible_interval": (lo, hi),
        "credible_level": float(credible_level),
    }
