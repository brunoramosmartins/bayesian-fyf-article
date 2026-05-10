"""Plotting helpers for the FYF model and its diagnostics.

The functions take an :class:`src.fyf_model.FYFModel` (or its history
of :class:`src.fyf_model.MonthlyReview` records) and draw onto a
matplotlib :class:`~matplotlib.axes.Axes`. They never call
``plt.show`` or ``savefig`` themselves — composing into multi-panel
figures and saving is the caller's responsibility (see
``scripts/fig_fyf_scenarios.py``).

Conventions:

- Months are 1-indexed on the x-axis.
- Posterior trajectories (parameter inference) use **steelblue**.
- Predictive trajectories or predictions for next observation use
  **crimson**.
- The prior mean / true value horizontal lines use a **black**
  dashed line.
- The budget ceiling uses a **goldenrod** dashed line.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from matplotlib.axes import Axes

from src.fyf_model import QUARTER_END_MONTHS, FYFModel, MonthlyReview

# =============================================================================
# Posterior evolution
# =============================================================================


def plot_posterior_evolution(
    ax: Axes,
    reviews: list[MonthlyReview],
    *,
    prior_mean: float | None = None,
    true_mean: float | None = None,
    show_actuals: bool = True,
    credible_level: float = 0.95,
    title: str = "Posterior trajectory across the FYF cycle",
) -> Axes:
    """Plot the posterior mean μ_n with a credible-interval band.

    Optional overlays:

    - ``prior_mean``: horizontal dashed line at :math:`\\mu_0`.
    - ``true_mean``: horizontal dashed line at :math:`\\theta_\\star`
      (only known in simulation).
    - ``show_actuals``: scatter the observed monthly actuals as small
      grey dots.
    """
    if not reviews:
        raise ValueError("reviews is empty.")
    z = _z_for_level(credible_level)
    months = np.array([r.month for r in reviews])
    means = np.array([r.posterior.mean() for r in reviews])
    stds = np.array([r.posterior.std() for r in reviews])

    ax.fill_between(
        months,
        means - z * stds,
        means + z * stds,
        color="steelblue",
        alpha=0.20,
        label=f"{int(credible_level * 100)} % credible interval",
    )
    ax.plot(months, means, "o-", color="steelblue", lw=2,
            label=r"Posterior mean $\mu_n$")

    if show_actuals:
        actuals = np.array([r.actual for r in reviews])
        ax.scatter(months, actuals, color="dimgray", s=22, alpha=0.7,
                   label="Monthly actuals", zorder=3)

    if prior_mean is not None:
        ax.axhline(prior_mean, color="black", ls="--", alpha=0.4,
                   label=fr"Prior $\mu_0={prior_mean:,.0f}$")
    if true_mean is not None:
        ax.axhline(true_mean, color="crimson", ls=":", alpha=0.5,
                   label=fr"True $\theta_\star={true_mean:,.0f}$")

    _decorate_monthly_axis(ax, months)
    ax.set_ylabel(r"Mean monthly cost (R\$)")
    ax.set_title(title)
    ax.legend(fontsize=8.5, loc="best")
    return ax


def plot_funnel_chart(
    ax: Axes,
    reviews: list[MonthlyReview],
    *,
    credible_level: float = 0.95,
    title: str = r"Posterior s.d. $\sigma_n$ across the cycle",
) -> Axes:
    """Plot how the posterior s.d. shrinks across the cycle."""
    if not reviews:
        raise ValueError("reviews is empty.")
    months = np.array([r.month for r in reviews])
    stds = np.array([r.posterior.std() for r in reviews])
    z = _z_for_level(credible_level)

    ax.bar(months, 2 * z * stds, color="steelblue", alpha=0.65, width=0.7,
           label=f"{int(credible_level * 100)} % CI width")
    ax.plot(months, 2 * z * stds, "o-", color="darkblue", lw=1.5)
    _decorate_monthly_axis(ax, months)
    ax.set_ylabel(r"Width of 95 % credible interval (R\$)")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="best")
    return ax


# =============================================================================
# Surprise / diagnostics
# =============================================================================


def plot_surprise_scores(
    ax: Axes,
    reviews: list[MonthlyReview],
    *,
    threshold: float = 2.0,
    title: str = "Surprise z-scores",
) -> Axes:
    """Bar chart of surprise z-scores per month with ±threshold lines."""
    if not reviews:
        raise ValueError("reviews is empty.")
    months = np.array([r.month for r in reviews])
    z = np.array([r.surprise_z for r in reviews])

    colors = np.where(np.abs(z) > threshold, "crimson", "steelblue")
    ax.bar(months, z, color=colors, alpha=0.75, width=0.7)
    for y in (-threshold, threshold):
        ax.axhline(y, color="goldenrod", ls=":", alpha=0.7)
    ax.axhline(0.0, color="black", lw=0.6)
    _decorate_monthly_axis(ax, months)
    ax.set_ylabel(r"Surprise $z_t$")
    ax.set_title(title)
    return ax


def plot_year_end_forecast(
    ax: Axes,
    reviews: list[MonthlyReview],
    *,
    budget_ceiling: float | None = None,
    title: str = "Year-end total forecast across the cycle",
) -> Axes:
    """Plot the year-end forecast mean ± 1.96σ at each month."""
    if not reviews:
        raise ValueError("reviews is empty.")
    months = []
    means = []
    los = []
    his = []
    for r in reviews:
        if r.forecast is None:
            continue
        months.append(r.month)
        means.append(r.forecast.mean)
        lo, hi = r.forecast.credible_interval(0.95)
        los.append(lo)
        his.append(hi)
    months = np.array(months)
    means = np.array(means)
    los = np.array(los)
    his = np.array(his)

    if months.size > 0:
        ax.fill_between(months, los, his, color="crimson", alpha=0.18,
                        label="95 % predictive interval")
        ax.plot(months, means, "o-", color="crimson", lw=2,
                label=r"$\mathbb{E}[T \mid x_{1:n}]$")
    if budget_ceiling is not None:
        ax.axhline(budget_ceiling, color="goldenrod", ls="--",
                   label=fr"Budget $B={budget_ceiling:,.0f}$")
    _decorate_monthly_axis(ax, np.arange(1, len(reviews) + 1))
    ax.set_ylabel(r"Year-end total $T$ (R\$)")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="best")
    return ax


def plot_p_over_budget(
    ax: Axes,
    reviews: list[MonthlyReview],
    *,
    title: str = r"$P(T > B)$ across the cycle",
) -> Axes:
    """Plot the probability of exceeding the budget at each month."""
    if not reviews:
        raise ValueError("reviews is empty.")
    months: list[int] = []
    probs: list[float] = []
    for r in reviews:
        if r.p_over_budget is None:
            continue
        months.append(r.month)
        probs.append(r.p_over_budget)
    if not months:
        ax.text(0.5, 0.5, "No budget ceiling configured.",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return ax

    ax.plot(months, probs, "o-", color="darkgreen", lw=2)
    ax.axhline(0.5, color="goldenrod", ls=":", alpha=0.6, label="50 %")
    ax.set_ylim(-0.02, 1.02)
    _decorate_monthly_axis(ax, np.array(months))
    ax.set_ylabel(r"$P(\mathrm{annual\ total}>B)$")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="best")
    return ax


# =============================================================================
# Multi-model comparison (used by S5 prior sensitivity)
# =============================================================================


def plot_fyf_comparison(
    ax: Axes,
    models: Iterable[FYFModel],
    *,
    labels: Iterable[str] | None = None,
    title: str = "Posterior trajectory comparison",
) -> Axes:
    """Overlay the posterior-mean trajectories of several models."""
    models = list(models)
    if not models:
        raise ValueError("models is empty.")
    if labels is None:
        labels = [f"model {i+1}" for i in range(len(models))]
    labels = list(labels)
    if len(labels) != len(models):
        raise ValueError("labels must match models in length.")

    palette = ("steelblue", "crimson", "darkgreen", "goldenrod", "purple")
    for i, (model, label) in enumerate(zip(models, labels, strict=True)):
        rev = model.reviews()
        if not rev:
            continue
        months = np.array([r.month for r in rev])
        means = np.array([r.posterior.mean() for r in rev])
        ax.plot(months, means, "o-", lw=2,
                color=palette[i % len(palette)], label=label)
    months = np.arange(1, models[0].config.annual_horizon + 1)
    _decorate_monthly_axis(ax, months)
    ax.set_ylabel(r"Posterior mean (R\$)")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="best")
    return ax


# =============================================================================
# Internals
# =============================================================================


def _decorate_monthly_axis(ax: Axes, months: np.ndarray) -> None:
    ax.set_xlabel("Month $n$")
    ax.set_xticks(months)
    for q_end in QUARTER_END_MONTHS:
        if q_end in months:
            ax.axvline(q_end + 0.5, color="black", lw=0.4, alpha=0.20)
    ax.ticklabel_format(style="plain", axis="y")
    ax.grid(alpha=0.3)


def _z_for_level(level: float) -> float:
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1); got {level}")
    from scipy.stats import norm  # local import keeps top minimal

    return float(norm.ppf(0.5 + level / 2.0))


__all__ = [
    "plot_funnel_chart",
    "plot_fyf_comparison",
    "plot_p_over_budget",
    "plot_posterior_evolution",
    "plot_surprise_scores",
    "plot_year_end_forecast",
]
