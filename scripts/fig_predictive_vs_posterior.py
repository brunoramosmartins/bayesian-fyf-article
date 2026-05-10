"""Generate ``figures/predictive_vs_posterior.png``.

Two-panel figure for §5 of the article, contrasting:

- Left panel: the **posterior on theta** vs the **posterior predictive
  on next month's cost**, sharing a centre but with very different
  widths. The predictive band is wider because it adds sampling noise
  to parameter uncertainty.
- Right panel: the **predictive of the year-end total**, with the
  budget ceiling marked and the "over-budget" tail shaded. Compare
  the correct quadratic-h variance against the naïve
  iid-future-months formula side by side.

Usage
-----
From the repository root, with the package installed editable::

    python scripts/fig_predictive_vs_posterior.py

Output
------
``figures/predictive_vs_posterior.png`` at 300 DPI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.conjugate import NormalPosterior
from src.predictive import (
    posterior_predictive_normal,
    year_end_predictive_total,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = REPO_ROOT / "figures" / "predictive_vs_posterior.png"


def main() -> None:
    # FYF posterior at mid-year (Phase 3 / Phase 4 reference).
    mu_n = 1_085_000.0
    sigma_n = 32_000.0
    sigma = 80_000.0
    posterior = NormalPosterior(mu=mu_n, sigma_sq=sigma_n**2)

    # Year-end forecast inputs.
    n_remaining = 6
    observed_total = 6_510_000.0
    budget = 13_200_000.0

    # ----- Left panel: posterior vs predictive on x̃ ----------------------
    pred_next = posterior_predictive_normal(posterior, sigma_sq=sigma**2)
    pred_sigma = float(pred_next.std())
    grid = np.linspace(mu_n - 4 * pred_sigma, mu_n + 4 * pred_sigma, 800)

    pdf_post = stats.norm(loc=mu_n, scale=sigma_n).pdf(grid)
    pdf_pred = pred_next.pdf(grid)

    ci_post = (mu_n - 1.96 * sigma_n, mu_n + 1.96 * sigma_n)
    ci_pred = (mu_n - 1.96 * pred_sigma, mu_n + 1.96 * pred_sigma)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    axL.plot(grid, pdf_post, lw=2, color="steelblue",
             label=fr"Posterior $\theta$  $N({mu_n:,.0f},\ {sigma_n:,.0f}^2)$")
    axL.plot(grid, pdf_pred, lw=2, color="crimson",
             label=fr"Predictive $\tilde x$  $N({mu_n:,.0f},\ {pred_sigma:,.0f}^2)$")
    axL.axvspan(*ci_post, color="steelblue", alpha=0.12)
    axL.axvspan(*ci_pred, color="crimson", alpha=0.10)
    for x, c, ls in [(ci_post[0], "steelblue", ":"), (ci_post[1], "steelblue", ":"),
                     (ci_pred[0], "crimson", ":"), (ci_pred[1], "crimson", ":")]:
        axL.axvline(x, color=c, ls=ls, alpha=0.5)
    axL.set_title(
        f"Posterior vs predictive: predictive is {pred_sigma / sigma_n:.1f}× wider"
    )
    axL.set_xlabel(r"value (R\$)")
    axL.set_ylabel("density")
    axL.ticklabel_format(style="plain", axis="x")
    axL.grid(alpha=0.3)
    axL.legend(fontsize=9)

    # ----- Right panel: year-end total predictive with budget tail --------
    forecast_correct = year_end_predictive_total(
        posterior=posterior,
        sigma_sq=sigma**2,
        n_remaining=n_remaining,
        observed_total=observed_total,
    )
    sigma_T_correct = forecast_correct.std()

    # Naïve iid-future-months variance for comparison.
    naive_var = n_remaining * (posterior.variance() + sigma**2)
    sigma_T_naive = float(np.sqrt(naive_var))
    p_correct = forecast_correct.prob_above(budget)
    p_naive = float(stats.norm(loc=forecast_correct.mean, scale=sigma_T_naive).sf(budget))

    grid_T = np.linspace(
        forecast_correct.mean - 4 * sigma_T_correct,
        forecast_correct.mean + 4 * sigma_T_correct,
        800,
    )
    pdf_T_correct = stats.norm(loc=forecast_correct.mean, scale=sigma_T_correct).pdf(grid_T)
    pdf_T_naive = stats.norm(loc=forecast_correct.mean, scale=sigma_T_naive).pdf(grid_T)

    axR.plot(grid_T, pdf_T_correct, lw=2.5, color="crimson",
             label=fr"Correct  $N(T_\mu,\ {sigma_T_correct/1e3:,.0f}\text{{K}}^2)$")
    axR.plot(grid_T, pdf_T_naive, lw=1.5, color="goldenrod", ls="--",
             label=fr"Naïve iid  $N(T_\mu,\ {sigma_T_naive/1e3:,.0f}\text{{K}}^2)$")

    # Shade the over-budget tail under the correct curve.
    mask = grid_T >= budget
    axR.fill_between(grid_T[mask], pdf_T_correct[mask], color="crimson", alpha=0.30,
                     label=f"P(T > B) = {p_correct:.1%} (correct)")
    axR.axvline(budget, color="black", lw=1.5, ls="--",
                label=fr"Budget $B=\text{{R\$}} {budget:,.0f}$")
    axR.axvline(forecast_correct.mean, color="crimson", ls=":", alpha=0.5,
                label=fr"$\mathbb{{E}}[T]=\text{{R\$}} {forecast_correct.mean:,.0f}$")

    axR.set_title(
        f"Year-end total: correct {p_correct:.1%} vs naïve {p_naive:.1%}"
    )
    axR.set_xlabel(r"annual total $T$ (R\$)")
    axR.set_ylabel("density")
    axR.ticklabel_format(style="plain", axis="x")
    axR.grid(alpha=0.3)
    axR.legend(fontsize=8.5, loc="upper left")

    fig.suptitle(
        "Posterior vs posterior predictive — FYF mid-year forecast"
        f"  ($\\mu_n={mu_n:,.0f}$, $\\sigma_n={sigma_n:,.0f}$, $\\sigma={sigma:,.0f}$)",
        fontsize=11,
    )

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=300)
    print(
        f"Saved {FIG_PATH.relative_to(REPO_ROOT)}  "
        f"({FIG_PATH.stat().st_size / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    main()
