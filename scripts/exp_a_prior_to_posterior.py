"""Experiment A — prior, likelihood, posterior on the same axis.

Visualises a single Normal-Normal update with the full variance
decomposition annotated:

- Prior  ``N(μ₀, σ₀²)``.
- Likelihood kernel as a function of θ:  ``N(x̄, σ²/n)``.
- Posterior  ``N(μₙ, σₙ²)``.

A side-panel inset reports σ₀, σ, σₙ and the precision-weighted mean
weights, so the reader can see *exactly* how prior strength and data
strength combine.

Output: ``figures/exp_a_prior_to_posterior.png`` at 300 DPI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.conjugate import NormalNormalUpdater

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = REPO_ROOT / "figures" / "exp_a_prior_to_posterior.png"


def main() -> None:
    mu0, sigma0, sigma = 1_050_000.0, 150_000.0, 80_000.0
    actuals = np.array([1_120_000.0, 1_080_000.0, 1_095_000.0])
    n = len(actuals)
    x_bar = float(actuals.mean())

    updater = NormalNormalUpdater(
        mu0=mu0, sigma0_sq=sigma0**2, sigma_sq=sigma**2
    )
    post = updater.update(actuals)
    mu_n, sigma_n = post.mean(), post.std()

    tau0 = 1.0 / sigma0**2
    tau = 1.0 / sigma**2
    w_prior = tau0 / (tau0 + n * tau)
    w_data = 1.0 - w_prior

    grid = np.linspace(mu0 - 4.5 * sigma0, mu0 + 4.5 * sigma0, 1200)
    pdf_prior = stats.norm(mu0, sigma0).pdf(grid)
    pdf_lik = stats.norm(x_bar, sigma / np.sqrt(n)).pdf(grid)
    pdf_post = stats.norm(mu_n, sigma_n).pdf(grid)

    fig, ax = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)

    ax.fill_between(grid, pdf_prior, alpha=0.15, color="steelblue")
    ax.fill_between(grid, pdf_lik, alpha=0.15, color="goldenrod")
    ax.fill_between(grid, pdf_post, alpha=0.20, color="crimson")

    ax.plot(grid, pdf_prior, color="steelblue", lw=2,
            label=fr"Prior  $N({mu0:,.0f},\ \sigma_0={sigma0:,.0f})$")
    ax.plot(grid, pdf_lik, color="goldenrod", lw=2, ls="--",
            label=fr"Likelihood kernel  $N(\bar x={x_bar:,.0f},\ \sigma/\sqrt{{n}}={sigma/np.sqrt(n):,.0f})$")
    ax.plot(grid, pdf_post, color="crimson", lw=2.5,
            label=fr"Posterior  $N({mu_n:,.0f},\ \sigma_n={sigma_n:,.0f})$")

    for x_, c in [(mu0, "steelblue"), (x_bar, "goldenrod"), (mu_n, "crimson")]:
        ax.axvline(x_, color=c, ls=":", alpha=0.5)

    annotation = (
        r"$\bf{Variance\ decomposition}$" + "\n"
        fr"$\sigma_0={sigma0:,.0f}$, $\sigma={sigma:,.0f}$, n={n}" + "\n"
        r"$\sigma_n^2 = \frac{\sigma^2 \sigma_0^2}{\sigma^2+n\sigma_0^2}$" + "\n"
        fr"$\Rightarrow\sigma_n={sigma_n:,.0f}$" + "\n\n"
        r"$\bf{Precision\!-\!weighted\ mean}$" + "\n"
        fr"$w_{{\mathrm{{prior}}}} = \tau_0/(\tau_0+n\tau) = {w_prior:.3f}$" + "\n"
        fr"$w_{{\mathrm{{data}}}}  = n\tau/(\tau_0+n\tau) = {w_data:.3f}$"
    )
    ax.text(
        0.985, 0.97, annotation,
        transform=ax.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.6"),
        fontsize=9, family="monospace",
    )

    ax.set_xlabel(r"$\theta$ — mean monthly cost (R\$)")
    ax.set_ylabel("density")
    ax.set_title(
        "Experiment A — single Normal-Normal update: "
        "the posterior is the precision-weighted blend of prior and likelihood"
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.ticklabel_format(style="plain", axis="x")
    ax.grid(alpha=0.3)

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=300)
    print(
        f"Saved {FIG_PATH.relative_to(REPO_ROOT)}  "
        f"({FIG_PATH.stat().st_size / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    main()
