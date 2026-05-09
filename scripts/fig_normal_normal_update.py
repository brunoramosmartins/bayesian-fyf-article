"""Generate ``figures/normal_normal_update.png``.

Visualises the FYF reference Normal-Normal update from
``docs/model-design.md``: prior + Q1 actuals → posterior. Three curves
on the same axis (prior, scaled likelihood kernel, posterior) plus
markers for the prior mean, sample mean, and posterior mean.

Usage
-----
From the repository root, with the package installed editable::

    python scripts/fig_normal_normal_update.py

Output
------
``figures/normal_normal_update.png`` at 300 DPI.

The script is deterministic — there is no random sampling — so no seed
is needed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.conjugate import NormalNormalUpdater


REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = REPO_ROOT / "figures" / "normal_normal_update.png"


def main() -> None:
    # FYF reference scenario from docs/model-design.md
    mu0 = 1_050_000.0
    sigma0 = 150_000.0
    sigma = 80_000.0
    actuals = np.array([1_120_000.0, 1_080_000.0, 1_095_000.0])
    x_bar = float(actuals.mean())

    updater = NormalNormalUpdater(mu0=mu0, sigma0_sq=sigma0**2, sigma_sq=sigma**2)
    post = updater.update(actuals)
    mu_n, sigma_n = post.mean(), post.std()

    # x-axis: 5 prior σ around prior mean — comfortably covers all curves.
    grid = np.linspace(mu0 - 5 * sigma0, mu0 + 5 * sigma0, 1000)

    prior_pdf = stats.norm(loc=mu0, scale=sigma0).pdf(grid)
    # Likelihood as a function of θ has the kernel of N(x_bar, σ²/n).
    n = len(actuals)
    lik_kernel = stats.norm(loc=x_bar, scale=sigma / np.sqrt(n)).pdf(grid)
    post_pdf = stats.norm(loc=mu_n, scale=sigma_n).pdf(grid)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(grid, prior_pdf, label=fr"Prior $N({mu0:,.0f},\, {sigma0:,.0f}^2)$",
            color="steelblue", lw=2)
    ax.plot(
        grid, lik_kernel,
        label=fr"Likelihood kernel  $N(\bar x,\, \sigma^2/n)$, $\bar x={x_bar:,.0f}$",
        color="goldenrod", lw=2, ls="--",
    )
    ax.plot(grid, post_pdf, label=fr"Posterior $N({mu_n:,.0f},\, {sigma_n:,.0f}^2)$",
            color="crimson", lw=2)

    for x, label, color in [
        (mu0, r"$\mu_0$", "steelblue"),
        (x_bar, r"$\bar x$", "goldenrod"),
        (mu_n, r"$\mu_n$", "crimson"),
    ]:
        ax.axvline(x, color=color, ls=":", alpha=0.55)
        ax.annotate(
            label, xy=(x, ax.get_ylim()[1]), xytext=(2, -8),
            textcoords="offset points", color=color, fontsize=10,
        )

    ax.set_title(
        "Normal-Normal update: prior + Q1 actuals → posterior\n"
        f"Posterior s.d. shrinks from R$ {sigma0:,.0f} to R$ {sigma_n:,.0f}",
        fontsize=11,
    )
    ax.set_xlabel(r"$\theta$ — mean monthly cost (R\$)")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.ticklabel_format(style="plain", axis="x")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=300)
    print(f"Saved {FIG_PATH.relative_to(REPO_ROOT)}  ({FIG_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
