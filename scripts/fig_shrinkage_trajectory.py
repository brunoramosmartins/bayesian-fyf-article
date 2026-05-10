"""Generate ``figures/shrinkage_trajectory.png``.

Two-panel figure for §4 of the article:

- Left panel: posterior mean :math:`\\mu_n` and 95 % credible intervals
  across 12 months, against the prior mean and the cumulative sample
  mean.
- Right panel: shrinkage weight :math:`w_0(n) = \\tau_0/(\\tau_0+n\\tau)`
  and data weight :math:`1-w_0(n)`, with horizontal markers at the 80 %
  and 95 % data-weight thresholds.

Usage
-----
From the repository root, with the package installed editable::

    python scripts/fig_shrinkage_trajectory.py

The simulation uses a fixed RNG seed so the figure is reproducible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.conjugate import NormalNormalUpdater
from src.updating import (
    SequentialUpdater,
    months_to_data_weight,
    normal_normal_shrinkage_weight,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = REPO_ROOT / "figures" / "shrinkage_trajectory.png"
SEED = 20260601
N_MONTHS = 12


def main() -> None:
    mu0, sigma0, sigma = 1_050_000.0, 150_000.0, 80_000.0
    theta_star = 1_080_000.0  # true mean used to simulate actuals

    rng = np.random.default_rng(seed=SEED)
    actuals = rng.normal(loc=theta_star, scale=sigma, size=N_MONTHS)

    prior = NormalNormalUpdater(
        mu0=mu0, sigma0_sq=sigma0**2, sigma_sq=sigma**2
    )
    seq = SequentialUpdater(prior)
    seq.feed_batch(list(actuals))

    months = np.arange(1, N_MONTHS + 1)
    means = np.array([e.posterior.mean() for e in seq.history()])
    stds = np.array([e.posterior.std() for e in seq.history()])
    cum_xbar = np.cumsum(actuals) / months

    # Closed-form weights for the right panel — extend slightly past N_MONTHS
    # so the asymptote is visible.
    n_grid = np.arange(1, 25)
    w0 = normal_normal_shrinkage_weight(sigma0, sigma, n_grid)
    n80 = months_to_data_weight(sigma0, sigma, 0.80)
    n95 = months_to_data_weight(sigma0, sigma, 0.95)

    # ---- Plot --------------------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)

    # Left: posterior trajectory with 95 % credible interval
    axL.fill_between(
        months,
        means - 1.96 * stds,
        means + 1.96 * stds,
        alpha=0.20,
        color="crimson",
        label="95 % credible interval",
    )
    axL.plot(months, means, "o-", color="crimson", lw=2, label=r"Posterior mean $\mu_n$")
    axL.plot(months, cum_xbar, "s--", color="goldenrod", lw=1.5,
             label=r"Cumulative sample mean $\bar x_n$")
    axL.axhline(mu0, color="steelblue", ls=":", alpha=0.7,
                label=fr"Prior mean $\mu_0={mu0:,.0f}$")
    axL.axhline(theta_star, color="black", ls="--", alpha=0.4,
                label=fr"True mean $\theta_\star={theta_star:,.0f}$")
    axL.set_xlabel("Month $n$")
    axL.set_ylabel("Mean monthly cost (R\\$)")
    axL.set_title("Posterior trajectory with shrinking 95 % CI")
    axL.set_xticks(months)
    axL.ticklabel_format(style="plain", axis="y")
    axL.grid(alpha=0.3)
    axL.legend(fontsize=8.5, loc="upper right")

    # Right: shrinkage weights
    axR.plot(n_grid, w0, "o-", color="steelblue", lw=2, label=r"Prior weight $w_0(n)$")
    axR.plot(n_grid, 1 - w0, "s-", color="crimson", lw=2,
             label=r"Data weight $1-w_0(n)$")
    axR.axhline(0.80, color="goldenrod", ls=":", alpha=0.6)
    axR.axhline(0.95, color="darkgreen", ls=":", alpha=0.6)
    axR.axvline(n80, color="goldenrod", ls=":", alpha=0.6,
                label=fr"$n={n80}$: data weight $> 80\%$")
    axR.axvline(n95, color="darkgreen", ls=":", alpha=0.6,
                label=fr"$n={n95}$: data weight $> 95\%$")
    axR.set_xlabel("Month $n$")
    axR.set_ylabel("Weight")
    axR.set_title(r"Shrinkage weight $w_0(n) = \tau_0/(\tau_0 + n\tau)$")
    axR.set_ylim(-0.02, 1.02)
    axR.set_xticks(np.arange(1, 25, 2))
    axR.grid(alpha=0.3)
    axR.legend(fontsize=8.5, loc="center right")

    fig.suptitle(
        "Shrinkage trajectory — Normal-Normal FYF reference scenario "
        f"($\\sigma_0={sigma0:,.0f}$, $\\sigma={sigma:,.0f}$, seed={SEED})",
        fontsize=11,
    )

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=300)
    print(f"Saved {FIG_PATH.relative_to(REPO_ROOT)}  ({FIG_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
