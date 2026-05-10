"""Experiment C — sequential shrinkage across the FYF cycle.

Plots the posterior mean trajectory with a 95 % credible band over
12 simulated months, marking the four FYF reviews. The right-hand
panel shows the closed-form shrinkage weight :math:`w_0(n)` decaying
toward zero, with horizontal markers at the 80 % and 95 % data-weight
thresholds.

Output: ``figures/exp_c_sequential_shrinkage.png`` at 300 DPI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.fyf_model import (
    QUARTER_END_MONTHS,
    FYFConfig,
    FYFModel,
)
from src.updating import months_to_data_weight, normal_normal_shrinkage_weight

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = REPO_ROOT / "figures" / "exp_c_sequential_shrinkage.png"
SEED = 20260911


def main() -> None:
    config = FYFConfig(
        prior_mean=1_050_000.0,
        prior_sd=150_000.0,
        obs_sd=80_000.0,
        budget_ceiling=13_200_000.0,
    )
    rng = np.random.default_rng(SEED)
    actuals = rng.normal(1_080_000.0, config.obs_sd, size=12)

    model = FYFModel(config)
    model.annual_cycle(list(actuals))
    reviews = model.reviews()

    months = np.arange(1, 13)
    means = np.array([r.posterior.mean() for r in reviews])
    stds = np.array([r.posterior.std() for r in reviews])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    # ----- left: trajectory ------------------------------------------------
    axL.fill_between(
        months, means - 1.96 * stds, means + 1.96 * stds,
        color="steelblue", alpha=0.20, label="95 % credible interval",
    )
    axL.plot(months, means, "o-", color="steelblue", lw=2,
             label=r"Posterior mean $\mu_n$")
    axL.scatter(months, actuals, color="dimgray", s=22, alpha=0.7,
                label="Monthly actuals", zorder=3)
    axL.axhline(config.prior_mean, color="black", ls="--", alpha=0.4,
                label=fr"Prior $\mu_0={config.prior_mean:,.0f}$")
    for q in QUARTER_END_MONTHS:
        axL.axvline(q + 0.5, color="goldenrod", ls=":", alpha=0.5)
    axL.set_xticks(months)
    axL.set_xlabel("Month $n$")
    axL.set_ylabel(r"Mean monthly cost (R\$)")
    axL.set_title("Posterior trajectory with shrinking 95 % CI")
    axL.ticklabel_format(style="plain", axis="y")
    axL.grid(alpha=0.3)
    axL.legend(fontsize=8.5, loc="upper right")
    for q in QUARTER_END_MONTHS:
        axL.text(q + 0.5, axL.get_ylim()[1], f"FYF #{q // 3}",
                 ha="center", va="top", fontsize=8.5,
                 color="goldenrod", rotation=0)

    # ----- right: shrinkage weight ----------------------------------------
    n_grid = np.arange(1, 25)
    w0 = normal_normal_shrinkage_weight(config.prior_sd, config.obs_sd, n_grid)
    n80 = months_to_data_weight(config.prior_sd, config.obs_sd, 0.80)
    n95 = months_to_data_weight(config.prior_sd, config.obs_sd, 0.95)

    axR.plot(n_grid, 1 - w0, "s-", color="crimson", lw=2,
             label=r"Data weight $1-w_0(n)$")
    axR.plot(n_grid, w0, "o-", color="steelblue", lw=2,
             label=r"Prior weight $w_0(n)$")
    axR.axhline(0.80, color="goldenrod", ls=":", alpha=0.5)
    axR.axhline(0.95, color="darkgreen", ls=":", alpha=0.5)
    axR.axvline(n80, color="goldenrod", ls=":", alpha=0.5,
                label=fr"$n={n80}$: data $> 80\%$")
    axR.axvline(n95, color="darkgreen", ls=":", alpha=0.5,
                label=fr"$n={n95}$: data $> 95\%$")
    axR.set_ylim(-0.02, 1.02)
    axR.set_xticks(np.arange(1, 25, 2))
    axR.set_xlabel("Month $n$")
    axR.set_ylabel("Weight")
    axR.set_title(r"Shrinkage weight $w_0(n)$ — closed form")
    axR.grid(alpha=0.3)
    axR.legend(fontsize=8.5, loc="center right")

    fig.suptitle(
        f"Experiment C — sequential shrinkage  "
        f"(σ₀={config.prior_sd:,.0f}, σ={config.obs_sd:,.0f}, "
        f"true θ=1,080,000, seed={SEED})",
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
