"""Experiment H — calibration and surprise diagnostic.

Three-panel figure for the article's diagnostic section:

- Left: empirical predictive coverage as a function of nominal level,
  averaged across many simulated annual cycles. A correctly specified
  model lies on the diagonal.
- Centre: histogram of predictive z-scores from the same simulation;
  should look approximately ``N(0, 1)``.
- Right: empirical CDF of two-sided p-values vs the Uniform CDF;
  deviations from the diagonal flag mis-calibration.

Output: ``figures/exp_h_posterior_predictive_check.png`` at 300 DPI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.conjugate import NormalPosterior
from src.diagnostics import (
    calibration_score,
    posterior_predictive_pvalues,
    priors_before_each_step,
    surprise_trajectory,
)
from src.fyf_model import FYFConfig, FYFModel

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = REPO_ROOT / "figures" / "exp_h_posterior_predictive_check.png"
SEED = 20260916


def main() -> None:
    config = FYFConfig(
        prior_mean=1_050_000.0,
        prior_sd=150_000.0,
        obs_sd=80_000.0,
    )
    rng = np.random.default_rng(SEED)
    n_replications = 250
    n_months = 12

    all_z = []
    all_pvals = []
    all_priors_per_rep: list[list[NormalPosterior]] = []
    all_actuals_per_rep: list[list[float]] = []

    for _ in range(n_replications):
        # Draw a single θ from the prior; then 12 actuals from N(θ, σ²).
        true_theta = float(rng.normal(config.prior_mean, config.prior_sd))
        actuals = list(rng.normal(true_theta, config.obs_sd, size=n_months))

        model = FYFModel(config)
        model.annual_cycle(actuals)
        prior = NormalPosterior(
            mu=config.prior_mean, sigma_sq=config.prior_sd**2
        )
        priors = priors_before_each_step(model.reviews(), prior)
        z = surprise_trajectory(priors, actuals, sigma_sq=config.obs_sd**2)
        p = posterior_predictive_pvalues(priors, actuals, sigma_sq=config.obs_sd**2)
        all_z.extend(z.tolist())
        all_pvals.extend(p.tolist())
        all_priors_per_rep.append(priors)
        all_actuals_per_rep.append(actuals)

    all_z_arr = np.array(all_z)
    all_p_arr = np.array(all_pvals)

    nominal_levels = np.linspace(0.50, 0.99, 25)
    empirical_coverage = []
    for level in nominal_levels:
        coverages = [
            calibration_score(p_, x_, sigma_sq=config.obs_sd**2, level=level)
            for p_, x_ in zip(all_priors_per_rep, all_actuals_per_rep, strict=True)
        ]
        empirical_coverage.append(float(np.mean(coverages)))
    empirical_coverage_arr = np.array(empirical_coverage)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)

    # ----- 1: calibration plot --------------------------------------------
    axes[0].plot([0, 1], [0, 1], color="black", ls="--", alpha=0.5,
                 label="ideal (model is calibrated)")
    axes[0].plot(nominal_levels, empirical_coverage_arr, "o-",
                 color="crimson", lw=2, label="empirical")
    axes[0].set_xlabel("nominal predictive level $1-\\alpha$")
    axes[0].set_ylabel("empirical coverage")
    axes[0].set_xlim(0.5, 1.0)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title(
        "Calibration plot — averaged over "
        f"{n_replications} simulated cycles"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8.5)

    # ----- 2: histogram of z-scores ---------------------------------------
    bins = np.linspace(-4, 4, 41)
    axes[1].hist(all_z_arr, bins=bins, density=True, color="steelblue",
                 alpha=0.65, edgecolor="white", label="empirical")
    grid = np.linspace(-4, 4, 400)
    axes[1].plot(grid, stats.norm.pdf(grid), color="black", lw=1.6,
                 label="$N(0, 1)$ reference")
    axes[1].set_xlabel(r"surprise $z_t$")
    axes[1].set_ylabel("density")
    axes[1].set_title("Surprise z-score distribution")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8.5)

    # ----- 3: empirical CDF of p-values vs Uniform ------------------------
    sorted_p = np.sort(all_p_arr)
    ecdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
    axes[2].plot([0, 1], [0, 1], color="black", ls="--", alpha=0.5,
                 label="Uniform reference")
    axes[2].plot(sorted_p, ecdf, color="crimson", lw=1.8, label="empirical")
    axes[2].set_xlabel(r"$p_t$")
    axes[2].set_ylabel("empirical CDF")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Predictive p-value CDF (should be Uniform)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8.5)

    fig.suptitle(
        f"Experiment H — posterior predictive check  "
        f"(reps={n_replications}, seed={SEED})",
        fontsize=12,
    )

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=300)
    print(
        f"Saved {FIG_PATH.relative_to(REPO_ROOT)}  "
        f"({FIG_PATH.stat().st_size / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    main()
