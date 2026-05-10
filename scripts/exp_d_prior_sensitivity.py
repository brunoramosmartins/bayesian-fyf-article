"""Experiment D — three priors converge to the same posterior.

Same data, three different priors (low, reference, high). The
posteriors converge by Q3, illustrating that the data dominates the
prior in the long run. The right panel shows the *gap* between
extreme priors decaying at exactly :math:`w_0(n)` (proved in Phase 3
§3.1).

Output: ``figures/exp_d_prior_sensitivity.png`` at 300 DPI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.fyf_model import FYFConfig, FYFModel
from src.updating import normal_normal_shrinkage_weight

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = REPO_ROOT / "figures" / "exp_d_prior_sensitivity.png"
SEED = 20260912


def _run(prior_mean: float, prior_sd: float, obs_sd: float,
         actuals: np.ndarray) -> FYFModel:
    config = FYFConfig(prior_mean=prior_mean, prior_sd=prior_sd, obs_sd=obs_sd)
    model = FYFModel(config)
    model.annual_cycle(list(actuals))
    return model


def main() -> None:
    sigma0, sigma = 150_000.0, 80_000.0
    true_mean = 1_080_000.0
    rng = np.random.default_rng(SEED)
    actuals = rng.normal(true_mean, sigma, size=12)

    priors = {
        "low":       900_000.0,
        "reference": 1_050_000.0,
        "high":      1_200_000.0,
    }
    colors = {"low": "darkgreen", "reference": "steelblue", "high": "crimson"}

    models = {
        name: _run(mu0, sigma0, sigma, actuals) for name, mu0 in priors.items()
    }

    months = np.arange(1, 13)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    for name, model in models.items():
        means = np.array([r.posterior.mean() for r in model.reviews()])
        axL.plot(months, means, "o-", color=colors[name], lw=2,
                 label=fr"{name}: $\mu_0={priors[name]:,.0f}$")
        axL.axhline(priors[name], color=colors[name], ls=":", alpha=0.4)
    axL.axhline(true_mean, color="black", ls="--", alpha=0.5,
                label=fr"True $\theta_\star={true_mean:,.0f}$")
    axL.set_xticks(months)
    axL.set_xlabel("Month $n$")
    axL.set_ylabel(r"Posterior mean $\mu_n$ (R\$)")
    axL.set_title("Three disagreeing priors converge to the same posterior")
    axL.ticklabel_format(style="plain", axis="y")
    axL.grid(alpha=0.3)
    axL.legend(fontsize=8.5, loc="lower right")

    high_traj = np.array([r.posterior.mean() for r in models["high"].reviews()])
    low_traj = np.array([r.posterior.mean() for r in models["low"].reviews()])
    gap_obs = high_traj - low_traj
    gap0 = priors["high"] - priors["low"]
    w0_n = np.array([
        normal_normal_shrinkage_weight(sigma0, sigma, n) for n in months
    ])
    gap_predicted = w0_n * gap0

    axR.plot(months, gap_obs, "o-", color="crimson", lw=2,
             label=r"Observed gap  $\mu_n^{\mathrm{high}}-\mu_n^{\mathrm{low}}$")
    axR.plot(months, gap_predicted, "s--", color="steelblue", lw=1.5,
             label=r"Theoretical  $w_0(n)\cdot \Delta\mu_0$")
    axR.axhline(gap0, color="black", ls=":", alpha=0.4,
                label=f"Initial gap = R$ {gap0:,.0f}")
    axR.set_xticks(months)
    axR.set_yscale("log")
    axR.set_xlabel("Month $n$")
    axR.set_ylabel("Disagreement (R\\$, log scale)")
    axR.set_title(r"Gap shrinks at rate $w_0(n)$ — exact theorem (Phase 3 §3.1)")
    axR.ticklabel_format(style="plain", axis="x")
    axR.grid(alpha=0.3, which="both")
    axR.legend(fontsize=8.5, loc="upper right")

    fig.suptitle(
        f"Experiment D — prior sensitivity  "
        f"(σ₀={sigma0:,.0f}, σ={sigma:,.0f}, true θ={true_mean:,.0f}, seed={SEED})",
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
