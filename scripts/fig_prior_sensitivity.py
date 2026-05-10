"""Generate ``figures/prior_sensitivity.png``.

Two-panel figure showing how disagreeing priors converge under the
same data:

- Left panel: posterior mean trajectories of three priors (low,
  reference, high) for the same simulated actuals. Disagreement
  shrinks visibly month-by-month.
- Right panel: the gap :math:`|\\mu_n^{(A)} - \\mu_n^{(B)}|` between the
  low and high prior, overlaid with the closed-form prediction
  :math:`w_0(n) \\cdot |\\mu_0^{(A)} - \\mu_0^{(B)}|`.

Usage
-----
From the repository root::

    python scripts/fig_prior_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.conjugate import NormalNormalUpdater
from src.updating import SequentialUpdater, normal_normal_shrinkage_weight


REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = REPO_ROOT / "figures" / "prior_sensitivity.png"
SEED = 20260602
N_MONTHS = 12


def main() -> None:
    sigma0, sigma = 150_000.0, 80_000.0
    theta_star = 1_080_000.0
    priors = {
        "low":       900_000.0,    # pessimistic planner
        "reference": 1_050_000.0,
        "high":      1_200_000.0,  # optimistic planner
    }
    colors = {
        "low":       "darkgreen",
        "reference": "steelblue",
        "high":      "crimson",
    }

    rng = np.random.default_rng(seed=SEED)
    actuals = rng.normal(loc=theta_star, scale=sigma, size=N_MONTHS)

    trajectories: dict[str, np.ndarray] = {}
    for name, mu0 in priors.items():
        upd = NormalNormalUpdater(mu0=mu0, sigma0_sq=sigma0**2, sigma_sq=sigma**2)
        seq = SequentialUpdater(upd)
        seq.feed_batch(list(actuals))
        trajectories[name] = np.array([e.posterior.mean() for e in seq.history()])

    months = np.arange(1, N_MONTHS + 1)

    # ---- Plot --------------------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)

    # Left: trajectories
    for name, traj in trajectories.items():
        axL.plot(
            months, traj, "o-",
            color=colors[name], lw=2,
            label=fr"Prior {name}: $\mu_0={priors[name]:,.0f}$",
        )
        axL.axhline(priors[name], color=colors[name], ls=":", alpha=0.4)
    axL.axhline(theta_star, color="black", ls="--", alpha=0.5,
                label=fr"True mean $\theta_\star={theta_star:,.0f}$")
    axL.set_xlabel("Month $n$")
    axL.set_ylabel(r"Posterior mean $\mu_n$ (R\$)")
    axL.set_title("Three disagreeing priors converge to the same posterior")
    axL.set_xticks(months)
    axL.ticklabel_format(style="plain", axis="y")
    axL.grid(alpha=0.3)
    axL.legend(fontsize=8.5, loc="lower right")

    # Right: gap shrinks at rate w_0(n)
    gap_observed = trajectories["high"] - trajectories["low"]
    gap_initial = priors["high"] - priors["low"]
    w0_n = np.array([normal_normal_shrinkage_weight(sigma0, sigma, n) for n in months])
    gap_predicted = w0_n * gap_initial

    axR.plot(months, gap_observed, "o-", color="crimson", lw=2,
             label="Observed gap $|\\mu_n^{(B)}-\\mu_n^{(A)}|$")
    axR.plot(months, gap_predicted, "s--", color="steelblue", lw=1.5,
             label=r"Theoretical $w_0(n) \cdot |\mu_0^{(B)}-\mu_0^{(A)}|$")
    axR.axhline(gap_initial, color="black", ls=":", alpha=0.4,
                label=f"Initial gap = R$ {gap_initial:,.0f}")
    axR.set_xlabel("Month $n$")
    axR.set_ylabel(r"Disagreement (R\$)")
    axR.set_title(r"Prior disagreement decays at rate $w_0(n)$")
    axR.set_xticks(months)
    axR.set_yscale("log")
    axR.ticklabel_format(style="plain", axis="x")
    axR.grid(alpha=0.3, which="both")
    axR.legend(fontsize=8.5, loc="upper right")

    fig.suptitle(
        f"Prior sensitivity — Normal-Normal, common $\\sigma_0={sigma0:,.0f}$, "
        f"$\\sigma={sigma:,.0f}$, seed={SEED}",
        fontsize=11,
    )

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=300)
    print(f"Saved {FIG_PATH.relative_to(REPO_ROOT)}  ({FIG_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
