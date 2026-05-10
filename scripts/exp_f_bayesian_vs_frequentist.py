"""Experiment F — frequentist confidence interval vs Bayesian credible
interval.

Top panel: 100 simulated experiments with 30 observations each, drawn
from ``N(θ, σ²)`` with known σ. For each experiment we compute the
classical 95 % confidence interval; intervals that *do not* cover the
true mean are highlighted in red. Coverage rate over the 100 runs is
reported.

Bottom panel: a single experiment with the corresponding 95 %
credible interval under a weakly informative prior. The frequentist
statement is *about the procedure* (5 % of intervals miss); the
Bayesian statement is *about θ given the observed data*. Both
intervals can numerically agree, but their interpretation differs.

Output: ``figures/exp_f_bayesian_vs_frequentist.png`` at 300 DPI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.conjugate import NormalNormalUpdater

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = REPO_ROOT / "figures" / "exp_f_bayesian_vs_frequentist.png"
SEED = 20260914


def main() -> None:
    rng = np.random.default_rng(SEED)
    n_runs = 100
    n_obs = 30
    theta_true = 1_080_000.0
    sigma = 80_000.0

    half_width = 1.96 * sigma / np.sqrt(n_obs)

    means = np.empty(n_runs)
    misses = np.zeros(n_runs, dtype=bool)
    for i in range(n_runs):
        x = rng.normal(theta_true, sigma, size=n_obs)
        means[i] = float(x.mean())
        lo = means[i] - half_width
        hi = means[i] + half_width
        misses[i] = not (lo <= theta_true <= hi)

    coverage = 1.0 - misses.mean()

    # Pick one experiment for the credible interval panel.
    rng_one = np.random.default_rng(SEED + 17)
    x_one = rng_one.normal(theta_true, sigma, size=n_obs)
    upd = NormalNormalUpdater(
        mu0=1_050_000.0, sigma0_sq=300_000.0**2, sigma_sq=sigma**2,
    )
    post = upd.update(x_one)
    cred_lo, cred_hi = post.credible_interval(0.95)
    freq_lo = float(x_one.mean()) - half_width
    freq_hi = float(x_one.mean()) + half_width

    fig, (axTop, axBot) = plt.subplots(
        2, 1, figsize=(11.5, 7.4), constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # ----- top: 100 frequentist CIs ---------------------------------------
    for i in range(n_runs):
        color = "crimson" if misses[i] else "steelblue"
        lo = means[i] - half_width
        hi = means[i] + half_width
        axTop.plot([lo, hi], [i, i], color=color, lw=1.0, alpha=0.85)
        axTop.plot(means[i], i, "o", color=color, ms=2.5)
    axTop.axvline(theta_true, color="black", lw=1.4,
                  label=fr"True $\theta_\star={theta_true:,.0f}$")
    axTop.set_yticks([])
    axTop.set_xlabel(r"$\bar x \pm 1.96\,\sigma/\sqrt{n}$ (R\$)")
    axTop.set_title(
        f"Frequentist 95 % CIs over {n_runs} simulated experiments — "
        f"coverage = {coverage:.0%}"
    )
    n_misses = int(misses.sum())
    axTop.text(
        0.985, 0.05,
        f"Red intervals miss the truth\n({n_misses} / {n_runs}, "
        f"≈ {(1-coverage)*100:.0f} %)",
        transform=axTop.transAxes, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.6"),
        fontsize=8.5,
    )
    axTop.ticklabel_format(style="plain", axis="x")
    axTop.legend(fontsize=8.5, loc="upper right")
    axTop.grid(alpha=0.3, axis="x")

    # ----- bottom: one experiment, frequentist vs Bayesian ----------------
    grid = np.linspace(theta_true - 5 * post.std(), theta_true + 5 * post.std(), 600)
    axBot.plot(grid, stats.norm(post.mean(), post.std()).pdf(grid),
               color="crimson", lw=2,
               label=fr"Posterior  $N({post.mean():,.0f}, {post.std():,.0f}^2)$")
    axBot.axvspan(cred_lo, cred_hi, color="crimson", alpha=0.18,
                  label=f"95 % credible: [{cred_lo:,.0f}, {cred_hi:,.0f}]")
    axBot.axvspan(freq_lo, freq_hi, color="steelblue", alpha=0.10,
                  label=f"95 % frequentist:  [{freq_lo:,.0f}, {freq_hi:,.0f}]")
    axBot.axvline(theta_true, color="black", lw=1.4,
                  label=fr"True $\theta_\star={theta_true:,.0f}$")
    axBot.set_xlabel(r"$\theta$ (R\$)")
    axBot.set_ylabel("posterior density")
    axBot.set_title(
        "Single experiment — frequentist CI vs Bayesian credible interval "
        "(numerically close, interpretively different)"
    )
    axBot.ticklabel_format(style="plain", axis="x")
    axBot.legend(fontsize=8.5, loc="upper right")
    axBot.grid(alpha=0.3)

    fig.suptitle(
        f"Experiment F — Bayesian credible interval vs frequentist confidence interval "
        f"(seed={SEED})",
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
