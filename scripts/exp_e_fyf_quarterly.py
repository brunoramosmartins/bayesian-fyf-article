"""Experiment E — full annual FYF cycle with quarterly review boxes.

Combines into a single article-grade figure: posterior trajectory +
year-end forecast + four FYF review summaries (Q1, Q2, Q3, Q4) with
posterior mean, σ, and P(over-budget) at each.

Output: ``figures/exp_e_fyf_quarterly.png`` at 300 DPI.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = REPO_ROOT / "figures" / "exp_e_fyf_quarterly.png"
SEED = 20260913


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
    quarterlies = model.quarterly_reviews()

    months = np.arange(1, 13)
    means = np.array([r.posterior.mean() for r in reviews])
    stds = np.array([r.posterior.std() for r in reviews])

    fig, (axTop, axBot) = plt.subplots(
        2, 1, figsize=(11.5, 7.4), constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # ----- Top: posterior trajectory + actuals + quarterly markers -------
    axTop.fill_between(
        months, means - 1.96 * stds, means + 1.96 * stds,
        color="steelblue", alpha=0.20, label="95 % credible interval",
    )
    axTop.plot(months, means, "o-", color="steelblue", lw=2,
               label=r"Posterior mean $\mu_n$")
    axTop.scatter(months, actuals, color="dimgray", s=22, alpha=0.7,
                  label="Monthly actuals", zorder=3)
    axTop.axhline(config.prior_mean, color="black", ls="--", alpha=0.4,
                  label=fr"Prior $\mu_0={config.prior_mean:,.0f}$")
    for q_end, qr in zip(QUARTER_END_MONTHS, quarterlies, strict=True):
        axTop.axvline(q_end + 0.5, color="goldenrod", ls=":", alpha=0.5)
        axTop.annotate(
            f"FYF #{qr.quarter}\nμ={qr.posterior.mean():,.0f}\n"
            f"σ={qr.posterior.std():,.0f}\n"
            f"P(T>B)={qr.p_over_budget:.0%}"
            if qr.p_over_budget is not None
            else f"FYF #{qr.quarter}\nμ={qr.posterior.mean():,.0f}\n"
                 f"σ={qr.posterior.std():,.0f}",
            xy=(q_end, means[q_end - 1]),
            xytext=(q_end - 0.3, axTop.get_ylim()[0] * 1.005 if False else means.min() - 35_000),
            fontsize=8, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.30", fc="lemonchiffon", ec="0.5"),
        )
    axTop.set_xticks(months)
    axTop.set_xlabel("Month $n$")
    axTop.set_ylabel(r"Mean monthly cost (R\$)")
    axTop.set_title("Posterior trajectory with quarterly FYF reviews")
    axTop.ticklabel_format(style="plain", axis="y")
    axTop.grid(alpha=0.3)
    axTop.legend(fontsize=8.5, loc="upper right")

    # ----- Bottom: year-end forecast and P(over-budget) -------------------
    fc_months: list[int] = []
    fc_means: list[float] = []
    fc_los: list[float] = []
    fc_his: list[float] = []
    for r in reviews:
        if r.forecast is None:
            continue
        fc_months.append(r.month)
        fc_means.append(r.forecast.mean)
        lo, hi = r.forecast.credible_interval(0.95)
        fc_los.append(lo)
        fc_his.append(hi)
    fc_months_arr = np.array(fc_months)
    fc_means_arr = np.array(fc_means)
    fc_los_arr = np.array(fc_los)
    fc_his_arr = np.array(fc_his)

    axBot.fill_between(fc_months_arr, fc_los_arr, fc_his_arr,
                       color="crimson", alpha=0.18,
                       label="Year-end 95 % predictive interval")
    axBot.plot(fc_months_arr, fc_means_arr, "o-", color="crimson", lw=2,
               label=r"$\mathbb{E}[T \mid x_{1:n}]$")
    if config.budget_ceiling is not None:
        axBot.axhline(config.budget_ceiling, color="goldenrod", ls="--",
                      label=fr"Budget $B={config.budget_ceiling:,.0f}$")
    for q_end in QUARTER_END_MONTHS:
        axBot.axvline(q_end + 0.5, color="goldenrod", ls=":", alpha=0.5)
    axBot.set_xticks(months)
    axBot.set_xlabel("Month $n$")
    axBot.set_ylabel(r"Year-end total $T$ (R\$)")
    axBot.set_title("Year-end forecast tightens around the data")
    axBot.ticklabel_format(style="plain", axis="y")
    axBot.grid(alpha=0.3)
    axBot.legend(fontsize=8.5, loc="best")

    fig.suptitle(
        f"Experiment E — full FYF annual cycle  (true θ=1,080,000, seed={SEED})",
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
