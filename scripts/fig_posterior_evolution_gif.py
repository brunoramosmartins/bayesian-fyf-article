"""Generate ``figures/posterior_evolution.gif``.

Animated posterior evolution across the FYF cycle: each frame shows
the posterior density at month n, with the actual observation marked
on the x-axis. Built with matplotlib's PillowWriter so the output is
a small portable GIF suitable for LinkedIn / Medium (target < 5 MB).

Frames: month 0 (prior) plus months 1..12 (post-update). 13 frames
at ~1.2 s each ≈ a 16-second loop.

Usage
-----
From the repository root::

    python scripts/fig_posterior_evolution_gif.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy import stats

from src.conjugate import NormalNormalUpdater, NormalPosterior
from src.updating import SequentialUpdater

REPO_ROOT = Path(__file__).resolve().parent.parent
GIF_PATH = REPO_ROOT / "figures" / "posterior_evolution.gif"
SEED = 20260920


def main() -> None:
    mu0, sigma0, sigma = 1_050_000.0, 150_000.0, 80_000.0
    rng = np.random.default_rng(SEED)
    actuals = rng.normal(1_080_000.0, sigma, size=12)

    upd = NormalNormalUpdater(mu0=mu0, sigma0_sq=sigma0**2, sigma_sq=sigma**2)
    seq = SequentialUpdater(upd)

    posteriors: list[NormalPosterior] = [
        NormalPosterior(mu=mu0, sigma_sq=sigma0**2)
    ]
    for x in actuals:
        post = seq.feed(float(x))
        assert isinstance(post, NormalPosterior)
        posteriors.append(post)

    grid = np.linspace(mu0 - 4 * sigma0, mu0 + 4 * sigma0, 600)

    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    line_post, = ax.plot([], [], color="crimson", lw=2.4)
    fill = ax.fill_between(grid, np.zeros_like(grid), np.zeros_like(grid),
                           color="crimson", alpha=0.18)
    ax.plot(grid, stats.norm(mu0, sigma0).pdf(grid),
            color="steelblue", lw=1.6, ls=":",
            label=fr"Prior $N({mu0:,.0f},\ {sigma0:,.0f})$")
    actuals_dot = ax.scatter([], [], color="goldenrod", s=42, zorder=5,
                             label="latest actual")
    ax.axvline(mu0, color="steelblue", ls=":", alpha=0.4)

    title_obj = ax.set_title("")
    ax.set_xlabel(r"$\theta$ — mean monthly cost (R\$)")
    ax.set_ylabel("posterior density")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.ticklabel_format(style="plain", axis="x")
    ax.grid(alpha=0.3)

    pdfs = np.array(
        [stats.norm(p.mean(), p.std()).pdf(grid) for p in posteriors]
    )
    ax.set_ylim(0, pdfs.max() * 1.10)
    ax.set_xlim(grid.min(), grid.max())

    def update(frame: int):
        nonlocal fill
        post = posteriors[frame]
        pdf = pdfs[frame]
        line_post.set_data(grid, pdf)
        fill.remove()
        fill = ax.fill_between(grid, np.zeros_like(grid), pdf,
                               color="crimson", alpha=0.18)
        if frame == 0:
            actuals_dot.set_offsets(np.empty((0, 2)))
            title_obj.set_text(
                f"Month 0 (prior)  —  $\\mu={post.mean():,.0f}$, "
                f"$\\sigma={post.std():,.0f}$"
            )
        else:
            actuals_dot.set_offsets([[actuals[frame - 1], 0.0]])
            title_obj.set_text(
                f"Month {frame}  —  actual={actuals[frame - 1]:,.0f}, "
                f"$\\mu_n={post.mean():,.0f}$, $\\sigma_n={post.std():,.0f}$"
            )
        return line_post, fill, actuals_dot, title_obj

    anim = FuncAnimation(fig, update, frames=len(posteriors), interval=1200, blit=False)

    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=1)  # 1 fps so each frame is ~1 s
    anim.save(GIF_PATH, writer=writer, dpi=110)
    print(
        f"Saved {GIF_PATH.relative_to(REPO_ROOT)}  "
        f"({GIF_PATH.stat().st_size / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    main()
