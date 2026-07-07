# reads the saved cell-percolation results and makes the per-size figures: a
# birth-density curve for each giant cycle (occupation), a bar chart of how many
# giants were present at half-full (rank), and one combined plot per dimension
# overlaying the curves across all sizes (unified). run this after run.py.

import json
import pathlib
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")           # render to files, no window
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from config import histograms

plt.rcParams.update(**histograms.occupation.rcParams)

STATS = pathlib.Path("output/statistics")
OUT = pathlib.Path("output/figures")
OUT.mkdir(parents=True, exist_ok=True)
X = np.linspace(0, 1, 2048)     # the x-axis grid (occupation 0 to 1) for the curves


# load every output folder and key the results by (dimension, size). folders
# without a finished metadata file are skipped.
def _load():
    runs = {}
    for d in sorted(STATS.glob("*")):
        if not d.is_dir() or not (d / "metadata.json").exists():
            continue
        try:
            meta = json.load(open(d / "metadata.json"))
            perc = np.load(d / "percentages.npy")
            giants = np.load(d / "giants.npy")
        except Exception:
            continue
        runs[(int(meta.get("dimension", 4)), meta["scale"])] = (meta, perc, giants)
    return runs


# index of a curve's tallest point, but only within the x-range we'll draw, so the
# little rank label lands on the visible peak.
def _peak_in_window(Z, xlim):
    win = (X >= xlim[0]) & (X <= xlim[1])
    return int(np.argmax(np.where(win, Z, -np.inf)))


# pick a sensible x-range from the data: tight around where the births sit, with a
# little padding each side, so the plot isn't mostly empty.
def _auto_xlim(births_or_perc, pad=0.12):
    b = np.asarray(births_or_perc)
    b = b[b > 0]
    lo, hi = float(b.min()), float(b.max())
    span = max(hi - lo, 1e-3)
    return (max(0.0, lo - pad * span), min(1.0, hi + pad * span))


# for one run: a smooth curve per giant cycle showing the spread of densities at
# which it was born. dividing by N keeps heights comparable across runs. the small
# numbers label which cycle (1st, 2nd, ...) each curve is.
def occupation_figure(dim, meta, perc):
    L, RANK, N = meta["scale"], meta["rank"], meta["iterations"]
    xlim = _auto_xlim(perc)
    colors = histograms.occupation.colors(RANK)
    fig, ax = plt.subplots(figsize=histograms.occupation.figsize)
    for i in range(RANK):
        h = perc[:, i][perc[:, i] > 0]      # this cycle's births, ignoring trials where it never appeared
        if len(h) < 2:
            continue
        Z = gaussian_kde(h)(X) / N           # smooth the histogram into a curve
        ax.plot(X, Z, lw=1 / 2, alpha=1 / 2, color=colors[i])
        ax.fill_between(X, Z, alpha=1 / 2, color=colors[i], lw=0)
        mm = _peak_in_window(Z, xlim)
        ax.text(X[mm], Z[mm], rf"${i+1}$", fontsize=4, alpha=3 / 4, ha="right", va="top")
    ax.set_xlim(*xlim)
    ax.set_xlabel(r"Cell occupation density at giant-cycle birth")
    ax.set_ylabel(r"Density")
    fig.savefig(OUT / f"occupation.d{dim}.L{L}.png", **histograms.occupation.savefig)
    plt.close(fig)


# for one run: bar chart of how often 0, 1, 2, ... giants were present at half-full.
def rank_figure(dim, meta, giants):
    L, RANK, N = meta["scale"], meta["rank"], meta["iterations"]
    vals, counts = np.unique(giants, return_counts=True)
    Y = np.zeros(RANK + 1)
    for v, c in zip(vals, counts):
        Y[v] = c / N                         # counts -> frequency
    fig, ax = plt.subplots(figsize=histograms.rank.figsize)
    ax.bar(np.arange(RANK + 1), Y, **histograms.rank.bar)
    ax.set_xticks(np.arange(RANK + 1))
    ax.set_xticklabels([f"${x}$" for x in range(RANK + 1)])
    ax.tick_params(bottom=False)
    ax.set_xlim(*histograms.rank.xlim(RANK))
    ax.set_xlabel(r"Number of giant cycles at $p=1/2$")
    ax.set_ylabel(r"Frequency")
    fig.savefig(OUT / f"rank.d{dim}.L{L}.png", **histograms.rank.savefig)
    plt.close(fig)


# one plot per dimension: overlay the birth-density curves from every size,
# colored by size, so you can watch them sharpen as the lattice grows.
def unified_figure(dim, runs_for_dim):
    xlim = _auto_xlim(np.concatenate([perc[perc > 0] for _, perc, _ in runs_for_dim]))
    # collect one smoothed curve per (size, cycle), averaging if a size repeats
    agg = defaultdict(list)
    RANK = homology = None
    for meta, perc, _ in runs_for_dim:
        RANK, homology, N = meta["rank"], meta["homology"], meta["iterations"]
        for i in range(RANK):
            h = perc[:, i][perc[:, i] > 0]
            if len(h) >= 2:
                agg[(meta["scale"], i)].append(gaussian_kde(h)(X) / N)
    scales = sorted({s for (s, _) in agg})
    if not scales:
        return
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(scales)))
    fig, ax = plt.subplots(figsize=(12, 7))
    for ci, L in enumerate(scales):
        for i in range(RANK):
            if (L, i) not in agg:
                continue
            Z = np.mean(agg[(L, i)], axis=0)
            # later cycles drawn fainter and thinner so the plot stays readable
            alpha = max(0.15, 1.0 - i * 0.15)
            lw = 1.5 if i == 0 else 0.7
            label = rf"$L={L},\ \mathrm{{Rank}}\ {i+1}$" if i in (0, RANK - 1) else None
            ax.plot(X, Z, color=colors[ci], alpha=alpha, lw=lw, label=label)
            ax.fill_between(X, Z, color=colors[ci], alpha=alpha * 0.2, lw=0)
            mm = _peak_in_window(Z, xlim)
            ax.text(X[mm], Z[mm], rf"${i+1}$", fontsize=7, color=colors[ci],
                    alpha=alpha, ha="center", va="bottom")
    ax.set_title(rf"Giant $H_{{{homology}}}$ birth density, cubical $Z^{{{dim}}}$ "
                 rf"Bernoulli cell percolation")
    ax.set_xlabel(r"Cell occupation probability $p$")
    ax.set_ylabel(r"Probability density $f(p)$")
    ax.set_xlim(*xlim)
    ax.legend(loc="upper right", fontsize="small", ncol=max(1, len(scales)))
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.savefig(OUT / f"unified.d{dim}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    runs = _load()
    if not runs:
        raise SystemExit("no statistics found; run run.py first")
    # per-run figures, then one combined figure per dimension
    by_dim = defaultdict(list)
    for (dim, scale), (meta, perc, giants) in sorted(runs.items()):
        occupation_figure(dim, meta, perc)
        rank_figure(dim, meta, giants)
        by_dim[dim].append((meta, perc, giants))
    for dim, lst in sorted(by_dim.items()):
        unified_figure(dim, lst)
    print("figures written")
