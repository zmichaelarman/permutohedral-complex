# plot styling shared by the figure scripts. nothing here computes results; it
# just keeps colors, sizes, and a couple of small helper functions in one place.

from scipy.stats import gaussian_kde
import numpy as np
import seaborn as sns


# a plain dict whose keys can also be read as attributes (so cfg.foo works as well
# as cfg["foo"]), just to keep the settings tidy.
class Bunch(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


# overall figure settings: latex text, size, resolution
plot = Bunch()
plot.rcParams = {"text.usetex": True, "font.family": "Helvetica"}
plot.figsize = (5, 3)
plot.dpi = 1200
plot.savefig = dict(dpi=plot.dpi, bbox_inches="tight")

histograms = Bunch()

# settings for the birth-density curves
histograms.occupation = Bunch()
histograms.occupation.rcParams = plot.rcParams
histograms.occupation.figsize = plot.figsize
histograms.occupation.colors = lambda n: sns.light_palette("seagreen", n_colors=n)               # n shades of green
histograms.occupation.histograms = lambda X, rank: [X[:, r][X[:, r] > 0] for r in range(rank)]   # drop the zero (never-appeared) entries
histograms.occupation.pdfs = lambda hs, X: np.array([gaussian_kde(h)(X) for h in hs])            # smooth each histogram into a curve
histograms.occupation.xlim = (0.48, 0.51)
histograms.occupation.savefig = plot.savefig

# settings for the bar chart of "how many giants"
histograms.rank = Bunch()
histograms.rank.rcParams = plot.rcParams
histograms.rank.figsize = plot.figsize
histograms.rank.bar = dict(
    width=0.65, edgecolor="k", linewidth=3 / 4,
    facecolor=histograms.occupation.colors(6)[-1], alpha=3 / 4,
)
histograms.rank.xlim = lambda r: (-1, r + 1)
histograms.rank.savefig = plot.savefig
