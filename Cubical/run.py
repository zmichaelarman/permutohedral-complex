# command-line driver. runs the cell-percolation experiment at one or more
# lattice sizes and saves each result with a small metadata file. examples:
#   python run.py 3 4 6 8 9            (dimension 4 by default)
#   python run.py --dim 2 10 20 40

import sys
import json
import time
import pathlib
import numpy as np
from math import comb

from percolation import cell_percolation

DEFAULT_DIM = 4
DEFAULT_TRIALS = 5000
# the lattice sizes used for each dimension when none are given on the command line
DEFAULT_SCALES = {2: [10, 20, 40, 80], 4: [3, 4, 6, 8, 9]}


# run one lattice size and write its output folder
def run_scale(dim, L, homology, trials, seed=None):
    # fixed seed per (dimension, size) so runs are reproducible
    if seed is None:
        seed = 1000 * dim + L
    t0 = time.time()
    perc, giants, occ, ncells = cell_percolation(dim, L, trials, homology=homology,
                                                 seed=seed, verify=True)
    out = pathlib.Path(f"output/statistics/d{dim}_L{L}")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "percentages", perc)     # birth density of each giant, per trial
    np.save(out / "giants", giants)        # how many giants present, per trial
    np.save(out / "occupation", occ)       # fraction of cells occupied, per trial
    meta = {
        "dimension": dim, "homology": homology, "scale": L,
        "rank": comb(dim, homology), "iterations": trials, "cells": ncells,
        "model": "Bernoulli cell percolation",
        "complex": "Cubical", "seconds": round(time.time() - t0, 1),
    }
    json.dump(meta, open(out / "metadata.json", "w"), indent=2)
    print(f"L={L}  birth {perc[perc>0].mean():.4f}  giants {giants.mean():.2f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    # read the command line: flags start with --, bare numbers are lattice sizes
    args = sys.argv[1:]
    dim, trials, homology, scales = DEFAULT_DIM, DEFAULT_TRIALS, None, []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--dim":
            dim = int(args[i + 1]); i += 2
        elif a == "--homology":
            homology = int(args[i + 1]); i += 2
        elif a == "--trials":
            trials = int(args[i + 1]); i += 2
        else:
            scales.append(int(a)); i += 1
    if homology is None:
        homology = dim // 2          # middle dimension by default
    if not scales:
        scales = DEFAULT_SCALES.get(dim, [3, 4, 6, 8, 9])
    for L in scales:
        run_scale(dim, L, homology, trials)
