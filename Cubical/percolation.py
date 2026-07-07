# runs the experiment on the cubic lattice: cell percolation. unlike the
# permutohedral side (which occupies corners), here we occupy the CELLS
# themselves - the squares/plaquettes of the middle dimension - at random. ATEAMS
# does the homology with its fast PHAT engine. we count giant cycles, the holes
# that wrap all the way around the torus; there are C(d, k) of them in the
# dimension k we track.

import numpy as np
from math import comb

from ateams.models import Bernoulli
from ateams import Chain
from ateams.arithmetic import ComputePersistencePairs

from cubical import build


# sanity check: with every cell present, the torus must have exactly
# C(d, homology) wrapping cycles. ComputePersistencePairs pairs up the cells that
# cancel each other out; the ones left unpaired in our dimension are the wrapping
# cycles. stop if that count is wrong.
def _verify_full(C, homology):
    d = C.dimension
    rank = comb(d, homology)
    low, high = int(C.breaks[homology]), int(C.breaks[homology + 1])
    cellCount = int(C.tranches[d][1])
    times = set(range(cellCount))
    filt = np.arange(cellCount)
    births, deaths = zip(*ComputePersistencePairs(C.matrices.full, filt, homology, C.breaks))
    ess = [e for e in (times - (set(births) | set(deaths))) if low <= e < high]
    assert len(ess) == rank, f"verify failed: H_{homology}={len(ess)}, expected {rank}"


# one full experiment: repeat the random-occupation run `trials` times and record
# how many giants were present each time and at what density they appeared.
def cell_percolation(d, N, trials, homology=None, seed=None, verify=True):
    if homology is None:
        homology = d // 2           # middle dimension, the p=1/2 case
    C = build(d, N)
    rank = comb(d, homology)
    # the cells of our dimension occupy the index range [low, low+ncells)
    low = int(C.breaks[homology])
    ncells = int(C.breaks[homology + 1]) - low
    if verify:
        _verify_full(C, homology)

    # the Bernoulli model occupies each middle-dimension cell independently with
    # probability 1/2 and reports the giants present in that configuration
    model = Bernoulli(C, dimension=homology)
    if seed is not None:
        model.RNG = np.random.default_rng(seed)

    percentages = np.zeros((trials, rank))
    giants = np.zeros(trials, dtype=int)
    occupation = np.zeros(trials)
    # Chain runs the model `trials` times; each step hands back the occupied cells
    # and the positions of the giants for that run
    for t, (occ, gpos) in enumerate(Chain(model, steps=trials)):
        gpos = np.asarray(gpos, dtype=float)
        k = min(gpos.shape[0], rank)
        occupation[t] = float(occ.sum()) / occ.shape[0]     # fraction of cells occupied (about 1/2)
        giants[t] = int(gpos.shape[0])                       # how many giants in this run
        if k:
            # turn each giant's position into the density (fraction of cells) at
            # which it appeared. a giant at filtration position e appears when
            # e - low + 1 cells are occupied, hence the +1 (matching the
            # permutohedral side and extra_figures.py; without it every birth
            # reads low by exactly 1/ncells).
            percentages[t, :k] = np.sort((gpos - low + 1) / ncells)[:k]
    return percentages, giants, occupation, ncells
