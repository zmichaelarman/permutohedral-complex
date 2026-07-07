# runs the experiment: site percolation on the permutohedral complex, with gudhi
# doing the homology. site percolation means we occupy the lattice corners in a
# random order, and a simplex counts as filled once all of its corners are
# occupied. we watch for giant cycles - holes that wrap all the way around the
# torus and so can never be filled in. there are exactly C(d, k) of them for the
# homology dimension k we track, and the interesting thing is the occupation
# density at which each one first appears.

import numpy as np
import gudhi as gd
from math import comb
from itertools import combinations

from permutohedral import delaunay_torus


# build the gudhi simplex tree (its homology data structure) and, if asked, check
# the finished torus has the right number of wrapping holes.
def build_tree(d, N, homology, verify=True, orientation="rhombic"):
    vertices, simplices, _ = delaunay_torus(d, N, orientation)
    Nv = len(vertices)
    st = gd.SimplexTree()
    # we only need cells up to one dimension above the homology we track, so
    # insert all faces of that size rather than the whole top simplex.
    W = homology + 2
    for s in simplices:
        for face in combinations(s, W):
            st.insert(list(face))
    # the finished torus must have exactly C(d, homology) wrapping cycles. if it
    # doesn't, the construction is wrong (or the torus is too small), so stop.
    if verify:
        st.compute_persistence(persistence_dim_max=True)
        betti = st.betti_numbers()
        expected = comb(d, homology)
        assert len(betti) > homology and betti[homology] == expected, \
            f"verify failed: betti={betti}, expected H_{homology}={expected}"
    return st, Nv


# one full experiment: repeat the random-fill many times and record, for each
# run, the occupation density at which each giant cycle was born.
def site_percolation(d, N, trials, homology=None, seed=None, verify=True, orientation="rhombic"):
    if homology is None:
        homology = d // 2          # default to the middle dimension, the p=1/2 case
    rank = comb(d, homology)        # how many giant cycles to expect
    st, Nv = build_tree(d, N, homology, verify=verify, orientation=orientation)

    # flatten every simplex to one fixed width so the filtration below is a single
    # fast numpy op. short simplices get padded by repeating their first corner,
    # which doesn't change the max we take from each row.
    all_s = [list(s) for s, _ in st.get_simplices()]
    W = homology + 2
    S_pad = np.array([s + [s[0]] * (W - len(s)) for s in all_s], dtype=np.int64)

    rng = np.random.default_rng(seed)
    percentages = np.zeros((trials, rank))
    giants = np.zeros(trials, dtype=int)
    occupation = np.zeros(trials)      # realized heads fraction, one per trial
    for t in range(trials):
        # bernoulli occupation: flip a fair coin per corner. heads (occupied) take
        # the low ranks, tails (empty) the high ranks, random order inside each
        # group. so the filtration fills exactly the occupied set first, then the
        # rest -- this lets the persistence below tell a real wrapping cycle (still
        # open after every corner is added) from one only closed up by empty corners.
        heads = rng.random(Nv) < 0.5
        n_heads = int(heads.sum())
        order = rng.permutation(Nv)                 # random order over all corners
        # stable sort keeps heads (key False) ahead of tails, random within each block
        blocks = order[np.argsort(~heads[order], kind="stable")]
        vrank = np.empty(Nv, dtype=np.int64)
        vrank[blocks] = np.arange(Nv)               # vrank[corner] = its fill slot
        # a simplex fills when its LAST corner fills, i.e. the largest rank among
        # its corners. that number is its slot in the filtration order.
        vals = vrank[S_pad].max(axis=1).astype(float)
        for s, fv in zip(all_s, vals):
            st.assign_filtration(s, fv)
        st.compute_persistence(homology_coeff_field=2, persistence_dim_max=True)
        # giant cycles are the ones that never fill in (death = infinity). convert
        # each one's birth slot into a density: how full the lattice was when it
        # appeared.
        ess = sorted((b + 1) / Nv for (b, dth) in
                     st.persistence_intervals_in_dimension(homology) if dth == float("inf"))
        k = min(len(ess), rank)
        percentages[t, :k] = ess[:k]
        # a giant counts for this bernoulli sample only if it was already open once
        # just the occupied (heads) corners were present -- i.e. born within the
        # first n_heads fill slots (birth density <= n_heads / Nv).
        giants[t] = int(np.sum(np.asarray(ess) <= n_heads / Nv))
        occupation[t] = n_heads / Nv
    return percentages, giants, occupation, Nv
