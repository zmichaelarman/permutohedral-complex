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
    occupation = np.zeros(trials, dtype=float)

    for t in range(trials):
        uniform = np.random.uniform(size=Nv)
        include = np.nonzero(uniform < 1/2)[0]
        exclude = np.nonzero(~(uniform < 1/2))[0]
        m = include.shape[0]

        occupation[t] = m/Nv

        np.random.shuffle(include)

        filtration = np.arange(Nv)
        filtration[:m] = include
        filtration[m:] = exclude

        # not actually sure what this is doing?
        vals = filtration[S_pad].max(axis=1).astype(float)
        for s, fv in zip(all_s, vals): st.assign_filtration(s, fv)

        # compute the persistence
        st.compute_persistence(homology_coeff_field=2, persistence_dim_max=True)
        pairs = st.persistence_intervals_in_dimension(homology)
        immortal = np.sum(filtration[np.where(np.isinf(pairs[:,1]))[0]] < m)

        giants[t] = immortal

    return np.array([]), giants, occupation, Nv

