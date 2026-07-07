# builds the shape we run percolation on: the triangulated version (delaunay
# triangulation) of the A*_d lattice, wrapped around on itself into a
# d-dimensional torus. the lattice points are the corners, and the cells are
# simplices (triangles in 2d, tetrahedra in 3d, and so on up). every top simplex
# follows one fixed pattern, so we can list them all with a loop instead of
# calling a geometry library.
#
# there are two ways to wrap the lattice into a torus:
#   rhombic - the lattice's own slanted box; wrapping is just coordinates mod N.
#   square  - an axis-aligned (rectangular/cubic) box, which takes more work to
#             wrap. it matches the cubic lattice's box so the two can be compared.

import numpy as np
from fractions import Fraction
from itertools import permutations, product
from collections import deque


# the d+1 steps you can take to walk around one simplex: the d unit-axis steps
# (1,0,..), (0,1,..), ... plus one step that goes back by one in every direction.
# those d+1 steps add up to zero, which is the property that lets the simplices
# fit together and tile the lattice.
def _direction_deltas(d):
    deltas = [np.eye(d, dtype=int)[k] for k in range(d)]   # the d unit-axis steps
    deltas.append(-np.ones(d, dtype=int))                  # the "back by one everywhere" step
    return deltas


# for the square wrapping we need an axis-aligned box built from lattice vectors
# that sit at right angles to each other (under the lattice's own slanted
# geometry). these are worked out by hand per dimension and checked at runtime by
# the betti-number test in percolation.py. d=3 happens to come out a true cube.
# only 2, 3, 4 are filled in, so the square option only works in those dimensions.
_SQUARE_GEN = {
    2: [[1, -1], [1, 1]],
    3: [[-1, -1, 0], [-1, 0, -1], [0, -1, -1]],
    4: [[1, -1, 0, 0], [1, 1, -1, -1], [0, 0, 1, -1], [1, 1, 1, 1]],
}


# how many times to tile the box along each axis to get a torus that's about as
# wide as the requested size N in every direction. axes with shorter steps get
# tiled more times. clamped to at least 2 so no direction is too thin to hold the
# topology.
def square_repetitions(d, N):
    B = np.array(_SQUARE_GEN[d], dtype=float)
    # length-squared of each box axis under the lattice's slanted geometry
    Q = np.diag(B @ (np.eye(d) - 1.0 / (d + 1)) @ B.T)
    # pick the repeat counts so every axis ends up about the same physical length
    return np.maximum(2, np.rint(N * np.sqrt(d / ((d + 1) * Q))).astype(int))


# exact integer inverse helper for the square wrapping. given the box matrix M it
# returns (adjugate, determinant), where M times the adjugate equals det times the
# identity. done with fractions so there's zero rounding error, which matters
# because we use it to decide which lattice points become the same point after
# wrapping. this is plain gaussian elimination that also tracks the determinant.
def _adjugate(M):
    d = len(M)
    A = [[Fraction(int(M[i][j])) for j in range(d)] +
         [Fraction(int(i == k)) for k in range(d)] for i in range(d)]
    det = Fraction(1)
    for c in range(d):
        piv = next(r for r in range(c, d) if A[r][c] != 0)
        if piv != c:
            A[c], A[piv] = A[piv], A[c]
            det = -det
        det *= A[c][c]
        f = Fraction(1) / A[c][c]
        A[c] = [x * f for x in A[c]]
        for r in range(d):
            if r != c and A[r][c] != 0:
                g = A[r][c]
                A[r] = [A[r][j] - g * A[c][j] for j in range(2 * d)]
    inv = [[A[i][d + j] for j in range(d)] for i in range(d)]
    deti = int(det)
    adj = np.array([[int(deti * inv[i][j]) for j in range(d)] for i in range(d)], dtype=np.int64)
    return adj, deti


# builds the function that "wraps" any lattice point onto the square torus, i.e.
# returns the one standard copy of it that lives inside the box. it's all integer
# math (floor-divide by the determinant) so the wrapping is exact and never
# drifts. flip the sign if the determinant came out negative so the floor divide
# behaves.
def _square_reducer(M):
    M = np.array(M, dtype=np.int64)
    adj, det = _adjugate(M)
    if det < 0:
        adj, det = -adj, -det
    Mt, adjT = M.T.copy(), adj.T.copy()

    # subtract off whole copies of the box until the point lands back inside it.
    # handles a whole array of points at once.
    def reduce(P):
        return P - ((P @ adjT) // det) @ Mt

    return reduce, det


# build every simplex for the natural (slanted) torus. visit every lattice point
# in the N-by-N-by-... box, and from each one trace out simplices by taking the
# step directions in every possible order. wrapping is just "mod N".
def _delaunay_rhombic(d, N):
    deltas = _direction_deltas(d)
    raw = set()
    for base in product(range(N), repeat=d):        # every lattice point in the box
        L = np.array(base, dtype=int)
        for perm in permutations(range(d + 1)):     # every ordering of the step directions
            verts, c = [tuple(L % N)], L.copy()
            for step in perm[:d]:                   # take the first d of those steps
                c = c + deltas[step]                # walk one step
                verts.append(tuple(c % N))          # record the corner we landed on, wrapped
            s = frozenset(verts)
            # keep it only if it has d+1 distinct corners. on a small torus a walk
            # can wrap back onto itself and collapse, and those are thrown out.
            if len(s) == d + 1:
                raw.add(s)                          # the set also drops repeated simplices
    return raw


# same simplex-tracing idea, but wrapped onto the axis-aligned box. two changes:
# we can't loop a clean N-by-N grid, so first we find all the distinct lattice
# points by flood fill, and wrapping uses the exact integer reducer, not "mod N".
def _delaunay_square(d, N):
    deltas = np.array([np.array(x, dtype=np.int64) for x in _direction_deltas(d)])
    m = square_repetitions(d, N)
    # the box: each chosen axis vector tiled m times
    M = np.array(_SQUARE_GEN[d], dtype=np.int64).T * m[None, :]
    reduce, det = _square_reducer(M)
    # find every distinct lattice point of this torus: start at the origin and
    # keep stepping in all directions, wrapping each result, until nothing new
    # turns up. det says how many points there should end up being.
    steps = np.vstack([deltas, -deltas])
    start = tuple(int(x) for x in reduce(np.zeros((1, d), dtype=np.int64))[0])
    seen, dq = {start}, deque([np.array(start, dtype=np.int64)])
    while dq:
        p = dq.popleft()
        for q in reduce(p[None, :] + steps):
            qt = tuple(int(x) for x in q)
            if qt not in seen:
                seen.add(qt)
                dq.append(q)
    cosets = np.array(sorted(seen), dtype=np.int64)     # all the torus points
    # now trace simplices from every point, same as rhombic but wrapping with reduce
    raw = set()
    for perm in permutations(range(d + 1)):
        cur, cols = cosets.copy(), [cosets]
        for step in perm[:d]:
            cur = cur + deltas[step]
            cols.append(reduce(cur))
        # cols now holds the d+1 corners for every starting point at once
        for pts in np.stack(cols, axis=1):
            s = frozenset(tuple(int(x) for x in v) for v in pts)
            if len(s) == d + 1:
                raw.add(s)
    return raw


# the one function the rest of the code calls. pick the wrapping, get the raw
# simplices, then relabel the corner coordinates as plain integers 0,1,2,... so
# gudhi can work with them. returns the corner list, the simplices as integer
# tuples, and the coordinate-to-integer lookup.
def delaunay_torus(d, N, orientation="rhombic"):
    raw = _delaunay_square(d, N) if orientation == "square" else _delaunay_rhombic(d, N)
    vertices = sorted({p for s in raw for p in s})
    vidx = {p: i for i, p in enumerate(vertices)}
    simplices = sorted(tuple(sorted(vidx[p] for p in s)) for s in raw)
    return vertices, simplices, vidx
