# the lattice for this side of the project: the ordinary cubic grid (squares in
# 2d, cubes in 3d, and so on) wrapped into a torus. ATEAMS builds the actual cell
# complex and its boundary matrices; this is just a one-line wrapper so the rest
# of the code has a simple build(d, N) to call.

from ateams.complexes import Cubical


# d-dimensional cubic grid of side N, wrapped around (periodic) so it's a torus
def build(d, N):
    return Cubical().fromCorners([N] * d, periodic=True)
