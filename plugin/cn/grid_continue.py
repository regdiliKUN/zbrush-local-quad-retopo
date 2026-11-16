"""Grid continuation: carry the regular quad grid around the mask through it.

When the faces just outside the mask form a regular grid (four quads at every
vertex, no poles), every border vertex has a place on one integer lattice. The
masked area is then filled with the lattice cells inside the border, so every
grid line of the surroundings runs straight through the patch and the patch
has the surroundings' density. The new vertices are spread evenly over the
sculpted surface and the result is checked like every other layout.
"""
from collections import defaultdict, deque

import numpy as np

from recovery_geometry import directed, edge

# Rings of outside quads that must line up with the lattice around the mask.
RINGS = 3


class NotAGrid(ValueError):
    """The surroundings of the mask are not a regular grid."""


def _rot(e):
    return (-e[1], e[0])


def lattice(outside_faces, seam):
    """Integer lattice coordinates of the mask border and the quads around it."""
    border = {v for e in seam for v in e}
    seam_edges = {edge(a, b) for a, b in seam}
    owners = defaultdict(list)
    for i, f in enumerate(outside_faces):
        for a, b in directed(f):
            owners[edge(a, b)].append(i)
    at_vertex = defaultdict(list)
    for i, f in enumerate(outside_faces):
        for v in f:
            at_vertex[v].append(i)
    # Outside faces within RINGS rings of the border.
    ring = {}
    frontier = {i for v in border for i in at_vertex.get(v, [])}
    for depth in range(RINGS):
        frontier = {i for i in frontier if i not in ring}
        for i in frontier:
            ring[i] = depth
        frontier = {j for i in frontier for v in outside_faces[i] for j in at_vertex[v]}
    for a, b in seam:
        near = owners.get(edge(a, b), [])
        if len(near) != 1 or len(outside_faces[near[0]]) != 4:
            raise NotAGrid("the faces along the mask border are not all quads")
    # Every vertex near the border must be a grid vertex: four quads around it.
    for i in ring:
        for v in outside_faces[i]:
            if v in border:
                continue
            around = at_vertex[v]
            if ring.get(i, RINGS) < RINGS - 1 and (len(around) != 4
                                                    or any(len(outside_faces[j]) != 4 for j in around)):
                raise NotAGrid("the mesh around the mask has poles or non-quad faces")
    start = owners[edge(*next(iter(seam)))][0]
    f = outside_faces[start]
    coords = {f[0]: (0, 0), f[1]: (1, 0)}
    queue, seen = deque([start]), {start}
    while queue:
        i = queue.popleft()
        f = outside_faces[i]
        for k in range(4):
            w0, w1 = f[k], f[(k + 1) % 4]
            if w0 in coords and w1 in coords:
                e = (coords[w1][0] - coords[w0][0], coords[w1][1] - coords[w0][1])
                if abs(e[0]) + abs(e[1]) != 1:
                    raise NotAGrid("the grid around the mask is not regular")
                r = _rot(e)
                want = {f[(k + 3) % 4]: (coords[w0][0] + r[0], coords[w0][1] + r[1]),
                        f[(k + 2) % 4]: (coords[w1][0] + r[0], coords[w1][1] + r[1])}
                for v, c in want.items():
                    if coords.setdefault(v, c) != c:
                        raise NotAGrid("the grid around the mask does not close up")
                break
        else:
            raise NotAGrid("the grid around the mask is not connected")
        for a, b in directed(f):
            if edge(a, b) in seam_edges:
                continue
            for j in owners[edge(a, b)]:
                if j in ring and j not in seen:
                    seen.add(j)
                    queue.append(j)
    if any(v not in coords for v in border):
        raise NotAGrid("the mask border is not connected through the grid")
    for a, b in seam:
        if abs(coords[a][0] - coords[b][0]) + abs(coords[a][1] - coords[b][1]) != 1:
            raise NotAGrid("the mask border does not follow the grid lines")
    placed = [coords[v] for v in border]
    if len(set(placed)) != len(placed):
        raise NotAGrid("the grid wraps around the mask")
    return {v: coords[v] for v in border}


def cells(seam, coords):
    """Lattice cells inside the border (even-odd rule over all border loops)."""
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    # Vertical border edges, as x and the y span they cover.
    vertical = defaultdict(list)
    for a, b in seam:
        (x0, y0), (x1, y1) = coords[a], coords[b]
        if x0 == x1:
            vertical[min(y0, y1)].append(x0)
    inside = []
    for y in range(min(ys), max(ys)):
        crossings = sorted(vertical.get(y, []))
        for left, right in zip(crossings[0::2], crossings[1::2]):
            inside += [(x, y) for x in range(left, right)]
    if not inside:
        raise NotAGrid("the mask covers no whole grid cell")
    return inside


def build(vertices, seam, outside_faces, surface, source_faces):
    try:
        return _build(vertices, seam, outside_faces, surface, source_faces)
    except NotAGrid:
        raise
    except Exception as exc:          # never take the whole run down
        raise NotAGrid(f"grid continuation failed: {type(exc).__name__}: {exc}") from exc


def _build(vertices, seam, outside_faces, surface, source_faces):
    """Return (positions, quads) continuing the surrounding grid through the mask.

    vertices: all vertices (border ones keep their index); seam: directed
    border edges of the patch; outside_faces: the untouched faces; surface:
    a patch_remesh.Surface of the masked source; source_faces: the masked
    source faces, used to check the winding.
    """
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    coords = lattice(outside_faces, seam)
    squares = cells(seam, coords)
    at = {c: v for v, c in coords.items()}
    corners = {(x + dx, y + dy) for x, y in squares for dx in (0, 1) for dy in (0, 1)}
    missing = [c for c in corners if c not in at]
    border_points = set(at)
    # A cell corner on the border must be a border vertex.
    seam_points = {coords[v] for e in seam for v in e}
    free = [c for c in missing]
    if any(c in seam_points for c in free):
        raise NotAGrid("the mask border does not follow the grid lines")
    index = {c: n for n, c in enumerate(free)}
    positions = [np.asarray(p, dtype=float) for p in vertices]
    fixed = {c: positions[v] for c, v in at.items()}
    neighbours = {}
    for c in free:
        neighbours[c] = [q for q in ((c[0] + 1, c[1]), (c[0] - 1, c[1]), (c[0], c[1] + 1), (c[0], c[1] - 1))
                         if q in index or q in border_points]
        if len(neighbours[c]) != 4:
            raise NotAGrid("a new grid vertex is not surrounded by the grid")
    n = len(free)
    if n:
        # Harmonic fill on the lattice: each new vertex the mean of its four.
        matrix = lil_matrix((n, n))
        rhs = np.zeros((n, 3))
        for c, row in index.items():
            matrix[row, row] = 4.0
            for q in neighbours[c]:
                if q in index:
                    matrix[row, index[q]] = -1.0
                else:
                    rhs[row] += fixed[q]
        place = np.asarray(spsolve(matrix.tocsr(), rhs)).reshape(n, 3)
        place, _ = surface.project(place)
        # Spread evenly over the sculpted surface: move towards the mean of the
        # four neighbours, then back onto the surface.
        nb_index = np.array([[index.get(q, -1) for q in neighbours[c]] for c in free])
        nb_fixed = np.array([[fixed[q] if q not in index else np.zeros(3) for q in neighbours[c]] for c in free])
        is_free = nb_index >= 0
        for _ in range(40):
            around = np.where(is_free[..., None], place[np.maximum(nb_index, 0)], nb_fixed).mean(axis=1)
            place, _ = surface.project(0.4 * place + 0.6 * around)
    else:
        place = np.zeros((0, 3))
    ids = {}
    for c, v in at.items():
        ids[c] = v
    out = [tuple(p) for p in vertices]
    for c, row in index.items():
        ids[c] = len(out)
        out.append(tuple(float(x) for x in place[row]))
    quads = [(ids[(x, y)], ids[(x + 1, y)], ids[(x + 1, y + 1)], ids[(x, y + 1)]) for x, y in squares]
    # The lattice follows the winding of the outside quads; the patch must
    # use the border edges in the direction the seam gives.
    used = {(a, b) for q in quads for a, b in directed(q)}
    if not all((a, b) in used for a, b in seam):
        quads = [tuple(reversed(q)) for q in quads]
        used = {(a, b) for q in quads for a, b in directed(q)}
        if not all((a, b) in used for a, b in seam):
            raise NotAGrid("the grid cells do not meet the mask border")
    return out, quads
