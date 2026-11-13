"""Regular-quad patch: keep ZRemesher's own block and join it to the border.

ZRemesher fills the masked area with a clean, regular quad block, but that
block ends in a smooth outline of its own instead of on the mask border, and
the faces left between the two overlap. This module keeps the block, discards
what lies between, and rebuilds only that narrow gap as a transition band whose
two sides meet the mask border and the block vertex for vertex.
"""
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

import lqr_validate as validation
from patch_remesh import Surface, TriMesh, _split, _unit, _vertex_normal, remesh
from recovery_geometry import directed, edge, polygon_triangles, quad_quality


def _face_key(vertices, face):
    points = tuple(vertices[i] for i in face)
    return min(points[k:] + points[:k] for k in range(len(points)))


def _triangles(vertices, faces):
    out = []
    for face in faces:
        if len(face) == 4:
            a, b, c, d = face
            p = [np.asarray(vertices[i]) for i in face]
            if np.linalg.norm(p[0] - p[2]) <= np.linalg.norm(p[1] - p[3]):
                out += [(a, b, c), (a, c, d)]
            else:
                out += [(a, b, d), (b, c, d)]
        else:
            out += polygon_triangles(vertices, face)
    return out


def _loops(edges):
    """Closed loops from directed edges; each vertex must have one successor."""
    successor = {}
    for a, b in edges:
        if a in successor:
            raise ValueError("An outline touches itself at one vertex")
        successor[a] = b
    loops, seen = [], set()
    for start in sorted(successor):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        while successor.get(loop[-1]) != start:
            nxt = successor.get(loop[-1])
            if nxt is None or nxt in seen:
                raise ValueError("An outline is not a closed loop")
            loop.append(nxt)
            seen.add(nxt)
        loops.append(loop)
    return loops


def _open_edges(faces):
    twins = Counter((a, b) for f in faces for a, b in directed(f))
    return [(a, b) for (a, b) in twins if (b, a) not in twins]


def _simple_block(faces, least):
    """Largest sets of edge-connected quads whose outline is a simple loop."""
    faces = list(faces)
    for _ in range(60):
        open_edges = _open_edges(faces)
        degree = Counter(v for e in open_edges for v in e)
        pinched = {v for v, n in degree.items() if n > 2}
        open_set = set(open_edges)
        keep = []
        for f in faces:
            if any(v in pinched for v in f):
                continue
            if sum((a, b) in open_set for a, b in directed(f)) >= 3:
                continue
            keep.append(f)
        # Drop small islands.
        owners = defaultdict(list)
        for i, f in enumerate(keep):
            for a, b in directed(f):
                owners[edge(a, b)].append(i)
        label = [-1] * len(keep)
        sizes = []
        for i in range(len(keep)):
            if label[i] >= 0:
                continue
            label[i] = len(sizes)
            stack, count = [i], 0
            while stack:
                j = stack.pop()
                count += 1
                for a, b in directed(keep[j]):
                    for k in owners[edge(a, b)]:
                        if label[k] < 0:
                            label[k] = label[i]
                            stack.append(k)
            sizes.append(count)
        limit = max(least, 0.05 * max(sizes)) if sizes else least
        kept = [f for i, f in enumerate(keep) if sizes[label[i]] >= limit]
        if len(kept) == len(faces):
            return kept
        faces = kept
    return faces


class _NoGap(Exception):
    """An island of the block has no gap of its own to be joined through."""

    def __init__(self, loop):
        super().__init__("island without a gap")
        self.loop = loop


def _untangle(triangles, place, surface, fixed, rim):
    """Pull free vertices of wrongly facing triangles back among their neighbours."""
    ring = defaultdict(set)
    for t in triangles:
        for a, b in directed(t):
            ring[a].add(b)
            ring[b].add(a)
    for _ in range(12):
        ids = sorted({v for t in triangles for v in t})
        _, facing = surface.project(np.array([place[v] for v in ids]))
        normal = dict(zip(ids, facing))
        loose = set()
        for t in triangles:
            p, q, r = (place[v] for v in t)
            n = np.cross(q - p, r - p)
            length = np.linalg.norm(n)
            ref = normal[t[0]] + normal[t[1]] + normal[t[2]]
            if length <= 1e-30 or float(np.dot(n, ref)) <= 0.15 * length * np.linalg.norm(ref):
                loose.update(v for v in t if v not in fixed)
        if not loose:
            return True
        order = sorted(loose)
        goals = []
        for v in order:
            around = [place[n] for n in ring[v] if n not in rim] or [place[n] for n in ring[v]]
            goals.append(0.3 * place[v] + 0.7 * np.mean(around, axis=0))
        snapped, _ = surface.project(np.array(goals))
        for v, p in zip(order, snapped):
            place[v] = p
    return False


def _turn_upright(strips, place, surface):
    """Flip inner edges of the stitched strip until its triangles face outward."""
    ids = sorted({v for t in strips for v in t})
    _, facing = surface.project(np.array([place[v] for v in ids]))
    normal = dict(zip(ids, facing))

    def upright(t):
        p, q, r = (place[v] for v in t)
        n = np.cross(q - p, r - p)
        length = np.linalg.norm(n)
        ref = normal[t[0]] + normal[t[1]] + normal[t[2]]
        return length > 1e-30 and float(np.dot(n, ref)) > 0.05 * length * np.linalg.norm(ref)

    strips = list(strips)
    for _ in range(8):
        owner = defaultdict(list)
        for i, t in enumerate(strips):
            for a, b in directed(t):
                owner[edge(a, b)].append(i)
        changed = False
        for i in range(len(strips)):
            if upright(strips[i]):
                continue
            for a, b in directed(strips[i]):
                pair = owner[edge(a, b)]
                if len(pair) != 2:
                    continue
                j = pair[0] if pair[1] == i else pair[1]
                c = next(v for v in strips[i] if v not in (a, b))
                d = next(v for v in strips[j] if v not in (a, b))
                if c == d or edge(c, d) in owner:
                    continue
                first, second = (a, d, c), (d, b, c)
                if upright(first) and upright(second):
                    strips[i], strips[j] = first, second
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    return strips


def build(vertices, work_faces, seam, outside_faces, native_vertices, native_faces,
          before_vertices, before_faces):
    """Return (vertices, quads, note) for the selected area, or raise ValueError."""
    dropped = set()
    for _ in range(6):
        try:
            return _build(vertices, work_faces, seam, outside_faces, native_vertices, native_faces,
                          before_vertices, before_faces, dropped)
        except _NoGap as missing:
            dropped.update(missing.loop)
    raise ValueError("ZRemesher's block could not be joined to the mask border")


def _build(vertices, work_faces, seam, outside_faces, native_vertices, native_faces,
           before_vertices, before_faces, dropped):
    known = Counter(_face_key(before_vertices, f) for f in before_faces)
    old_points = set(before_vertices)
    block = [f for f in native_faces
             if len(f) == 4 and not known[_face_key(native_vertices, f)]
             and not any(native_vertices[i] in old_points for i in f)
             and quad_quality(native_vertices, f) >= 0.12]
    if len(block) < 12:
        raise ValueError("ZRemesher left no regular block to keep")
    native = np.array(native_vertices, dtype=float)
    size = float(np.median([np.linalg.norm(native[a] - native[b]) for f in block for a, b in directed(f)]))

    source_points = np.array(vertices, dtype=float)
    source_triangles = np.array(_triangles(vertices, work_faces))
    inside = Surface(source_points, source_triangles, size / 2.0)
    outside = None
    if outside_faces:
        outside = Surface(source_points, np.array(_triangles(vertices, outside_faces)), size / 2.0)
    seam_samples, seam_length = [], []
    for a, b in seam:
        length = float(np.linalg.norm(source_points[a] - source_points[b]))
        for t in (0.0, 0.25, 0.5, 0.75):
            seam_samples.append(source_points[a] + t * (source_points[b] - source_points[a]))
            seam_length.append(length)
    seam_tree = cKDTree(np.array(seam_samples)) if seam_samples else None
    seam_length = np.array(seam_length)

    ids = sorted({i for f in block for i in f})
    points = native[ids]
    on_inside, _ = inside.project(points)
    reach = np.linalg.norm(on_inside - points, axis=1)
    usable = reach < 2.0 * size
    if outside is not None:
        on_outside, _ = outside.project(points)
        usable &= reach <= np.linalg.norm(on_outside - points, axis=1) + 1e-12
    if seam_tree is not None:
        gap, near = seam_tree.query(points)
        # Leave room for one row of transition faces everywhere.
        usable &= gap > np.maximum(1.6 * size, 1.3 * seam_length[near])
    allowed = {i for i, ok in zip(ids, usable) if ok}
    block = [f for f in block if all(i in allowed for i in f)]
    if len(block) < 12:
        raise ValueError("ZRemesher's block does not leave room for a transition band")
    hits, _, _ = validation.find_intersecting_faces(native_vertices, block,
                                                    validation.bbox_scale(native_vertices) * 1e-6)
    crossing = {i for pair in hits for i in pair}
    block = _simple_block([f for i, f in enumerate(block) if i not in crossing], 12)
    if dropped:
        # Leave out whole islands that an earlier attempt could not join.
        owners = defaultdict(list)
        for i, f in enumerate(block):
            for v in f:
                owners[v].append(i)
        gone, stack = set(), [i for v in dropped for i in owners.get(v, [])]
        while stack:
            i = stack.pop()
            if i in gone:
                continue
            gone.add(i)
            stack.extend(j for v in block[i] for j in owners[v])
        block = [f for i, f in enumerate(block) if i not in gone]
    if len(block) < 12:
        raise ValueError("ZRemesher's block is too broken to keep")

    # Same facing as the source surface.
    centres = np.array([native[list(f)].mean(axis=0) for f in block])
    _, facing = inside.project(centres)
    normals = np.array([np.cross(native[f[2]] - native[f[0]], native[f[3]] - native[f[1]]) for f in block])
    if float(np.sum(np.einsum("ij,ij->i", normals, facing))) < 0:
        block = [f[::-1] for f in block]

    outline = _loops(_open_edges(block))
    # A hole of one missing quad is simply given its quad back.
    single = [loop for loop in outline if len(loop) == 4]
    if single and len(outline) > len(single):
        block += [tuple(reversed(loop)) for loop in single]
        outline = _loops(_open_edges(block))
    if any(len(loop) % 2 for loop in outline):
        raise ValueError("ZRemesher's block has an outline with an odd number of edges")
    block_triangles = np.array(_triangles(native_vertices, block))
    block_surface = Surface(native, block_triangles, size / 2.0)
    rim_samples = []
    for loop in outline:
        for a, b in zip(loop, loop[1:] + loop[:1]):
            for t in np.linspace(0.0, 1.0, 8, endpoint=False):
                rim_samples.append(native[a] + t * (native[b] - native[a]))
    rim_tree = cKDTree(np.array(rim_samples))

    # Source surface as triangles, refined near the block's outline so that
    # what is cut away follows the outline closely.
    used = {v for t in source_triangles for v in t}
    mesh = TriMesh({int(v): source_points[v].copy() for v in used}, [tuple(int(i) for i in t) for t in source_triangles])
    mesh.next_vertex = max(len(vertices), mesh.next_vertex)
    for v in list(mesh.p):
        mesh.n[v] = _vertex_normal(mesh, v)
        mesh.size[v] = size
    border = {v for e in seam for v in e}
    for _ in range(5):
        order = list(mesh.p)
        distance, _ = rim_tree.query(np.array([mesh.p[v] for v in order]))
        near_rim = {v for v, d in zip(order, distance) if d < 4.0 * size}
        changed = False
        for a, b in sorted(mesh.edges(), key=lambda e: -mesh.length(*e)):
            if a not in near_rim and b not in near_rim:
                continue
            if len(mesh.edge_faces(a, b)) != 2 or mesh.length(a, b) <= 0.7 * size:
                continue
            _split(mesh, a, b, 0.5 * (mesh.p[a] + mesh.p[b]))
            changed = True
        if not changed:
            break

    order = list(mesh.p)
    where = np.array([mesh.p[v] for v in order])
    on_block, _ = block_surface.project(where)
    to_block = np.linalg.norm(on_block - where, axis=1)
    to_rim, _ = rim_tree.query(where)
    covered = {v for v, s, r in zip(order, to_block, to_rim)
               if s < 0.35 * size or (s < 0.97 * r and s < 6.0 * size)}
    face_ids = list(mesh.f)
    middle = np.array([sum(mesh.p[v] for v in mesh.f[i]) / 3.0 for i in face_ids])
    on_block, _ = block_surface.project(middle)
    to_block = np.linalg.norm(on_block - middle, axis=1)
    to_rim, _ = rim_tree.query(middle)
    for i, s, r in zip(face_ids, to_block, to_rim):
        triangle = mesh.f[i]
        hidden = s < 0.35 * size or (s < 0.97 * r and s < 6.0 * size) or any(v in covered for v in triangle)
        if not hidden:
            continue
        if any(v in border for v in triangle):
            raise ValueError("ZRemesher's block reaches the mask border")
        mesh.remove(i)

    # The cut edge must be a set of simple loops.
    seam_edges = {edge(a, b) for a, b in seam}
    for _ in range(60):
        cut = [(a, b) for (a, b) in _open_edges(list(mesh.f.values())) if edge(a, b) not in seam_edges]
        degree = Counter(a for a, b in cut)
        pinched = {v for v, n in degree.items() if n > 1}
        stranded = [i for i, t in mesh.f.items()
                    if sum(len(mesh.edge_faces(a, b)) == 1 and edge(a, b) not in seam_edges
                           for a, b in directed(t)) >= 2 and not any(v in border for v in t)]
        doomed = {i for v in pinched for i in mesh.vf[v]} | set(stranded)
        doomed = {i for i in doomed if not any(v in border for v in mesh.f[i])}
        if not doomed:
            if pinched:
                raise ValueError("The gap around ZRemesher's block could not be outlined")
            break
        for i in doomed:
            mesh.remove(i)
    cut = [(a, b) for (a, b) in _open_edges(list(mesh.f.values())) if edge(a, b) not in seam_edges]
    cut_loops = _loops(cut)
    def perimeter(loop, place):
        return sum(float(np.linalg.norm(place[a] - place[b])) for a, b in zip(loop, loop[1:] + loop[:1]))

    # Pair each outline of the block with the cut loop that surrounds it.
    cut_owner = {}
    for n, loop in enumerate(cut_loops):
        for v in loop:
            cut_owner[v] = n
    cut_ids = sorted(cut_owner)
    cut_tree = cKDTree(np.array([mesh.p[v] for v in cut_ids]))
    block_id = {}
    taken = set()
    strips = []
    for loop in sorted(outline, key=len, reverse=True):
        for v in loop:
            if v not in block_id:
                block_id[v] = mesh.next_vertex
                mesh.p[mesh.next_vertex] = native[v].copy()
                mesh.next_vertex += 1
        reach, near = cut_tree.query(native[loop])
        votes = Counter(cut_owner[cut_ids[i]] for i in near)
        partner = votes.most_common(1)[0][0]
        ratio = perimeter(cut_loops[partner], mesh.p) / max(perimeter(loop, native), 1e-30)
        mismatch = partner in taken or not 0.6 <= ratio <= 2.0
        if mismatch and len(loop) > 12:
            raise _NoGap(loop)
        if mismatch or float(np.mean(reach)) > 6.0 * size:
            # A small hole inside the block, with nothing of the source left
            # under it: close it from its own centre.
            centre = mesh.next_vertex
            mesh.next_vertex += 1
            mesh.p[centre] = inside.project(native[loop].mean(axis=0))[0][0]
            for a, b in zip(loop, loop[1:] + loop[:1]):
                strips.append((block_id[b], block_id[a], centre))
            continue
        taken.add(partner)
        rim = [block_id[v] for v in reversed(loop)]      # runs alongside the cut loop
        ring = cut_loops[partner]
        start = int(np.argmin([np.linalg.norm(mesh.p[v] - mesh.p[rim[0]]) for v in ring]))
        ring = ring[start:] + ring[:start]
        both = ring + rim
        _, facing = inside.project(np.array([mesh.p[v] for v in both]))
        normal = dict(zip(both, facing))

        def upright(triangle):
            p, q, r = (mesh.p[v] for v in triangle)
            n = np.cross(q - p, r - p)
            return float(np.dot(n, normal[triangle[0]] + normal[triangle[1]] + normal[triangle[2]])) > 0.0

        i = j = 0
        while i < len(ring) or j < len(rim):
            a, b = ring[i % len(ring)], rim[j % len(rim)]
            by_ring = (ring[(i + 1) % len(ring)], a, b)
            by_rim = (b, rim[(j + 1) % len(rim)], a)
            step_ring = np.linalg.norm(mesh.p[by_ring[0]] - mesh.p[b])
            step_rim = np.linalg.norm(mesh.p[a] - mesh.p[by_rim[1]])
            if j >= len(rim):
                take_ring = True
            elif i >= len(ring):
                take_ring = False
            else:
                # Prefer the step that faces the right way; then the shorter one.
                ok_ring, ok_rim = upright(by_ring), upright(by_rim)
                take_ring = step_ring <= step_rim if ok_ring == ok_rim else ok_ring
            if take_ring:
                strips.append(by_ring)
                i += 1
            else:
                strips.append(by_rim)
                j += 1
    for n, loop in enumerate(cut_loops):
        if n in taken:
            continue
        # A scrap of the source left under the block: remove it whole.
        scrap, stack = set(), [i for i in mesh.vf[loop[0]]]
        while stack:
            i = stack.pop()
            if i in scrap or i not in mesh.f:
                continue
            scrap.add(i)
            stack.extend(j for v in mesh.f[i] for j in mesh.vf[v])
        if any(v in border for i in scrap for v in mesh.f[i]):
            raise ValueError("Part of the gap around ZRemesher's block has no outline to join")
        for i in scrap:
            mesh.remove(i)

    strips = _turn_upright(strips, mesh.p, inside)
    _untangle(list(mesh.f.values()) + strips, mesh.p, inside,
              border | set(block_id.values()), set(block_id.values()))

    total = mesh.next_vertex
    positions = [tuple(p) for p in vertices] + [(0.0, 0.0, 0.0)] * (total - len(vertices))
    for v, p in mesh.p.items():
        if v >= len(vertices):
            positions[v] = tuple(p)
    band = list(mesh.f.values()) + strips
    # Triangles of about twice the block's edge length become quads of its size.
    positions, quads = remesh(positions, band, 0, surface=inside, edge_length=2.2 * size)
    # The block's inner vertices.
    for f in block:
        for v in f:
            if v not in block_id:
                block_id[v] = len(positions)
                positions.append(tuple(native[v]))
    kept = [tuple(block_id[v] for v in f) for f in block]
    note = f"Regular block kept from ZRemesher: {len(kept)} quads; transition band: {len(quads)} quads."
    return positions, quads + kept, note
