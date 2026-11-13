"""Rebuild a selected patch as an even all-quad mesh inside its fixed border.

The patch is re-triangulated at an even edge length on the source surface.
Every triangle then becomes three quads. Along the border the triangulation
uses every second border vertex, so the edge midpoints added by that step
land exactly on the remaining border vertices: the finished patch joins the
untouched mesh vertex for vertex, with no extra points on the seam.
"""
from collections import defaultdict
import math

import numpy as np
from scipy.spatial import cKDTree

from recovery_geometry import directed, edge, polygon_triangles, quad_quality


def _unit(vector):
    length = np.linalg.norm(vector)
    return vector / length if length > 0 else vector


class Surface:
    """Closest-point queries on the source triangles."""

    def __init__(self, points, triangles, spacing):
        self.a = points[triangles[:, 0]]
        self.b = points[triangles[:, 1]]
        self.c = points[triangles[:, 2]]
        samples, owner = [], []
        longest = np.maximum.reduce([
            np.linalg.norm(self.b - self.a, axis=1),
            np.linalg.norm(self.c - self.b, axis=1),
            np.linalg.norm(self.a - self.c, axis=1)])
        levels = np.clip(np.ceil(longest / max(spacing, 1e-12)), 1, 8).astype(int)
        for level in np.unique(levels):
            ids = np.nonzero(levels == level)[0]
            for i in range(level + 1):
                for j in range(level + 1 - i):
                    u, w = i / level, j / level
                    samples.append(self.a[ids] * (1 - u - w) + self.b[ids] * u + self.c[ids] * w)
                    owner.append(ids)
        self.owner = np.concatenate(owner)
        self.tree = cKDTree(np.concatenate(samples))
        normal = np.cross(self.b - self.a, self.c - self.a)
        length = np.linalg.norm(normal, axis=1)
        self.normal = normal / np.where(length > 0, length, 1)[:, None]

    def project(self, points, normals=None):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        k = min(6, len(self.owner))
        _, near = self.tree.query(points, k=k)
        ids = self.owner[near.reshape(len(points), -1)]
        closest = _closest_on_triangles(points[:, None, :], self.a[ids], self.b[ids], self.c[ids])
        distance = np.sum((closest - points[:, None, :]) ** 2, axis=2)
        if normals is not None:
            # Stay on the same side of a thin wall or a tight fold.
            facing = np.einsum("qkj,qj->qk", self.normal[ids], normals)
            distance = np.where(facing < 0.0, distance * 16 + 1e-18, distance)
        pick = np.argmin(distance, axis=1)
        rows = np.arange(len(points))
        return closest[rows, pick], self.normal[ids[rows, pick]]


def _closest_on_triangles(p, a, b, c):
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    nn = np.sum(normal * normal, axis=-1)
    safe = np.where(nn > 0, nn, 1)
    ap = p - a
    u = np.sum(np.cross(ap, ac) * normal, axis=-1) / safe
    w = np.sum(np.cross(ab, ap) * normal, axis=-1) / safe
    inside = (nn > 0) & (u >= 0) & (w >= 0) & (u + w <= 1)
    best = a + u[..., None] * ab + w[..., None] * ac
    best_distance = np.where(inside, np.sum((best - p) ** 2, axis=-1), np.inf)
    for x, y in ((a, b), (b, c), (c, a)):
        xy = y - x
        length = np.sum(xy * xy, axis=-1)
        t = np.clip(np.sum((p - x) * xy, axis=-1) / np.where(length > 0, length, 1), 0, 1)
        q = x + t[..., None] * xy
        distance = np.sum((q - p) ** 2, axis=-1)
        better = distance < best_distance
        best = np.where(better[..., None], q, best)
        best_distance = np.where(better, distance, best_distance)
    return best


class TriMesh:
    def __init__(self, positions, triangles):
        self.p = dict(positions)
        self.n = {}
        self.size = {}
        self.f = {}
        self.vf = defaultdict(set)
        self.next_face = 0
        self.next_vertex = max(self.p) + 1
        for triangle in triangles:
            self.add(triangle)

    def add(self, triangle):
        self.f[self.next_face] = triangle
        for v in triangle:
            self.vf[v].add(self.next_face)
        self.next_face += 1

    def remove(self, face):
        for v in self.f.pop(face):
            self.vf[v].discard(face)

    def edge_faces(self, a, b):
        return [i for i in self.vf[a] if i in self.vf[b]]

    def neighbours(self, v):
        return {n for i in self.vf[v] for n in self.f[i] if n != v}

    def normal(self, triangle):
        a, b, c = (self.p[i] for i in triangle)
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        return np.array((uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx))

    def sound(self, triangle, least=0.0):
        """The triangle faces the same way as the surface at its corners."""
        normal = self.normal(triangle)
        length = np.linalg.norm(normal)
        if length <= max(least, 1e-30):
            return False
        reference = self.n[triangle[0]] + self.n[triangle[1]] + self.n[triangle[2]]
        return np.dot(normal, reference) > 0.2 * length * np.linalg.norm(reference)

    def opposite(self, face, a, b):
        return next(v for v in self.f[face] if v != a and v != b)

    def edges(self):
        return {edge(a, b) for t in self.f.values() for a, b in directed(t)}

    def length(self, a, b):
        return float(np.linalg.norm(self.p[a] - self.p[b]))


def _border_loops(mesh):
    successor = {}
    for triangle in mesh.f.values():
        for a, b in directed(triangle):
            if len(mesh.edge_faces(a, b)) == 1:
                if a in successor:
                    raise ValueError("The selection border touches itself at one vertex")
                successor[a] = b
    loops = []
    seen = set()
    for start in sorted(successor):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        while successor.get(loop[-1]) != start:
            nxt = successor.get(loop[-1])
            if nxt is None or nxt in seen:
                raise ValueError("The selection border is not a set of closed loops")
            loop.append(nxt)
            seen.add(nxt)
        loops.append(loop)
    return loops


def _collapse(mesh, gone, kept, limit=None, guard_normals=True):
    """Merge vertex `gone` into `kept`. Returns False if it would damage the mesh."""
    shared = mesh.edge_faces(gone, kept)
    if not shared:
        return False
    apexes = {mesh.opposite(i, gone, kept) for i in shared}
    if (mesh.neighbours(gone) & mesh.neighbours(kept)) != apexes:
        return False
    moved = [i for i in mesh.vf[gone] if i not in shared]
    target = mesh.p[kept]
    for i in moved:
        triangle = mesh.f[i]
        after = tuple(kept if v == gone else v for v in triangle)
        new = np.cross(mesh.p[after[1]] - mesh.p[after[0]], mesh.p[after[2]] - mesh.p[after[0]])
        old = mesh.normal(triangle)
        new_length, old_length = np.linalg.norm(new), np.linalg.norm(old)
        if new_length <= 1e-30:
            return False
        if guard_normals and old_length > 0 and np.dot(new, old) < 0.2 * new_length * old_length:
            return False
        if guard_normals and not mesh.sound(after):
            return False
        if limit is not None:
            for v in triangle:
                if v != gone and np.linalg.norm(mesh.p[v] - target) > limit:
                    return False
    for i in shared:
        mesh.remove(i)
    for i in moved:
        triangle = mesh.f[i]
        mesh.remove(i)
        mesh.add(tuple(kept if v == gone else v for v in triangle))
    del mesh.vf[gone]
    del mesh.p[gone]
    del mesh.n[gone]
    mesh.size.pop(gone, None)
    return True


def _split(mesh, a, b, point):
    faces = mesh.edge_faces(a, b)
    new = mesh.next_vertex
    mesh.next_vertex += 1
    mesh.p[new] = point
    mesh.n[new] = _unit(mesh.n[a] + mesh.n[b])
    mesh.size[new] = 0.5 * (mesh.size[a] + mesh.size[b])
    for i in faces:
        triangle = mesh.f[i]
        mesh.remove(i)
        k = triangle.index(a)
        if triangle[(k + 1) % 3] == b:
            c = triangle[(k + 2) % 3]
            mesh.add((a, new, c))
            mesh.add((new, b, c))
        else:
            c = triangle[(k + 1) % 3]
            mesh.add((a, c, new))
            mesh.add((new, c, b))
    return new


def _flip(mesh, a, b, border, force=False):
    faces = mesh.edge_faces(a, b)
    if len(faces) != 2:
        return False
    c, d = (mesh.opposite(i, a, b) for i in faces)
    if c == d or mesh.edge_faces(c, d):
        return False

    def deviation(v, change):
        goal = 4 if v in border else 6
        return abs(len(mesh.vf[v]) + change - goal)

    before = deviation(a, 0) + deviation(b, 0) + deviation(c, 0) + deviation(d, 0)
    after = deviation(a, -1) + deviation(b, -1) + deviation(c, 1) + deviation(d, 1)
    if after >= before and not force:
        return False
    first = mesh.f[faces[0]]
    k = first.index(a)
    if first[(k + 1) % 3] != b:
        a, b = b, a
    # faces[0] is (a, b, c) and faces[1] is (b, a, d).
    n1, n2 = mesh.normal((a, b, c)), mesh.normal((b, a, d))
    l1, l2 = np.linalg.norm(n1), np.linalg.norm(n2)
    if l1 <= 0 or l2 <= 0 or np.dot(n1, n2) < (0.5 if force else 0.94) * l1 * l2:
        return False
    new1, new2 = (a, d, c), (d, b, c)
    m1, m2 = mesh.normal(new1), mesh.normal(new2)
    reference = n1 / l1 + n2 / l2
    for m in (m1, m2):
        length = np.linalg.norm(m)
        if length <= 1e-30 or np.dot(m, reference) < 0.5 * length * np.linalg.norm(reference):
            return False
        if length < (0.02 if force else 0.15) * min(l1, l2):
            return False
    if not (mesh.sound(new1) and mesh.sound(new2)):
        return False
    for i in faces:
        mesh.remove(i)
    mesh.add(new1)
    mesh.add(new2)
    return True


def _vertex_normal(mesh, v):
    total = np.zeros(3)
    for i in mesh.vf[v]:
        total += mesh.normal(mesh.f[i])
    return _unit(total)


def _relax(mesh, surface, movable, strength=0.6):
    ids = [v for v in movable if v in mesh.p and mesh.vf[v]]
    if not ids:
        return
    targets, normals = [], []
    for v in ids:
        ring = mesh.neighbours(v)
        centre = np.mean([mesh.p[n] for n in ring], axis=0)
        normal = mesh.n[v]
        step = centre - mesh.p[v]
        step -= normal * np.dot(step, normal)
        targets.append(mesh.p[v] + strength * step)
        normals.append(normal)
    projected, facing = surface.project(np.array(targets), np.array(normals))
    for v, point, surface_normal in zip(ids, projected, facing):
        old, old_normal = mesh.p[v], mesh.n[v]
        least = [0.2 * np.linalg.norm(mesh.normal(mesh.f[i])) for i in mesh.vf[v]]
        mesh.p[v] = point
        if np.dot(surface_normal, old_normal) > 0.3:
            mesh.n[v] = surface_normal
        if not all(mesh.sound(mesh.f[i], floor) for i, floor in zip(mesh.vf[v], least)):
            mesh.p[v], mesh.n[v] = old, old_normal


def _turn(mesh, before, v, after):
    """Signed turn of the border at v; positive at an outward corner."""
    first, second = mesh.p[v] - mesh.p[before], mesh.p[after] - mesh.p[v]
    return math.atan2(float(np.dot(np.cross(first, second), mesh.n[v])), float(np.dot(first, second)))


def _shares_apex(mesh, pinned, v, corner, other):
    """An outward corner between two pairs that both lie on straight border."""
    straight = math.radians(30)
    return (_turn(mesh, v, corner, other) >= math.radians(55)
            and abs(_turn(mesh, pinned[v][0], v, corner)) < straight
            and abs(_turn(mesh, corner, other, pinned[other][1])) < straight)


def _seat_apexes(mesh, surface, pinned, room=None):
    """Move the inner apex of each border pair to where both triangles are sound."""
    goal = {}
    for v, (left, right) in pinned.items():
        normal = mesh.n[v]
        first, second = mesh.p[v] - mesh.p[left], mesh.p[right] - mesh.p[v]
        inward = _unit(_unit(np.cross(normal, first)) + _unit(np.cross(normal, second)))
        if not np.any(inward):
            continue
        height = 0.87 * (np.linalg.norm(first) + np.linalg.norm(second))
        if room and v in room:
            # In a narrow band the apex stays short of the facing border.
            height = min(height, 0.5 * room[v])
        goal[v] = [mesh.p[v] + height * inward]
    # At an outward corner between two pairs, both pairs share one apex on the
    # corner's bisector instead of squeezing a sliver between two apexes.
    after = {left: v for v, (left, right) in pinned.items()}

    def apex_of(pin):
        return next(iter(mesh.neighbours(pin) - set(pinned[pin])))

    for v, (left, corner) in pinned.items():
        other = after.get(corner)
        if other is None or v not in goal or other not in goal or apex_of(v) != apex_of(other):
            continue
        if not _shares_apex(mesh, pinned, v, corner, other):
            continue
        normal = mesh.n[corner]
        first, second = mesh.p[corner] - mesh.p[v], mesh.p[other] - mesh.p[corner]
        inward = _unit(_unit(np.cross(normal, first)) + _unit(np.cross(normal, second)))
        shared = mesh.p[corner] + 0.75 * (np.linalg.norm(first) + np.linalg.norm(second)) * inward
        for pin in (v, other):
            goal[pin] = [g for g in goal[pin] if len(g) == 4] + [np.append(shared, 1.0)]
    wanted = defaultdict(list)
    for v, (left, right) in pinned.items():
        if v not in goal:
            continue
        wanted[apex_of(v)].append((np.mean([g[:3] for g in goal[v]], axis=0), mesh.n[v]))
    if not wanted:
        return
    ids = sorted(wanted)
    targets = np.array([np.mean([p for p, _ in wanted[i]], axis=0) for i in ids])
    normals = np.array([_unit(sum(n for _, n in wanted[i])) for i in ids])
    projected, _ = surface.project(targets, normals)
    for apex, goal in zip(ids, projected):
        start = mesh.p[apex]
        least = [0.2 * np.linalg.norm(mesh.normal(mesh.f[i])) for i in mesh.vf[apex]]
        for fraction in (1.0, 0.6, 0.3, 0.12):
            mesh.p[apex] = start + fraction * (goal - start)
            if all(mesh.sound(mesh.f[i], floor) for i, floor in zip(mesh.vf[apex], least)):
                break
            mesh.p[apex] = start


def remesh(vertices, faces, target_quads, surface=None, edge_length=None):
    """Return (vertices, quads) for the selected faces with the same border.

    `vertices` is the full vertex list; border vertices keep their indices and
    positions, and new vertices are appended. `surface`, when given, is the
    Surface the result must lie on (the faces themselves otherwise).
    `edge_length` overrides the triangle size derived from `target_quads`.
    """
    triangles = []
    for face in faces:
        if len(face) == 4:
            a, b, c, d = face
            p = [np.asarray(vertices[i]) for i in face]
            if np.linalg.norm(p[0] - p[2]) <= np.linalg.norm(p[1] - p[3]):
                triangles += [(a, b, c), (a, c, d)]
            else:
                triangles += [(a, b, d), (b, c, d)]
        else:
            triangles += polygon_triangles(vertices, face)
    used = {v for t in triangles for v in t}
    mesh = TriMesh({v: np.asarray(vertices[v], dtype=float) for v in used}, triangles)
    for a, b in mesh.edges():
        if len(mesh.edge_faces(a, b)) > 2:
            raise ValueError("The selected surface is not manifold")
    if surface is None:
        for v in used:
            mesh.n[v] = _vertex_normal(mesh, v)
    else:
        ids = sorted(used)
        _, facing = surface.project(np.array([mesh.p[v] for v in ids]))
        mesh.n.update(zip(ids, facing))
    source_points = np.array(vertices, dtype=float)
    source_triangles = np.array(triangles)
    area = 0.5 * np.linalg.norm(np.cross(
        source_points[source_triangles[:, 1]] - source_points[source_triangles[:, 0]],
        source_points[source_triangles[:, 2]] - source_points[source_triangles[:, 0]]), axis=1).sum()
    if area <= 0:
        raise ValueError("The selected surface has no area")

    loops = _border_loops(mesh)
    border_all = {v for loop in loops for v in loop}
    seam_length = [mesh.length(a, b) for loop in loops for a, b in zip(loop, loop[1:] + loop[:1])]
    # Each triangle yields three quads.
    count = max(2.0, target_quads / 3.0)
    size = math.sqrt(4.0 / math.sqrt(3.0) * area / count)
    if seam_length:
        # A border far finer than the request would force a ring of slivers.
        size = min(size, 6.0 * 2.0 * float(np.median(seam_length)))
    if edge_length:
        size = edge_length
    if surface is None:
        surface = Surface(source_points, source_triangles, size / 4.0)

    # Every second border vertex is "pinned": it keeps exactly two triangles,
    # (previous, pinned, apex) and (pinned, next, apex). That pair later
    # behaves as one triangle whose border edge already has its midpoint.
    pinned = {}
    for loop in loops:
        if len(loop) % 2 or len(loop) < 4:
            raise ValueError("A border loop is too short or has an odd number of edges")
        turning = [0.0, 0.0]
        for i, v in enumerate(loop):
            before = _unit(mesh.p[v] - mesh.p[loop[i - 1]])
            after = _unit(mesh.p[loop[(i + 1) % len(loop)]] - mesh.p[v])
            turning[i % 2] += float(np.dot(np.cross(before, after), mesh.n[v]))
        # A pinned vertex splits its corner between two triangles, which
        # suits outward and straight vertices; inward corners need the full
        # fan of an ordinary vertex.
        start = 0 if turning[0] <= turning[1] else 1
        for i in range(start + 1, start + 1 + len(loop), 2):
            pinned[loop[i % len(loop)]] = (loop[(i - 1) % len(loop)], loop[(i + 1) % len(loop)])
    border = border_all

    def locked(a, b):
        return any(v in ready for i in mesh.edge_faces(a, b) for v in mesh.f[i])

    def serves_border(n):
        """True if n is the inner apex of a prepared border pair."""
        return any(v in ready for i in mesh.vf[n] for v in mesh.f[i])

    ready = set()
    for v, (left, right) in pinned.items():
        # Replace the fan of v by two triangles on a new inner apex g, with g
        # joined to every vertex the fan touched. Close to v, g sees them all.
        chain = [left]
        while chain[-1] != right:
            step = [mesh.opposite(i, v, chain[-1]) for i in mesh.vf[v]
                    if chain[-1] in mesh.f[i] and mesh.f[i][(mesh.f[i].index(chain[-1]) + 1) % 3] == v]
            if len(step) != 1 or step[0] in chain:
                raise ValueError("A border vertex has a broken fan of faces")
            chain.append(step[0])
        if len(chain) - 1 != len(mesh.vf[v]):
            raise ValueError("A border vertex has a broken fan of faces")
        normal = mesh.n[v]
        first, second = mesh.p[v] - mesh.p[left], mesh.p[right] - mesh.p[v]
        inward = _unit(_unit(np.cross(normal, first)) + _unit(np.cross(normal, second)))
        if not np.any(inward):
            inward = _unit(np.cross(normal, first))
        reach = min(mesh.length(v, n) for n in chain)
        old_faces = list(mesh.vf[v])
        old = [mesh.f[i] for i in old_faces]
        for i in old_faces:
            mesh.remove(i)
        apex = mesh.next_vertex
        mesh.next_vertex += 1
        mesh.n[apex] = normal
        mesh.size[apex] = size
        placed = False
        for fraction in (0.35, 0.15, 0.05, 0.015):
            mesh.p[apex] = mesh.p[v] + fraction * reach * inward
            fan = [(left, v, apex), (v, right, apex)]
            # chain runs clockwise around v, so the outer triangles use it reversed.
            fan += [(apex, chain[k + 1], chain[k]) for k in range(len(chain) - 1)]
            if all(mesh.sound(triangle) for triangle in fan):
                placed = True
                break
        if not placed:
            del mesh.p[apex], mesh.n[apex], mesh.size[apex]
            for triangle in old:
                mesh.add(triangle)
            raise ValueError("A border vertex could not be prepared for an all-quad patch")
        for triangle in fan:
            mesh.add(triangle)
        ready.add(v)
    for v, (left, right) in pinned.items():
        apex = mesh.neighbours(v) - {left, right}
        if len(mesh.vf[v]) != 2 or len(apex) != 1 or next(iter(apex)) in border:
            raise ValueError("The border is too tight for an all-quad patch")

    # At an outward corner the two neighbouring pairs share one apex.
    following = {left: v for v, (left, right) in pinned.items()}
    for v, (left, corner) in pinned.items():
        other = following.get(corner)
        if other is None or other == v:
            continue
        if not _shares_apex(mesh, pinned, v, corner, other):
            continue
        keep = next(iter(mesh.neighbours(v) - {left, corner}))
        for _ in range(8):
            last = next(iter(mesh.neighbours(other) - set(pinned[other])))
            if last == keep:
                break
            # Walk the corner's fan from the kept apex towards the other one.
            step = [mesh.opposite(i, corner, keep) for i in mesh.vf[corner]
                    if keep in mesh.f[i] and mesh.opposite(i, corner, keep) not in (v, other)]
            step = [n for n in step if n not in border]
            if not step or not _collapse(mesh, step[0], keep):
                break

    def mend_slivers():
        """Remove flat triangles: turn their longest edge, or close their shortest."""
        for _ in range(3):
            changed = False
            for i in list(mesh.f):
                if i not in mesh.f:
                    continue
                triangle = mesh.f[i]
                sides = sorted(((mesh.length(a, b), a, b) for a, b in directed(triangle)))
                area = np.linalg.norm(mesh.normal(triangle))
                if area >= 0.12 * sides[2][0] ** 2:
                    continue
                _, a, b = sides[2]
                if len(mesh.edge_faces(a, b)) == 2 and not locked(a, b) and _flip(mesh, a, b, border, force=True):
                    changed = True
                    continue
                _, a, b = sides[0]
                if len(mesh.edge_faces(a, b)) != 2 or locked(a, b):
                    continue
                for gone, kept in ((a, b), (b, a)):
                    if gone in border or (kept in border and serves_border(gone)):
                        continue
                    if _collapse(mesh, gone, kept):
                        changed = True
                        break
            if not changed:
                break

    # Edge length grows steadily from the border's own spacing to the
    # requested size, so a fine border is not met by a ring of long slivers.
    border_ids = sorted(border)
    if border_ids:
        spacing = {}
        for loop in loops:
            for i, v in enumerate(loop):
                spacing[v] = mesh.length(loop[i - 1], v) + mesh.length(v, loop[(i + 1) % len(loop)])
        border_tree = cKDTree(np.array([mesh.p[v] for v in border_ids]))
        border_spacing = np.array([spacing[v] for v in border_ids])

    # Distance from each pinned vertex to the nearest border that is not its own.
    room = {}
    if len(loops) > 1:
        loop_of = {v: n for n, loop in enumerate(loops) for v in loop}
        k = min(len(border_ids), 48)
        for v in pinned:
            distance, near = border_tree.query(mesh.p[v], k=k)
            for d, i in zip(np.atleast_1d(distance), np.atleast_1d(near)):
                if loop_of[border_ids[i]] != loop_of[v]:
                    room[v] = float(d)
                    break

    def refresh_sizes():
        ids = list(mesh.p)
        if not border_ids:
            mesh.size.update((v, size) for v in ids)
            return
        distance, near = border_tree.query(np.array([mesh.p[v] for v in ids]))
        local = np.minimum(size, 1.1 * border_spacing[near] + 0.7 * distance)
        mesh.size.update(zip(ids, local.tolist()))

    def wanted(a, b):
        return 0.5 * (mesh.size[a] + mesh.size[b])

    for _ in range(6):
        refresh_sizes()
        for a, b in sorted(mesh.edges(), key=lambda e: -mesh.length(*e)):
            if a not in mesh.p or b not in mesh.p or len(mesh.edge_faces(a, b)) != 2:
                continue
            if mesh.length(a, b) <= 4.0 / 3.0 * wanted(a, b) or locked(a, b):
                continue
            _split(mesh, a, b, 0.5 * (mesh.p[a] + mesh.p[b]))
        for a, b in sorted(mesh.edges(), key=lambda e: mesh.length(*e)):
            if a not in mesh.p or b not in mesh.p or len(mesh.edge_faces(a, b)) != 2:
                continue
            if mesh.length(a, b) >= 0.8 * wanted(a, b) or locked(a, b):
                continue
            high = 4.0 / 3.0 * wanted(a, b)
            if a in border and b in border:
                continue
            if b in border:
                if not serves_border(a):
                    _collapse(mesh, a, b, high)
            elif a in border:
                if not serves_border(b):
                    _collapse(mesh, b, a, high)
            elif not _collapse(mesh, a, b, high):
                _collapse(mesh, b, a, high)
        for a, b in sorted(mesh.edges()):
            if a in mesh.p and b in mesh.p and len(mesh.edge_faces(a, b)) == 2 and not locked(a, b):
                _flip(mesh, a, b, border)
        mend_slivers()
        _relax(mesh, surface, [v for v in mesh.p if v not in border])
        _seat_apexes(mesh, surface, pinned, room)
    mend_slivers()
    for v, (left, right) in pinned.items():
        apex = mesh.neighbours(v) - {left, right}
        if len(mesh.vf[v]) != 2 or len(apex) != 1 or next(iter(apex)) in border:
            raise ValueError("The border structure was lost while rebuilding the patch")

    # Three quads per triangle.
    positions = [tuple(p) for p in vertices]
    index = {}

    def output(v):
        if v in border_all:
            return v
        if v not in index:
            index[v] = len(positions)
            positions.append(tuple(mesh.p[v]))
        return index[v]

    new_points, new_normals, new_ids = [], [], []

    def add_point(point, normal):
        positions.append(None)
        new_ids.append(len(positions) - 1)
        new_points.append(point)
        new_normals.append(normal)
        return len(positions) - 1

    middle = {}
    for a, b in sorted(mesh.edges()):
        if a in pinned or b in pinned:
            continue
        normal = _unit(sum(mesh.normal(mesh.f[i]) for i in mesh.edge_faces(a, b)))
        middle[a, b] = add_point(0.5 * (mesh.p[a] + mesh.p[b]), normal)
    quads = []
    centres = []
    for triangle in mesh.f.values():
        if any(v in pinned for v in triangle):
            continue
        centre = add_point(sum(mesh.p[v] for v in triangle) / 3.0, _unit(mesh.normal(triangle)))
        for k in range(3):
            v, nxt, prev = triangle[k], triangle[(k + 1) % 3], triangle[k - 1]
            quads.append((output(v), middle[edge(v, nxt)], centre, middle[edge(prev, v)]))
    for v, (left, right) in pinned.items():
        apex = next(iter(mesh.neighbours(v) - {left, right}))
        normal = _unit(sum(mesh.normal(mesh.f[i]) for i in mesh.vf[v]))
        centre = add_point(mesh.p[v] + (mesh.p[apex] - mesh.p[v]) / 3.0, normal)
        pieces = [(left, v, centre, middle[edge(left, apex)]),
                  (v, right, middle[edge(right, apex)], centre),
                  (output(apex), middle[edge(left, apex)], centre, middle[edge(right, apex)])]
        quads.extend(pieces)
        centres.append((centre, len(new_points) - 1, pieces, v, apex))
    projected, _ = surface.project(np.array(new_points), np.array(new_normals))
    for i, point in zip(new_ids, projected):
        positions[i] = tuple(point)
    # Put the centre of each border pair where its three quads are best.
    trial_points, trial_normals, trial_owner = [], [], []
    for centre, slot, pieces, v, apex in centres:
        for fraction in (0.2, 0.27, 0.4, 0.5, 0.6):
            trial_points.append(mesh.p[v] + fraction * (mesh.p[apex] - mesh.p[v]))
            trial_normals.append(new_normals[slot])
            trial_owner.append((centre, pieces))
    if trial_points:
        projected, _ = surface.project(np.array(trial_points), np.array(trial_normals))
        best = {}
        for (centre, pieces), point in zip(trial_owner, projected):
            keep = positions[centre]
            if centre not in best:
                best[centre] = (min(quad_quality(positions, q) for q in pieces), keep)
            positions[centre] = tuple(point)
            score = min(quad_quality(positions, q) for q in pieces)
            positions[centre] = keep
            if score > best[centre][0]:
                best[centre] = (score, tuple(point))
        for centre, (_, point) in best.items():
            positions[centre] = point
    positions = _smooth_quads(positions, quads, border_all, surface)
    return positions, quads


def _smooth_quads(positions, quads, fixed, surface):
    points = {v: np.asarray(positions[v], dtype=float) for q in quads for v in q}
    ring = defaultdict(set)
    star = defaultdict(list)
    for i, quad in enumerate(quads):
        for a, b in directed(quad):
            ring[a].add(b)
            ring[b].add(a)
        for v in quad:
            star[v].append(i)
    movable = sorted(v for v in points if v not in fixed)

    def quality(v):
        return min(quad_quality(points, quads[i]) for i in star[v])

    def normal(v):
        total = np.zeros(3)
        for i in star[v]:
            a, b, c, d = (points[n] for n in quads[i])
            total += np.cross(c - a, d - b)
        return _unit(total)

    for _ in range(5):
        targets, normals = [], []
        for v in movable:
            n = normal(v)
            step = np.mean([points[m] for m in ring[v]], axis=0) - points[v]
            step -= n * np.dot(step, n)
            targets.append(points[v] + 0.5 * step)
            normals.append(n)
        if not targets:
            break
        projected, _ = surface.project(np.array(targets), np.array(normals))
        for v, point in zip(movable, projected):
            old, before = points[v], quality(v)
            points[v] = point
            after = quality(v)
            if after < before and after < 0.35:
                points[v] = old
    # Repair pass: a vertex of a poor quad takes whichever nearby position
    # gives its worst quad the best shape.
    for _ in range(16):
        poor = {v for i, quad in enumerate(quads) if quad_quality(points, quad) < 0.3
                for v in quad if v not in fixed}
        if not poor:
            break
        order = sorted(poor)
        trial_points, trial_normals, owner = [], [], []
        for v in order:
            n = normal(v)
            goals = [np.mean([points[m] for m in ring[v]], axis=0),
                     np.mean([points[m] for i in star[v] for m in quads[i]], axis=0)]
            for goal in goals:
                for fraction in (0.3, 0.6, 1.0):
                    trial_points.append(points[v] + fraction * (goal - points[v]))
                    trial_normals.append(n)
                    owner.append(v)
            # Also probe around the vertex itself, in the surface plane.
            reach = min(np.linalg.norm(points[m] - points[v]) for m in ring[v])
            side = _unit(np.cross(n, points[next(iter(ring[v]))] - points[v]))
            up = np.cross(n, side)
            for angle in range(8):
                heading = math.cos(angle * math.pi / 4) * side + math.sin(angle * math.pi / 4) * up
                for fraction in (0.2, 0.5):
                    trial_points.append(points[v] + fraction * reach * heading)
                    trial_normals.append(n)
                    owner.append(v)
        projected, _ = surface.project(np.array(trial_points), np.array(trial_normals))
        changed = False
        for v in order:
            best, chosen = quality(v), points[v]
            for w, point in zip(owner, projected):
                if w != v:
                    continue
                points[v] = point
                score = quality(v)
                if score > best + 1e-6:
                    best, chosen = score, point
                    changed = True
            points[v] = chosen
        if not changed:
            break
    result = list(positions)
    for v in movable:
        result[v] = tuple(points[v])
    return result


def ease_faces(positions, quads, flagged, fixed, source_vertices, source_faces):
    """Relax the vertices of flagged quads and of their neighbours on the surface.

    Used when a finished patch still has a pair of quads that touch: pulling
    the few vertices involved towards their neighbours opens the pair up.
    """
    triangles = [t for face in source_faces for t in polygon_triangles(source_vertices, face)]
    points = np.array(source_vertices, dtype=float)
    spacing = np.median([np.linalg.norm(np.asarray(positions[q[0]]) - np.asarray(positions[q[1]])) for q in quads])
    surface = Surface(points, np.array(triangles), spacing / 2.0)
    ring = defaultdict(set)
    star = defaultdict(list)
    for i, quad in enumerate(quads):
        for a, b in directed(quad):
            ring[a].add(b)
            ring[b].add(a)
        for v in quad:
            star[v].append(i)
    chosen = {v for i in flagged for v in quads[i]}
    chosen |= {n for v in list(chosen) for n in ring[v]}
    chosen = sorted(v for v in chosen if v not in fixed)
    result = {v: np.asarray(positions[v], dtype=float) for q in quads for v in q}
    for _ in range(4):
        targets, normals = [], []
        for v in chosen:
            normal = np.zeros(3)
            for i in star[v]:
                a, b, c, d = (result[n] for n in quads[i])
                normal += np.cross(c - a, d - b)
            normal = _unit(normal)
            step = np.mean([result[m] for m in ring[v]], axis=0) - result[v]
            step -= normal * np.dot(step, normal)
            targets.append(result[v] + 0.6 * step)
            normals.append(normal)
        if not targets:
            break
        projected, _ = surface.project(np.array(targets), np.array(normals))
        for v, point in zip(chosen, projected):
            result[v] = point
    eased = list(positions)
    for v in chosen:
        eased[v] = tuple(result[v])
    return eased
