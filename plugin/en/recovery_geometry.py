from collections import Counter, defaultdict
from functools import lru_cache
from itertools import product
import math

import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
import lqr_validate as v

def edge(a, b):
    return min(a,b), max(a,b)

def directed(face):
    return list(zip(face, face[1:] + face[:1]))

def union_loop(faces):
    # Cancel reversed internal edges and trace the single directed perimeter.
    perimeter = set()
    for f in faces:
        for a,b in directed(f):
            if (b,a) in perimeter:
                perimeter.remove((b,a))
            else:
                perimeter.add((a,b))
    successor = dict(perimeter)
    if len(successor) != len(perimeter) or not successor:
        return ()
    start = min(successor)
    loop = [start]
    while len(loop) <= len(successor):
        nxt = successor.get(loop[-1])
        if nxt == start:
            return tuple(loop) if len(loop) == len(successor) else ()
        if nxt is None or nxt in loop:
            break
        loop.append(nxt)
    return ()

def quad_quality(vertices, quad):
    points = [vertices[i] for i in quad]
    edges = [v.vsub(b,a) for a,b in directed(points)]
    lens = [v.vlen(e) for e in edges]
    if min(lens) == 0:
        return -1.0
    turns = [v.vcross(edges[i-1], edges[i]) for i in range(4)]
    avg = tuple(sum(n[k] for n in turns) for k in range(3))
    avg_len = v.vlen(avg)
    if avg_len == 0:
        return -1.0
    jacobian = min(v.vdot(avg, n)/(avg_len*lens[i-1]*lens[i])
                   for i,n in enumerate(turns))
    return jacobian * math.sqrt(min(lens)/max(lens)) if jacobian > 0 else jacobian

def polygon_triangles(vertices, face):
    """Ear clipping for a simple polygon, with its original winding."""
    if len(face) <= 4:
        return [(face[0], face[i], face[i + 1]) for i in range(1, len(face) - 1)]
    normal = newell(vertices, face)
    axis = max(range(3), key=lambda i: abs(normal[i]))
    u, w = (axis + 1) % 3, (axis + 2) % 3
    points = {i: (vertices[i][u], vertices[i][w]) for i in face}
    orientation = 1 if normal[axis] > 0 else -1
    scale = max(max(p[k] for p in points.values()) - min(p[k] for p in points.values())
                for k in (0, 1))
    epsilon = scale * scale * 1e-12

    def turn(a, b, c):
        x, y, z = points[a], points[b], points[c]
        return orientation * ((y[0] - x[0]) * (z[1] - x[1])
                              - (y[1] - x[1]) * (z[0] - x[0]))

    remaining = list(face)
    triangles = []
    while len(remaining) > 3:
        found = False
        for i, b in enumerate(remaining):
            a, c = remaining[i - 1], remaining[(i + 1) % len(remaining)]
            if turn(a, b, c) <= epsilon:
                continue
            inside = any(turn(a, b, p) >= -epsilon and turn(b, c, p) >= -epsilon
                         and turn(c, a, p) >= -epsilon
                         for p in remaining if p not in (a, b, c))
            if inside:
                continue
            triangles.append((a, b, c))
            remaining.pop(i)
            found = True
            break
        if not found:
            raise ValueError("Selected polygon is self-crossing or has a degenerate boundary")
    triangles.append(tuple(remaining))
    return triangles


def quadrangulate(vertices, faces, selected, partial=False):
    # Pair triangles sharing an internal edge. Prefer keeping sound input quads
    # where the exact matching permits it. Each triangle must be used once.
    tris = []
    by_edge = defaultdict(list)
    source = []
    for i in sorted(selected):
        f = faces[i]
        rim = {edge(a, b) for a, b in directed(f)}
        for t in polygon_triangles(vertices, f):
            tid = len(tris)
            for a,b in directed(t):
                # A diagonal inside one source face belongs to that face only.
                # Two faces that span the same pair of opposite corners must
                # not be mistaken for four triangles around one real edge.
                key = edge(a,b)
                by_edge[key if key in rim else (key, i)].append(tid)
            tris.append(t)
            source.append(i)
    candidates = []
    weights = []
    quads = {}
    for key, owners in by_edge.items():
        if len(owners) != 2:
            continue
        a, b = key[0] if isinstance(key[0], tuple) else key
        i,j = owners
        same_quad = source[i] == source[j] and len(faces[source[i]]) == 4
        quad = faces[source[i]] if same_quad else union_loop([tris[i],tris[j]])
        if len(quad) != 4:
            continue
        quality = quad_quality(vertices, quad)
        if quality < 0.07 and not same_quad:
            continue
        if not same_quad and quad.index(a) % 2:
            quad = quad[1:]+quad[:1]
        candidates.append((i,j))
        weights.append(round(1000*quality)+int(same_quad)*1500)
        quads[edge(i,j)] = quad
    if partial and not candidates:
        return tris
    if not candidates or (len(tris) % 2 and not partial):
        raise ValueError("No all-quad pairing with the fixed border")
    cols = np.repeat(np.arange(len(candidates)), 2)
    rows = np.array(candidates).ravel()
    matrix = coo_matrix((np.ones(len(rows)), (rows,cols)), shape=(len(tris),len(candidates))).tocsc()
    scores = np.array(weights)
    if partial:
        difficult = []
        for triangle in tris:
            longest = max(v.vlen(v.vsub(vertices[a], vertices[b]))
                          for a, b in directed(triangle))
            ratio = v.polygon_area(vertices, triangle) / longest ** 2 if longest else 0
            difficult.append(10000 if ratio < 0.03 else 0)
        scores = scores + np.array([difficult[i] + difficult[j] for i, j in candidates])
        scores = scores + 30000 * len(tris)
    result = milp(-scores, integrality=np.ones(len(candidates)),
                  bounds=Bounds(0, 1), constraints=LinearConstraint(matrix, 0 if partial else 1, 1),
                  options={"time_limit": 8})
    if result.x is None:
        raise ValueError("No valid all-quad pairing with the fixed border: " + result.message)
    matched = [pair for pair,x in zip(candidates,result.x) if x > 0.5]
    used = {i for pair in matched for i in pair}
    if len(used) != 2 * len(matched) or (not partial and len(used) != len(tris)):
        raise ValueError("Incomplete triangle pairing")
    result_faces = [quads[edge(i,j)] for i,j in sorted(matched)]
    if partial:
        result_faces.extend(t for i, t in enumerate(tris) if i not in used)
    return result_faces


@lru_cache(maxsize=2048)
def quad_tilings(loop):
    if len(loop) == 2:
        return ((),)
    options = []
    for i in range(1,len(loop)-2,2):
        for j in range(i+1,len(loop)-1,2):
            for k in range(j+1,len(loop),2):
                q = loop[0], loop[i], loop[j], loop[k]
                for pieces in product(quad_tilings(loop[:i+1]),
                                      quad_tilings(loop[i:j+1]),
                                      quad_tilings(loop[j:k+1]),
                                      quad_tilings(loop[k:]+loop[:1])):
                    options.append((q,)+sum(pieces, ()))
    return tuple(options)

def newell(vertices, loop):
    normals = [v.vcross(vertices[a],vertices[b]) for a,b in directed(loop)]
    return tuple(sum(n[k] for n in normals) for k in range(3))

def coarsen(vertices, quads, frozen, target):
    # Collapse an interior vertex only when the whole one-ring has a clean
    # quadrangulation. The surrounding edges and their endpoints stay fixed.
    quads = dict(enumerate(quads))
    incident = defaultdict(set)
    for i, q in quads.items():
        for p in q:
            incident[p].add(i)
    next_id = len(quads)
    edge_use = Counter(edge(a, b) for q in quads.values() for a, b in directed(q))

    def new_edges(loop, tiles):
        rim = {edge(a, b) for a, b in directed(loop)}
        return {edge(a, b) for q in tiles for a, b in directed(q)} - rim

    while len(quads) > target:
        candidates = []
        for p in sorted(incident):
            if p in frozen or len(incident[p]) not in (3,4):
                continue
            ids = sorted(incident[p])
            star = [quads[i] for i in ids]
            loop = union_loop(star)
            if len(loop) != 2*len(star) or p in loop:
                continue
            if any(n in frozen for n in loop):
                continue
            normal = newell(vertices, loop)
            normal_length = v.vlen(normal)
            if normal_length == 0:
                continue
            if any(v.vdot(n,normal) < 0.985*v.vlen(n)*normal_length
                   for n in (newell(vertices,q) for q in star)):
                continue
            old_quality = min(quad_quality(vertices,q) for q in star)
            best = None
            for tiles in quad_tilings(loop):
                score = min(quad_quality(vertices,q) for q in tiles)
                if score < max(0.30, old_quality*0.60):
                    continue
                # A new inner edge must not duplicate an edge that already
                # exists outside the star, or three faces would share it.
                if any(edge_use[e] for e in new_edges(loop, tiles)):
                    continue
                if best is None or score > best[0]:
                    best = score, tiles
            if best is not None:
                candidates.append((best[0],p,ids,best[1],loop))
        if not candidates:
            break
        progressed = False
        for _, p, ids, tiles, loop in sorted(candidates,reverse=True):
            if len(quads) <= target:
                break
            if any(i not in quads for i in ids):
                continue
            if any(edge_use[e] for e in new_edges(loop, tiles)):
                continue
            progressed = True
            for i in ids:
                removed = quads.pop(i)
                edge_use.subtract(edge(a, b) for a, b in directed(removed))
                for n in removed:
                    incident[n].remove(i)
            del incident[p]
            for q in tiles:
                quads[next_id] = q
                edge_use.update(edge(a, b) for a, b in directed(q))
                for n in q:
                    incident[n].add(next_id)
                next_id += 1
        if not progressed:
            break
    return list(quads.values())



def midpoint(vertices, a, b):
    return tuple((x + y) * 0.5 for x, y in zip(vertices[a], vertices[b]))


def split_triangles_on_edge(face, seam_edge, middle):
    pieces = []
    for i in range(1, len(face) - 1):
        tri = (face[0], face[i], face[i + 1])
        if not set(seam_edge).issubset(tri):
            pieces.append(tri)
            continue
        for j, a in enumerate(tri):
            b, c = tri[(j + 1) % 3], tri[(j + 2) % 3]
            if edge(a, b) == seam_edge:
                pieces.extend([(a, middle, c), (middle, b, c)])
                break
    return pieces


def refine_faces(vertices, faces, split):
    """Subdivide marked edges on both incident faces, keeping every other edge."""
    positions = list(vertices)
    new_points = {}
    for a, b in sorted(split):
        new_points[a, b] = len(positions)
        positions.append(midpoint(positions, a, b))

    output = []
    for face in faces:
        perimeter = []
        added = []
        for a, b in directed(face):
            perimeter.append(a)
            point = new_points.get(edge(a, b))
            if point is not None:
                added.append(len(perimeter))
                perimeter.append(point)
        if not added:
            if len(face) != 4:
                raise ValueError("Unresolved triangle in the fallback patch")
            output.append(face)
            continue
        if len(perimeter) % 2:
            raise ValueError("Nonconforming split inside the patch")
        if len(face) == 4 and len(added) == 2 and added[1] - added[0] == 3:
            start = added[0]
            ring = perimeter[start:] + perimeter[:start]
            pieces = (tuple(ring[:4]), (ring[3], ring[4], ring[5], ring[0]))
            if min(quad_quality(positions, q) for q in pieces) > 0.07:
                output.extend(pieces)
                continue

        center = tuple(sum(positions[i][axis] for i in face) / len(face)
                       for axis in range(3))
        if len(face) == 3 and len(added) == 1:
            # A single subdivided triangle edge has a collinear boundary
            # corner. An inset plus a center quad retains the planar surface.
            inner = []
            for i, corner in enumerate(perimeter):
                scale = 0.77 if i == added[0] else 0.52
                inner.append(len(positions))
                positions.append(tuple(c + scale * (p - c)
                                       for c, p in zip(center, positions[corner])))
            for i, a in enumerate(perimeter):
                j = (i + 1) % 4
                output.append((a, perimeter[j], inner[j], inner[i]))
            output.append(tuple(inner))
            continue

        centers = [center]
        if len(face) == 4:
            # Test points on the source triangles as well as the diagonal.
            diagonal = midpoint(positions, face[0], face[2])
            centers = [diagonal]
            for corner in (face[1], face[3]):
                for fraction in (0.2, 0.4, 0.6):
                    centers.append(tuple(a + fraction * (b - a)
                                         for a, b in zip(diagonal, positions[corner])))
        center_id = len(positions)
        positions.append(center)
        count = len(perimeter)
        variants = [
            [(center_id, perimeter[i], perimeter[(i + 1) % count],
              perimeter[(i + 2) % count]) for i in range(start, count, 2)]
            for start in (0, 1)
        ]
        # At a newly inserted edge midpoint the two original endpoints
        # must never occur as its two neighbours in the same quad.
        edge_midpoints = {perimeter[i] for i in added}
        allowed = [variant for variant in variants
                   if all(quad[2] not in edge_midpoints for quad in variant)]
        if not allowed:
            allowed = variants
        best = None
        for point in centers:
            positions[center_id] = point
            for variant in allowed:
                score = min(quad_quality(positions, q) for q in variant)
                if best is None or score > best[0]:
                    best = score, point, variant
        positions[center_id] = best[1]
        output.extend(best[2])
    return positions, output


def parity_quads(vertices, faces):
    """Pick shared edges whose splits make each face perimeter even."""
    owners = defaultdict(list)
    for i, face in enumerate(faces):
        for a, b in directed(face):
            owners[edge(a, b)].append(i)
    links = defaultdict(list)
    for shared, touching in owners.items():
        if len(touching) == 2:
            a, b = touching
            links[a].append((b, shared))
            links[b].append((a, shared))
    visited = set()
    odd = [len(f) % 2 for f in faces]
    split = set()
    for root in range(len(faces)):
        if root in visited:
            continue
        visited.add(root)
        queue = [(root, None, None)]
        for i, _, _ in queue:
            for j, shared in links[i]:
                if j not in visited:
                    visited.add(j)
                    queue.append((j, i, shared))
        for i, parent, shared in reversed(queue[1:]):
            if odd[i]:
                split.add(shared)
                odd[parent] ^= 1
        if odd[root]:
            raise ValueError("Selected component has an odd number of frozen boundary edges")
    return refine_faces(vertices, faces, split)


def refine_quads(vertices, faces, fixed, target):
    """Increase density along closed paths entirely within the mask."""
    positions = list(vertices)
    patch = faces
    while len(patch) < target:
        owners = Counter(edge(a, b) for f in patch for a, b in directed(f))
        stars = defaultdict(set)
        neighbours = defaultdict(set)
        for a, b in owners:
            neighbours[a].add(b)
            neighbours[b].add(a)
        for i, face in enumerate(patch):
            for point in face:
                stars[point].add(i)
        eligible = [point for point, near in neighbours.items()
                    if point not in fixed and all(owners[edge(point, other)] == 2 for other in near)]
        eligible.sort(key=lambda point: (-max(v.polygon_area(positions, patch[i])
                                             for i in stars[point]), point))
        split = set()
        blocked = set()
        added_faces = 0
        for point in eligible:
            if point in blocked:
                continue
            if added_faces and len(patch) + added_faces >= target:
                break
            split.update(edge(point, other) for other in neighbours[point])
            blocked.update(neighbours[point])
            added_faces += 2 * len(stars[point])
        if not split:
            # A single quad, or a narrow strip, may have no interior vertex.
            # Inset one face without dividing its shared boundary edges.
            which = max(range(len(patch)), key=lambda i: v.polygon_area(positions, patch[i]))
            face = patch[which]
            center = midpoint(positions, face[0], face[2])
            inner = []
            for corner in face:
                inner.append(len(positions))
                positions.append(tuple(a + 0.5 * (b - a) for a, b in zip(center, positions[corner])))
            tiles = [(face[i], face[(i + 1) % 4], inner[(i + 1) % 4], inner[i])
                     for i in range(4)] + [tuple(inner)]
            patch = patch[:which] + tiles + patch[which + 1:]
        else:
            positions, patch = refine_faces(positions, patch, split)
    return positions, patch



def project_surface(point, triangles):
    a,b,c=triangles[:,0],triangles[:,1],triangles[:,2]
    ab,ac=b-a,c-a
    ap=point-a
    d00=np.einsum("ij,ij->i",ab,ab); d01=np.einsum("ij,ij->i",ab,ac)
    d11=np.einsum("ij,ij->i",ac,ac); d20=np.einsum("ij,ij->i",ap,ab)
    d21=np.einsum("ij,ij->i",ap,ac)
    den=d00*d11-d01*d01
    good=np.abs(den)>1e-30
    u=np.divide(d11*d20-d01*d21,den,out=np.zeros_like(den),where=good)
    w=np.divide(d00*d21-d01*d20,den,out=np.zeros_like(den),where=good)
    normal=a+u[:,None]*ab+w[:,None]*ac
    dist=np.sum((normal-point)**2,axis=1)
    dist[~(good & (u>=0) & (w>=0) & (u+w<=1))]=np.inf
    candidates=[normal]; distances=[dist]
    for x,y in [(a,b),(b,c),(c,a)]:
        xy=y-x
        num=np.einsum("ij,ij->i",point-x,xy); denominator=np.einsum("ij,ij->i",xy,xy)
        t=np.clip(np.divide(num,denominator,out=np.zeros_like(num),where=denominator>0),0,1)
        p=x+t[:,None]*xy
        candidates.append(p); distances.append(np.sum((p-point)**2,axis=1))
    kind,index=divmod(np.argmin(np.asarray(distances)),len(triangles))
    return tuple(candidates[kind][index])



def extreme_faces(vertices, faces):
    result = []
    for i, face in enumerate(faces):
        longest = max(v.vlen(v.vsub(vertices[a], vertices[b])) for a, b in directed(face))
        ratio = v.polygon_area(vertices, face) / longest ** 2 if longest else 0.0
        if ratio < 0.02:
            result.append(i)
    return result


def relax_slivers(vertices, faces, fixed, source_vertices, source_faces):
    """Improve new interior slivers with local moves projected onto the source."""
    bad = extreme_faces(vertices, faces)
    if not bad:
        return vertices
    triangles = np.asarray([[source_vertices[f[0]], source_vertices[f[i]], source_vertices[f[i + 1]]]
                            for f in source_faces for i in range(1, len(f) - 1)])
    positions = list(vertices)
    original = list(vertices)
    incident = defaultdict(set)
    neighbours = defaultdict(set)
    for i, face in enumerate(faces):
        for a, b in directed(face):
            incident[a].add(i)
            incident[b].add(i)
            neighbours[a].add(b)
            neighbours[b].add(a)
    region = {point for i in bad for point in faces[i]}
    for _ in range(2):
        region.update(point for old in region.copy() for point in neighbours[old])
    movable = sorted(region - fixed)
    lengths = [v.vlen(v.vsub(positions[a], positions[b])) for face in faces for a, b in directed(face)]
    limit = float(np.median(lengths)) * 0.75
    for _ in range(30):
        moved = False
        for point_id in movable:
            local = sorted(incident[point_id])
            previous = positions[point_id]
            average = np.mean([positions[n] for n in neighbours[point_id]], axis=0)
            best = min(quad_quality(positions, faces[i]) for i in local)
            chosen = previous
            for fraction in (0.2, 0.5, 1.0):
                point = project_surface(np.asarray(previous) + fraction * (average - previous), triangles)
                if np.linalg.norm(np.asarray(point) - original[point_id]) > limit:
                    continue
                positions[point_id] = point
                score = min(quad_quality(positions, faces[i]) for i in local)
                if score > best + 1e-7:
                    best, chosen = score, point
            positions[point_id] = chosen
            moved |= chosen != previous
        if not moved:
            break
    return positions
