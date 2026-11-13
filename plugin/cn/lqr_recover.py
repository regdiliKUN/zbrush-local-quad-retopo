"""Recover a local all-quad OBJ patch with a fixed, verified seam."""
import argparse
from collections import Counter, defaultdict
import json
import re
from pathlib import Path
import sys

import lqr_validate as validation
from recovery_geometry import (coarsen, directed, edge, midpoint, parity_quads,
                               polygon_triangles, quadrangulate, refine_quads, split_triangles_on_edge,
                               extreme_faces, relax_slivers)


def read_obj(path):
    stats = {}
    vertices, faces, numbers = validation.parse_obj(path, stats)
    if (not faces or stats["bad_faces"] or stats["bad_vertices"]
            or any(i < 0 or i >= len(vertices) for f in faces for i in f)):
        raise ValueError(f"Malformed or empty OBJ: {path}")
    return vertices, faces, numbers


def face_key(vertices, face):
    points = tuple(vertices[i] for i in face)
    return min(points[k:] + points[:k] for k in range(len(points)))


def identify(vertices, faces, other_vertices, other_faces, is_selection):
    known = Counter(face_key(other_vertices, f) for f in other_faces)
    selected = set()
    for i, f in enumerate(faces):
        key = face_key(vertices, f)
        found = known[key] > 0
        if found:
            known[key] -= 1
        if found == is_selection:
            selected.add(i)
    if is_selection and any(known.values()):
        raise ValueError("Selection contains faces that do not exactly match the source")
    return selected


def boundary(faces):
    occurrences = Counter((a,b) for f in faces for a,b in directed(f))
    undirected = Counter(edge(a,b) for a,b in occurrences.elements())
    if any(n > 2 or (n == 2 and occurrences[a,b] != occurrences[b,a])
           for (a,b),n in undirected.items()):
        raise ValueError("Selected faces must be consistently oriented and manifold along edges")
    return {(a,b) for a,b in occurrences if undirected[edge(a,b)] == 1}


def stray_faces(faces, selected):
    """Selected faces that are not part of the 2-manifold surface.

    A flap is a face hanging off an edge that already joins two other faces;
    a floater touches the rest of the mesh by vertices only. Neither can be
    paired into the quad patch, and both are leftovers of earlier edits, so
    the patch is rebuilt without them.
    """
    owners = defaultdict(list)
    for i, face in enumerate(faces):
        for a, b in directed(face):
            owners[edge(a, b)].append(i)
    stray = set()
    while True:
        found = set()
        for i in selected - stray:
            counts = [sum(j not in stray for j in owners[edge(a, b)])
                      for a, b in directed(faces[i])]
            open_edges = sum(n == 1 for n in counts)
            if open_edges == len(counts):
                found.add(i)
            elif open_edges and any(n > 2 for n in counts) and all(n != 2 for n in counts):
                found.add(i)
        if not found:
            return stray
        stray |= found


def unpinch(faces, selected):
    """Leave out masked faces where the mask border meets itself at one vertex.

    A painted mask often touches itself diagonally across a vertex. The patch
    border would then pass that vertex twice, which no fixed-border rebuild
    can handle. Those few faces stay as they are instead.
    """
    selected = set(selected)
    start = len(selected)
    for _ in range(40):
        outgoing = Counter()
        owners = Counter(edge(a, b) for i in selected for a, b in directed(faces[i]))
        for i in selected:
            for a, b in directed(faces[i]):
                if owners[edge(a, b)] == 1:
                    outgoing[a] += 1
        pinched = {v for v, n in outgoing.items() if n > 1}
        if not pinched:
            break
        selected -= {i for i in selected if any(v in pinched for v in faces[i])}
        if not selected:
            raise ValueError("The mask border touches itself everywhere")
    else:
        raise ValueError("The mask border keeps touching itself")
    if len(selected) < 0.9 * start:
        raise ValueError("The mask border touches itself in too many places")
    return selected


def seam_geometry(vertices, faces):
    return Counter((vertices[a],vertices[b]) for a,b in boundary(faces))


def source_records(path, face_lines):
    rows = path.read_text(encoding="utf-8-sig").splitlines()
    if any(row.lstrip().startswith(("#MRGB", "vt ", "vn ", "usemtl ", "mtllib "))
           for row in rows):
        raise ValueError("This recovery build supports geometry and OBJ groups only")
    vertex_rows = []
    group_for_face = {}
    group = "g"
    face_line_set = set(face_lines)
    for number, row in enumerate(rows, 1):
        fields = row.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertex_rows.append(row)
        elif fields[0] == "g":
            group = row
        if number in face_line_set:
            group_for_face[number] = group
    return vertex_rows, group_for_face


def face_center(vertices, face):
    return tuple(sum(vertices[i][axis] for i in face) / len(face) for axis in range(3))


def inherit_patch_groups(vertices, source, candidate, source_groups, seam):
    """Carry the source grouping across the mask boundary."""
    from scipy.spatial import cKDTree

    tree = cKDTree([face_center(vertices, face) for face in source])
    _, neighbours = tree.query([face_center(vertices, face) for face in candidate])
    original = {frozenset(face): group for face, group in zip(source, source_groups)}
    border_groups = {e: group for face, group in zip(source, source_groups)
                     for e in directed(face) if e in seam}
    groups = []
    for face, nearest in zip(candidate, neighbours):
        group = original.get(frozenset(face))
        if group is None:
            votes = Counter()
            for a, b in directed(face):
                if (a, b) in border_groups:
                    length = validation.vlen(validation.vsub(vertices[a], vertices[b]))
                    votes[border_groups[a, b]] += length
            group = votes.most_common(1)[0][0] if votes else source_groups[nearest]
        groups.append(group)
    return groups


def build_obj(vertex_rows, unchanged, patch, groups):
    used = sorted({i for f in unchanged + patch for i in f})
    index = {original: out for out, original in enumerate(used, 1)}
    lines = ["# LocalQuadRetopo recovery", *(vertex_rows[i] for i in used)]
    current = None
    for group, face in zip(groups, unchanged + patch):
        if group != current:
            lines.append(group)
            current = group
        lines.append("f " + " ".join(str(index[i]) for i in face))
    return "\n".join(lines) + "\n", used, index


def check_shared_edges(vertices, faces, active, tolerance):
    """Detect coplanar overlap between neighbours skipped by the base scan."""
    triangles = defaultdict(list)
    for i, face in enumerate(faces):
        for j in range(1, len(face) - 1):
            indices = face[0], face[j], face[j + 1]
            tri = tuple(vertices[n] for n in indices)
            for a, b in directed(indices):
                triangles[edge(a, b)].append((i, tri))
    for entries in triangles.values():
        for pos, (i, first) in enumerate(entries):
            for j, second in entries[pos + 1:]:
                if i == j or (i not in active and j not in active):
                    continue
                normal = validation.vcross(validation.vsub(first[1], first[0]),
                                           validation.vsub(first[2], first[0]))
                limit = validation.vlen(normal) * tolerance
                if all(abs(validation.vdot(normal, validation.vsub(p, first[0]))) <= limit
                       for p in second):
                    if validation.coplanar_overlap(first, second, tolerance):
                        raise ValueError(f"Overlapping neighbours: faces {i + 1} and {j + 1}")


def check_candidate(vertices, faces, active, options):
    report, tolerance = validation.analyse(vertices, faces, list(range(1,len(faces)+1)),
                                   options, active=active)
    folds = report["info"].get("patch_health", {}).get("fold_edges")
    if folds and folds > options.get("inherited_folds", 0):
        report["fail"]["folded_edges"] = folds
    # A surface that already passes through itself cannot be rebuilt without
    # doing the same. Such pairs are tolerated up to the number the source had.
    crossing = report["fail"].get("intersecting_faces")
    if crossing and crossing["count"] <= options.get("inherited_intersections", 0):
        report["info"]["inherited_intersections"] = report["fail"].pop("intersecting_faces")["count"]
    if report["fail"]:
        raise ValueError("Candidate rejected: " + json.dumps(report["fail"]))
    check_shared_edges(vertices, faces, active, tolerance)
    return report


def source_folds(vertices, faces, selected):
    """Number of sharply folded edges the selected source faces already have."""
    if not selected:
        return 0
    report, _ = validation.analyse(vertices, faces, list(range(1, len(faces) + 1)),
                                   {"intersect": False, "strict_seam": False}, active=set(selected))
    return report["info"].get("patch_health", {}).get("fold_edges", 0) or 0


def source_intersections(vertices, faces, selected):
    """Number of face pairs in the source that cross and involve the selection."""
    if not selected:
        return 0
    hits, _, _ = validation.find_intersecting_faces(vertices, faces, validation.bbox_scale(vertices) * 1e-6)
    return sum(a in selected or b in selected for a, b in hits)


def balance_boundary(vertices, selected, outside):
    """Resolve one odd seam with a single shared midpoint, if necessary."""
    owners = defaultdict(list)
    neighbours = defaultdict(set)
    for i, face in enumerate(selected):
        for a, b in directed(face):
            owners[edge(a, b)].append(i)
    for touching in owners.values():
        if len(touching) == 2:
            a, b = touching
            neighbours[a].add(b)
            neighbours[b].add(a)
    remaining = set(range(len(selected)))
    odd = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        component = [root]
        for i in component:
            added = sorted(neighbours[i] & remaining)
            component.extend(added)
            remaining.difference_update(added)
        if sum(len(selected[i]) for i in component) % 2:
            odd.append(set(component))
    if not odd:
        return vertices, selected, outside, None
    if len(odd) > 1:
        raise ValueError("More than one selected island needs an extra boundary vertex")
    external = defaultdict(list)
    for i, face in enumerate(outside):
        for a, b in directed(face):
            external[edge(a, b)].append(i)
    candidates = []
    for shared, internal in owners.items():
        if len(internal) != 1 or internal[0] not in odd[0]:
            continue
        near = external[shared]
        if len(near) > 1:
            continue
        outside_id = near[0] if near else None
        inside_id = internal[0]
        if outside_id is not None and len(outside[outside_id]) not in (3, 4):
            continue
        outside_cost = len(outside[outside_id]) - 2 if outside_id is not None else 0
        inside_cost = len(selected[inside_id]) - 3
        length = validation.vlen(validation.vsub(vertices[shared[0]], vertices[shared[1]]))
        candidates.append((outside_cost, inside_cost, length, shared, inside_id, outside_id))
    if not candidates:
        raise ValueError("No manifold boundary edge can accept the shared midpoint")
    _, _, _, shared, inside_id, outside_id = min(candidates)
    point_id = len(vertices)
    vertices = [*vertices, midpoint(vertices, *shared)]
    selected = selected[:inside_id] + split_triangles_on_edge(
        selected[inside_id], shared, point_id) + selected[inside_id + 1:]
    if outside_id is not None:
        outside = outside[:outside_id] + split_triangles_on_edge(
            outside[outside_id], shared, point_id) + outside[outside_id + 1:]
    return vertices, selected, outside, (shared, point_id, outside_id)


def recover(args):
    vertices, faces, lines = read_obj(args.before)
    other_vertices, other_faces, _ = read_obj(args.selection or args.scope_from)
    selected = identify(vertices, faces, other_vertices, other_faces, bool(args.selection))
    if not selected:
        raise ValueError("The selection must contain source faces")
    stray = stray_faces(faces, selected) if args.selection else set()
    if len(stray) > max(2, len(selected) // 50):
        raise ValueError("Too many loose faces in the selection to rebuild it safely")
    selected -= stray
    if not selected:
        raise ValueError("The selection contains only loose faces")
    if args.selection:
        selected = unpinch(faces, selected)
    patch_source = [faces[i] for i in sorted(selected)]
    preserved = [i for i in range(len(faces)) if i not in selected and i not in stray]
    unchanged = [faces[i] for i in preserved]
    seam = boundary(patch_source)
    if args.scope_from:
        changed = identify(other_vertices, other_faces, vertices, faces, False)
        old_border = seam_geometry(vertices, patch_source)
        new_border = seam_geometry(other_vertices, [other_faces[i] for i in sorted(changed)])
        if old_border != new_border:
            raise ValueError("The failed result does not establish an exact selection boundary")
    old_area = sum(validation.polygon_area(vertices, f) for f in patch_source)
    outside_area = sum(validation.polygon_area(vertices, f) for f in unchanged)
    if not old_area:
        raise ValueError("The selected surface must have nonzero area")
    change_ratio = max(len(selected) / len(faces), old_area / (old_area + outside_area))
    if change_ratio > args.max_changed_ratio:
        raise ValueError(f"Selection covers {change_ratio:.1%}; exceeds the locality limit")

    vertex_rows, groups = source_records(args.before, lines)
    outside_groups = [groups[lines[i]] for i in preserved]
    original_vertices = vertices
    work_faces = [triangle for face in patch_source for triangle in (
        polygon_triangles(vertices, face) if len(face) > 4 else [face])]
    vertices, work_faces, unchanged, adjustment = balance_boundary(vertices, work_faces, unchanged)
    seam = boundary(work_faces)
    if adjustment and adjustment[2] is not None:
        outside_id = adjustment[2]
        count = len(unchanged) - len(preserved) + 1
        outside_groups[outside_id:outside_id + 1] = [outside_groups[outside_id]] * count
    # Density in quads per area; a triangle counts as half a quad. This is the
    # same figure the Recommend button reports.
    def quad_count(group):
        return sum(0.5 if len(face) == 3 else 1.0 for face in group)
    outside_density = (quad_count(unchanged) / outside_area if outside_area
                       else quad_count(patch_source) / old_area)
    target = args.target_faces or max(1, round(old_area * outside_density))
    fixed = {i for a,b in seam for i in (a,b)} | {i for f in unchanged for i in f}

    def paired():
        """Pair the source triangles into quads, then adjust the density."""
        work_ids = set(range(len(work_faces)))
        base = vertices
        try:
            patch = quadrangulate(base, work_faces, work_ids)
        except ValueError:
            alternate = [face[1:] + face[:1] if len(face) == 4 else face for face in work_faces]
            try:
                patch = quadrangulate(base, alternate, work_ids)
            except ValueError:
                partial = quadrangulate(base, alternate, work_ids, partial=True)
                base, patch = parity_quads(base, partial)
        dense = base
        if len(patch) < target:
            dense, preferred = refine_quads(base, patch, fixed, target)
        elif len(patch) > target:
            preferred = coarsen(base, patch, fixed, target)
        else:
            preferred = patch
        if Counter(map(frozenset, preferred)) == Counter(map(frozenset, patch_source)):
            dense, preferred = refine_quads(base, patch, fixed, len(patch) + 4)
        found = [("paired", dense, preferred)]
        if preferred != patch:
            found.append(("paired-unadjusted", base, patch))
        return found

    def rebuilt():
        """Re-mesh the patch surface evenly; independent of the source layout."""
        from patch_remesh import remesh
        return [("rebuilt", *remesh(vertices, work_faces, target))]

    notes = []

    def regular():
        """Keep ZRemesher's regular block and join it to the mask border."""
        from core_patch import build
        native_vertices, native_faces, _ = read_obj(args.native)
        positions, quads, note = build(vertices, work_faces, seam, unchanged,
                                       native_vertices, native_faces, original_vertices, faces)
        notes.append(note)
        return [("regular", positions, quads)]

    # A clean quad source keeps its own edge flow through pairing. Triangle
    # soup has no flow worth keeping and is rebuilt instead.
    triangle_share = sum(len(f) == 3 for f in patch_source) / len(patch_source)
    builders = [rebuilt, paired] if triangle_share > 0.25 else [paired, rebuilt]
    if triangle_share > 0.25 and len(patch_source) > 3000:
        # Pairing a large triangle soup takes minutes and has not produced a
        # usable patch; the rebuild is the only method worth the wait there.
        builders = [rebuilt]

    if getattr(args, "style", "") == "grid" and getattr(args, "native", None):
        builders = [regular] + builders

    options = {"intersect": True, "strict_seam": True,
               "inherited_intersections": source_intersections(original_vertices, faces, selected | stray),
               "inherited_folds": source_folds(original_vertices, faces, selected | stray)}
    source_extreme = len(extreme_faces(original_vertices, patch_source))
    report = None
    reasons = []
    for builder in builders:
        try:
            candidates = builder()
        except (ValueError, IndexError, KeyError, ZeroDivisionError) as exc:
            reasons.append(f"{builder.__name__}: {exc}")
            continue
        for method, candidate_vertices, candidate in candidates:
          for attempt in range(4 if method in ("rebuilt", "regular") else 1):
              try:
                  # The sliver relaxation is slow; it is meant for a handful of faces.
                  if (method not in ("rebuilt", "regular")
                          and source_extreme < len(extreme_faces(candidate_vertices, candidate)) <= 24):
                      candidate_vertices = relax_slivers(candidate_vertices, candidate, fixed,
                                                         original_vertices, patch_source)
                  if boundary(candidate) != seam:
                      raise ValueError("Recovery changed the selected area's directed boundary")
                  if Counter(map(frozenset, candidate)) == Counter(map(frozenset, patch_source)):
                      raise ValueError("Recovery could not change this patch with its fixed boundary and requested density")
                  if len(extreme_faces(candidate_vertices, candidate)) > source_extreme:
                      raise ValueError("Recovery left extremely narrow faces")
                  patch_groups = inherit_patch_groups(
                      candidate_vertices, patch_source, candidate, [groups[lines[i]] for i in sorted(selected)], seam)
                  output_rows = [
                      vertex_rows[i] if i < len(vertex_rows) and point == original_vertices[i]
                      else "v " + " ".join(format(x, ".12g") for x in point)
                      for i, point in enumerate(candidate_vertices)]
                  obj, used, index = build_obj(output_rows, unchanged, candidate, outside_groups + patch_groups)
                  out_vertices = [candidate_vertices[i] for i in used]
                  out_faces = [tuple(index[i]-1 for i in face) for face in unchanged + candidate]
                  active = set(range(len(unchanged),len(out_faces)))
                  report = check_candidate(out_vertices, out_faces, active, options)
                  # Also test the alternate quad diagonals in the recovered region.
                  rotated = [f[1:] + f[:1] if i in active else f for i,f in enumerate(out_faces)]
                  check_candidate(out_vertices, rotated, active, options)
              except ValueError as exc:
                report = None
                flagged = sorted({int(n) - len(unchanged) for n in re.findall(r"[0-9]+", str(exc))
                                  if len(unchanged) <= int(n) < len(unchanged) + len(candidate)})
                if (method in ("rebuilt", "regular") and attempt < 3 and flagged
                        and str(exc).startswith("Candidate rejected") and "non_manifold" not in str(exc)):
                    # A few quads still touch or are flat: ease them and re-check.
                    from patch_remesh import ease_faces
                    candidate_vertices = ease_faces(candidate_vertices, candidate, flagged, fixed,
                                                    original_vertices, patch_source)
                    continue
                reasons.append(f"{method}: {exc}")
                break
              break
          if report is not None:
            break
        if report is not None:
            break
    if report is None:
        raise ValueError("; ".join(reasons)[:900])

    info = report["info"]
    patch_area = sum(validation.polygon_area(candidate_vertices, f) for f in candidate)
    density = len(candidate) / patch_area / outside_density
    whole_boundary_before = Counter(edge(a,b) for i,f in enumerate(faces) if i not in stray
                                    for a,b in directed(f))
    whole_boundary_after = Counter(edge(a,b) for f in unchanged + candidate for a,b in directed(f))
    original_open = {e for e,n in whole_boundary_before.items() if n == 1}
    if adjustment and adjustment[0] in original_open:
        (a, b), point_id, _ = adjustment
        original_open.remove(edge(a, b))
        original_open.update((edge(a, point_id), edge(b, point_id)))
    if original_open != {e for e,n in whole_boundary_after.items() if n == 1}:
        raise ValueError("Recovery changed the source mesh's open boundary")
    patch_info = info.get("patch", {})
    health = info.get("patch_health", {})
    diagnostics = [
        "LQR_STATUS:PASS",
        f"Source: {len(original_vertices)} vertices, {len(faces)} faces",
        f"Result: {len(out_vertices)} vertices, {len(out_faces)} faces",
        f"Selected patch: {len(selected) + len(stray)} source faces -> {len(candidate)} quads",
        f"Loose source faces removed: {len(stray)}",
        f"Source triangles in selection: {sum(len(f)==3 for f in patch_source)}",
        f"Exact untouched faces: {len(preserved) - int(bool(adjustment and adjustment[2] is not None))}",
        f"Patch boundary edges: {len(seam)}",
        f"Boundary adjustment: {1 if adjustment else 0} shared midpoint",
        f"Outside transition faces: {1 if adjustment and adjustment[2] is not None else 0}",
        f"Patch islands: {patch_info.get('islands', 0)}",
        f"Target patch faces: {target}",
        f"Patch-to-surrounding density ratio: {density:.3f}",
        f"Patch area change: {(patch_area/old_area-1)*100:+.3f}%",
        f"Source extremely narrow faces: {source_extreme}",
        f"Recovered extremely narrow faces: {len(extreme_faces(candidate_vertices, candidate))}",
        f"Method: {method}",
        *(notes if method == "regular" else []),
        f"Patch folded edges: {health.get('fold_edges', 0)}",
        f"Patch crowded vertices: {health.get('crowded_vertices', 0)}",
        "Strict patch checks: 0 non-quads, 0 degenerate or duplicate faces, 0 intersections",
        "Both patch triangulations checked; original mesh open boundaries preserved.",
        f"Intersection candidate pairs tested: {info['intersection_pairs_tested']}",
    ]
    if target != len(candidate):
        diagnostics.append("Target not reached with the current fixed-border quality constraints.")
    if options["inherited_intersections"]:
        diagnostics.append(
            f"The selected surface already crossed itself in {options['inherited_intersections']} face pairs; "
            f"{report['info'].get('inherited_intersections', 0)} crossing pairs remain in the new patch.")
    if reasons:
        diagnostics.append("Earlier attempts rejected: " + " | ".join(reasons)[:600])
    return obj, "\n".join(diagnostics) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--selection", type=Path, help="OBJ of the exact original selected faces")
    source.add_argument("--scope-from", type=Path, help="failed OBJ with an exact original seam")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-faces", type=int)
    parser.add_argument("--max-changed-ratio", type=float, default=1.0)
    args = parser.parse_args()
    if args.out.resolve() == args.report.resolve() or args.out.exists() or args.report.exists():
        parser.error("Use two different new paths for the output and report")
    if (args.target_faces is not None and args.target_faces < 1
            or not 0 < args.max_changed_ratio <= 1):
        parser.error("Invalid target face count or locality limit")
    try:
        obj, report = recover(args)
        args.out.write_text(obj, encoding="utf-8")
        args.report.write_text(report, encoding="ascii")
    except (OSError, ValueError) as exc:
        print(f"LQR_STATUS:FAIL\n{exc}", file=sys.stderr)
        return 1
    print(report, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())

