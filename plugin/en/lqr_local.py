"""Local mode for large SubTools.

ZBrush always exports the whole SubTool. On a mesh of millions of faces the
pure-Python checks would take minutes, so the work is cut down to the masked
area and a few rings of faces around it:

  phase 1  the full exports are read with NumPy, the masked faces and a band
           of rings around them are written to small OBJ files, the normal
           pipeline runs on those, and an accepted patch is written back into
           a full-size OBJ for ZBrush to import;
  phase 2  ZBrush's re-export of that OBJ is compared with it in full (NumPy),
           then the normal phase-2 checks run on the small area again.

Everything outside the band is compared exactly, face by face, so a change
outside the mask is still caught.
"""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import numpy as np

import lqr_mrgb
import lqr_validate as validation

RINGS = 4


class BigObj:
    """A Wavefront OBJ read with NumPy. Keeps the original lines."""

    def __init__(self, path):
        data = Path(path).read_bytes()
        # Polypaint and mask (#MRGB) are carried through; UVs, normals and
        # materials are not supported yet.
        self.has_extras = any(tag in data for tag in (b"\nvt ", b"\nvn ", b"\nusemtl ", b"\nmtllib "))
        if b"\r" in data:
            data = data.replace(b"\r", b"")
        self.lines = data.split(b"\n")
        vline, fline, gline, mline = [], [], [], []
        for i, line in enumerate(self.lines):
            head = line[:2]
            if head == b"v ":
                vline.append(i)
            elif head == b"f ":
                fline.append(i)
            elif head == b"g " or line == b"g":
                gline.append(i)
            elif head == b"#M" and line.startswith(b"#MRGB"):
                mline.append(i)
        self.vline = np.array(vline, dtype=np.int64)
        self.fline = np.array(fline, dtype=np.int64)
        self.gline = np.array(gline, dtype=np.int64)
        lines = self.lines
        coords = np.array(b" ".join(lines[i][2:] for i in vline).split(), dtype=np.float64)
        if coords.size != 3 * len(vline):
            raise ValueError(f"Unexpected vertex line format in {path}")
        self.v = coords.reshape(-1, 3)
        # Lines of the #MRGB block with ZBrush's comment line before and after it.
        self.mline = mline
        if mline:
            self.mline = [i for i in (mline[0] - 1, mline[-1] + 1)
                          if 0 <= i < len(lines) and lqr_mrgb.is_block_line(lines[i])] + mline
        # Polypaint: one value per vertex, or None. A block that does not match
        # the vertices is kept as an error and only refused where it is needed.
        self.mrgb, self.mrgb_error = None, None
        if mline:
            try:
                self.mrgb = lqr_mrgb.parse((lines[i] for i in mline), len(self.v))
            except ValueError as exc:
                self.mrgb_error = str(exc)
        text = [lines[i] for i in fline]
        counts = np.fromiter((len(t.split()) - 1 for t in text), dtype=np.int64, count=len(text))
        joined = b" ".join(t[2:] for t in text)
        if b"/" in joined:
            joined = b" ".join(token.split(b"/")[0] for token in joined.split())
        flat = np.array(joined.split(), dtype=np.int64) - 1
        if flat.size != counts.sum() or (counts < 3).any() or (flat < 0).any() or (flat >= len(self.v)).any():
            raise ValueError(f"Unexpected face line format in {path}")
        width = int(counts.max()) if len(counts) else 3
        self.f = np.full((len(counts), width), -1, dtype=np.int64)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        for k in range(3, width + 1):
            rows = np.nonzero(counts == k)[0]
            if len(rows):
                self.f[rows, :k] = flat[starts[rows][:, None] + np.arange(k)]
        self.counts = counts
        # Group of each face: the last "g" line before it.
        self.group_names = [lines[i] for i in gline]
        self.face_group = np.searchsorted(self.gline, self.fline) - 1

    def face_count(self):
        return len(self.f)


def canonical(f, counts):
    """Rotate every face so that its smallest vertex comes first (order kept)."""
    out = np.full_like(f, -1)
    for k in np.unique(counts):
        rows = np.nonzero(counts == k)[0]
        sub = f[rows, :k]
        start = np.argmin(sub, axis=1)
        idx = (start[:, None] + np.arange(k)) % k
        out[rows, :k] = np.take_along_axis(sub, idx, axis=1)
    return out


def sorted_rows(rows):
    if not len(rows):
        return rows
    order = np.lexsort(rows.T[::-1])
    return rows[order]


def touching(f, vertices):
    """Faces with at least one vertex in the sorted array `vertices`."""
    return np.isin(f, vertices).any(axis=1)


def _coordinate_hash(points):
    weights = np.array([0.7548776662466927, 0.5698402909980532, 0.3517509325486279])
    return np.asarray(points, dtype=np.float64) @ weights


def find_selection(big, sel_vertices, sel_faces):
    """Indices of the faces of `big` that are exactly the selection faces."""
    used = sorted({i for face in sel_faces for i in face})
    sv = np.asarray([sel_vertices[i] for i in used], dtype=np.float64)
    near = np.nonzero(np.isin(_coordinate_hash(big.v), _coordinate_hash(sv)))[0]
    rows = np.nonzero(np.isin(big.f, np.append(near, -1)).all(axis=1))[0]
    by_key = defaultdict(list)
    for r in rows:
        k = int(big.counts[r])
        key = tuple(big.v[i].tobytes() for i in big.f[r, :k])
        key = min(key[j:] + key[:j] for j in range(k))
        by_key[key].append(int(r))
    chosen = []
    for face in sel_faces:
        key = tuple(np.asarray(sel_vertices[i], dtype=np.float64).tobytes() for i in face)
        key = min(key[j:] + key[:j] for j in range(len(key)))
        hits = by_key.get(key)
        if not hits:
            raise ValueError("Selection contains faces that do not exactly match the source")
        chosen.append(hits.pop())
    return np.array(sorted(chosen), dtype=np.int64)


def grow(big, faces, rings):
    """Vertices within `rings` face rings of the given faces."""
    verts = np.unique(big.f[faces])
    verts = verts[verts >= 0]
    for _ in range(rings):
        mask = touching(big.f, verts)
        verts = np.unique(big.f[mask])
        verts = verts[verts >= 0]
    return verts


def write_local(path, big, faces):
    """Write the given faces of `big` with their own vertices and groups."""
    used = np.unique(big.f[faces])
    used = used[used >= 0]
    local = np.full(len(big.v), -1, dtype=np.int64)
    local[used] = np.arange(len(used))
    out = [b"# LocalQuadRetopo local extract"]
    out += [big.lines[big.vline[i]] for i in used]
    current = None
    for r in faces:
        g = int(big.face_group[r])
        if g != current:
            if g >= 0:
                out.append(big.group_names[g])
            current = g
        k = int(big.counts[r])
        out.append(b"f " + b" ".join(str(int(local[i]) + 1).encode() for i in big.f[r, :k]))
    Path(path).write_bytes(b"\n".join(out) + b"\n")
    return used


def map_vertices(source, target, tolerance):
    """For each target vertex the source vertex at the same place, or -1."""
    from scipy.spatial import cKDTree
    distance, index = cKDTree(source).query(target, distance_upper_bound=tolerance if tolerance > 0 else 1e-300)
    index = np.where(np.isfinite(distance) & (distance <= tolerance), index, -1)
    return index


def same_faces(a_rows, a_counts, b_rows, b_counts):
    """True if the two face lists (in the same vertex numbering) are equal as multisets."""
    if len(a_rows) != len(b_rows):
        return False
    width = max(a_rows.shape[1] if len(a_rows) else 0, b_rows.shape[1] if len(b_rows) else 0, 3)

    def prep(rows, counts):
        if rows.shape[1] < width:
            rows = np.hstack([rows, np.full((len(rows), width - rows.shape[1]), -1, dtype=rows.dtype)])
        return sorted_rows(canonical(rows, counts))
    return bool(np.array_equal(prep(a_rows, a_counts), prep(b_rows, b_counts)))


def state_path(mod):
    return Path(mod.PREFIX + "local_state.npz")


def run(args, mod):
    """Local-mode replacement for mod.run(args)."""
    try:
        return _run(args, mod)
    except ValueError:
        raise
    except Exception as exc:          # MemoryError, NumPy errors and the like
        raise ValueError(f"Local mode failed: {type(exc).__name__}: {exc}") from exc


def _run(args, mod):
    prefix = mod.PREFIX
    local_prefix = prefix + "L_"
    full = dict(before=mod.BEFORE, after=mod.AFTER, selection=mod.SELECTION,
                recovered=mod.RECOVERED, roundtrip=mod.ROUNDTRIP, report=mod.REPORT)
    if args.mode == 3:
        return recommend(args, mod, full, local_prefix)
    if args.phase == 2:
        return phase2(args, mod, full, local_prefix)
    return phase1(args, mod, full, local_prefix)


def run_local(args, mod, local_prefix, ratio=1.0):
    """Run the normal pipeline on the local files and return its report text."""
    saved = mod.PREFIX
    mod.use_prefix(local_prefix)
    try:
        local_args = type(args)(**vars(args))
        local_args.ratio = ratio
        mod.run(local_args)
        return mod.REPORT.read_text(encoding="ascii") if mod.REPORT.exists() else \
            "LQR_STATUS:FAIL\n[FAIL] The local run produced no report.\n"
    finally:
        mod.use_prefix(saved)


def context_of(big, selected):
    verts = grow(big, selected, RINGS)
    return np.nonzero(touching(big.f, verts))[0], verts


def recommend(args, mod, full, local_prefix):
    big = BigObj(full["before"])
    sv, sf, _ = validation.parse_obj(full["after"])
    selected = find_selection(big, sv, sf)
    context, _ = context_of(big, selected)
    write_local(local_prefix + "before.obj", big, context)
    Path(local_prefix + "after.obj").write_bytes(Path(full["after"]).read_bytes())
    text = run_local(args, mod, local_prefix)
    mod.publish(full["report"], text)


def phase1(args, mod, full, local_prefix):
    big = BigObj(full["before"])
    sv, sf, _ = validation.parse_obj(full["selection"])
    selected = find_selection(big, sv, sf)
    share = len(selected) / max(len(big.f), 1)
    if share > args.ratio:
        raise ValueError(f"Selection covers {share:.1%}; exceeds the locality limit")
    context, region = context_of(big, selected)
    Path(local_prefix + "selection.obj").write_bytes(Path(full["selection"]).read_bytes())
    used = write_local(local_prefix + "before.obj", big, context)

    native_note = ""
    after_bytes = Path(full["after"]).read_bytes()
    before_bytes = Path(full["before"]).read_bytes()
    if after_bytes == before_bytes:
        Path(local_prefix + "after.obj").write_bytes(Path(local_prefix + "before.obj").read_bytes())
    else:
        after = BigObj(full["after"])
        lo, hi = big.v.min(axis=0), big.v.max(axis=0)
        tolerance = float(np.linalg.norm(hi - lo)) * 1e-6
        to_before = map_vertices(big.v, after.v, tolerance)
        mapped = np.where(after.f >= 0, to_before[np.maximum(after.f, 0)], -1)
        new_vertex = (after.f >= 0) & (mapped < 0)
        inside = new_vertex.any(axis=1) | np.isin(mapped, region).any(axis=1)
        outside_after = np.nonzero(~inside)[0]
        outside_before = np.nonzero(~touching(big.f, region))[0]
        if same_faces(mapped[outside_after], after.counts[outside_after],
                      big.f[outside_before], big.counts[outside_before]):
            write_local(local_prefix + "after.obj", after, np.nonzero(inside)[0])
        else:
            # ZRemesher touched faces outside the band. Its result is not
            # usable; the patch is rebuilt from the source alone, as the
            # full-size pipeline does in the same case.
            native_note = "ZRemesher changed faces outside the mask; its result was not used.\n"
            Path(local_prefix + "after.obj").write_bytes(Path(local_prefix + "before.obj").read_bytes())

    text = run_local(args, mod, local_prefix)
    note = f"Local mode: {len(context)} of {len(big.f)} faces checked around the mask.\n" + native_note
    if text.startswith("LQR_STATUS:IMPORT"):
        if big.has_extras:
            raise ValueError("This recovery build supports geometry, OBJ groups and polypaint only "
                             "(the SubTool has UVs, normals or materials)")
        if big.mrgb_error:
            raise ValueError(big.mrgb_error)
        expand(big, used, context, local_prefix, full, mod)
    lines = text.splitlines()
    mod.publish(full["report"], "\n".join(lines[:2] + [note.rstrip()] + lines[2:]) + "\n")


def parse_local(path):
    """Small OBJ: vertices (as floats and text), faces, group line per face."""
    vtext, faces, groups = [], [], []
    group = None
    for raw in Path(path).read_bytes().replace(b"\r", b"").split(b"\n"):
        if raw[:2] == b"v ":
            vtext.append(raw)
        elif raw[:2] == b"g " or raw == b"g":
            group = raw
        elif raw[:2] == b"f ":
            faces.append(tuple(int(t.split(b"/")[0]) - 1 for t in raw.split()[1:]))
            groups.append(group)
    coords = [tuple(float(x) for x in t.split()[1:4]) for t in vtext]
    return vtext, coords, faces, groups


def expand(big, used, context, local_prefix, full, mod):
    """Write the full-size recovered OBJ: the source with the patch swapped in."""
    n_old = len(big.v)
    vstart = int(big.vline[0]) if n_old else 0
    if n_old and not np.array_equal(big.vline, np.arange(vstart, vstart + n_old)):
        raise ValueError("The vertex lines of the export are not in one block")
    _, lb_coords, lb_faces, _ = parse_local(local_prefix + "before.obj")
    lr_text, lr_coords, lr_faces, lr_groups = parse_local(local_prefix + "recovered.obj")
    where = {}
    for i, p in enumerate(lb_coords):
        where.setdefault(p, int(used[i]))

    def key(coords, face):
        k = tuple(coords[i] for i in face)
        return min(k[j:] + k[:j] for j in range(len(k)))

    lb_keys = [key(lb_coords, f) for f in lb_faces]
    lr_keys = [key(lr_coords, f) for f in lr_faces]
    if len(context) != len(lb_faces):
        raise ValueError("Local extract does not line up with the source mesh")
    kept = Counter(lr_keys)
    removed = []
    for row, k in zip(context, lb_keys):
        if kept[k] > 0:
            kept[k] -= 1
        else:
            removed.append(int(row))
    old_keys = Counter(lb_keys)
    added = []
    for face, k, g in zip(lr_faces, lr_keys, lr_groups):
        if old_keys[k] > 0:
            old_keys[k] -= 1
        else:
            added.append((face, g))
    removed = np.array(sorted(removed), dtype=np.int64)
    keep_face = np.ones(len(big.f), dtype=bool)
    keep_face[removed] = False

    # Full index of each local recovered vertex; new vertices get one below.
    lr_full = np.array([where.get(p, -1) for p in lr_coords], dtype=np.int64)
    needed = sorted({i for face, _ in added for i in face if lr_full[i] < 0})
    candidates = np.unique(big.f[removed]) if len(removed) else np.zeros(0, dtype=np.int64)
    candidates = candidates[candidates >= 0]
    near = np.nonzero(keep_face & np.isin(big.f, candidates).any(axis=1))[0]
    still = set(np.unique(big.f[near]).tolist())
    still |= {int(lr_full[i]) for face, _ in added for i in face if lr_full[i] >= 0}
    slots = sorted(set(candidates.tolist()) - still)
    text_of = {}
    point_of = {}
    for j, i in enumerate(needed):
        ident = slots[j] if j < len(slots) else n_old + (j - len(slots))
        lr_full[i] = ident
        text_of[ident] = lr_text[i]
        point_of[ident] = lr_coords[i]
    size = n_old + max(0, len(needed) - len(slots))
    dead = sorted(slots[len(needed):])
    # Keep the vertex list dense: move the last live vertices into dead slots.
    dead_set = set(dead)
    moved = {}
    end = size - 1
    for d in dead:
        while end in dead_set and end > d:
            end -= 1
        if end <= d:
            break
        moved[end] = d
        end -= 1
    total = size - len(dead)
    remap = np.arange(size, dtype=np.int64)
    for old, new in moved.items():
        remap[old] = new
    source_at = list(range(total))
    for old, new in moved.items():
        source_at[new] = old

    def vertex_text(ident):
        if ident in text_of:
            return text_of[ident]
        return big.lines[vstart + ident]

    vertex_block = [vertex_text(ident) for ident in source_at]
    if any(t is None for t in vertex_block) or len(set(remap[i] for i in range(size) if i not in dead_set)) != total:
        raise ValueError("Vertex bookkeeping failed while writing the full result")

    # Polypaint: kept vertices keep their value; a new vertex (also one that
    # reuses a freed slot) takes the value of the nearest source vertex of
    # the area that was rebuilt.
    colour_block = []
    if big.mrgb is not None:
        order = np.asarray(source_at, dtype=np.int64)
        values = np.empty(total, dtype="S8")
        fresh = np.fromiter((int(i) in text_of for i in order), dtype=bool, count=total)
        values[~fresh] = big.mrgb[order[~fresh]]
        if fresh.any():
            near = np.asarray(used, dtype=np.int64)
            values[fresh] = lqr_mrgb.nearest(big.v[near], big.mrgb[near],
                                             [point_of[int(i)] for i in order[fresh]])
        colour_block = lqr_mrgb.block(values)

    out = list(big.lines)
    for row in removed:
        out[big.fline[row]] = None
    for i in big.mline:
        out[i] = None
    if moved:
        olds = np.array(sorted(moved), dtype=np.int64)
        olds = olds[olds < n_old]
        for r in np.nonzero(keep_face & np.isin(big.f, olds).any(axis=1))[0]:
            k = int(big.counts[r])
            out[big.fline[r]] = b"f " + b" ".join(str(int(remap[i]) + 1).encode() for i in big.f[r, :k])
    tail, current = [], None
    for face, g in added:
        if g is not None and g != current:
            tail.append(g)
            current = g
        tail.append(b"f " + b" ".join(str(int(remap[lr_full[i]]) + 1).encode() for i in face))
    head = out[:vstart]
    rest = out[vstart + n_old:]
    data = b"\n".join(line for line in head + vertex_block + colour_block + rest + tail
                      if line is not None) + b"\n"
    _write(full["recovered"], data)
    np.savez(local_prefix + "state.npz", local_full=remap[lr_full], vertex_count=np.array([total]))


def _write(path, data):
    pending = Path(str(path) + ".pending")
    pending.write_bytes(data)
    os.replace(pending, path)


def phase2(args, mod, full, local_prefix):
    state = np.load(local_prefix + "state.npz")
    total = int(state["vertex_count"][0])
    recovered = BigObj(full["recovered"])
    roundtrip = BigObj(full["roundtrip"])
    if len(recovered.v) != total or len(roundtrip.v) != total:
        raise ValueError("ZBrush round trip altered the vertex count")
    tolerance = validation.bbox_scale([tuple(recovered.v.min(axis=0)), tuple(recovered.v.max(axis=0))]) * 1e-6
    to_recovered = map_vertices(recovered.v, roundtrip.v, tolerance)
    if (to_recovered < 0).any() or len(np.unique(to_recovered)) != total:
        raise ValueError("ZBrush round trip shifted or merged vertices")
    mapped = np.where(roundtrip.f >= 0, to_recovered[np.maximum(roundtrip.f, 0)], -1)
    if not same_faces(mapped, roundtrip.counts, recovered.f, recovered.counts):
        raise ValueError("ZBrush round trip altered winding or face layout")
    # Local round trip: the local recovered file with ZBrush's coordinates.
    inverse = np.empty(total, dtype=np.int64)
    inverse[to_recovered] = np.arange(total)
    local_full = state["local_full"]
    raw = Path(local_prefix + "recovered.obj").read_bytes().replace(b"\r", b"").split(b"\n")
    out, n = [], 0
    for line in raw:
        if line[:2] == b"v ":
            j = inverse[local_full[n]]
            out.append(roundtrip.lines[roundtrip.vline[j]])
            n += 1
        else:
            out.append(line)
    Path(local_prefix + "roundtrip.obj").write_bytes(b"\n".join(out))
    text = run_local(args, mod, local_prefix)
    note = lqr_mrgb.roundtrip_note(recovered.mrgb, roundtrip.mrgb, to_recovered)
    if note and text.startswith("LQR_STATUS:PASS"):
        lines = text.splitlines()
        text = "\n".join(lines[:2] + [note] + lines[2:]) + "\n"
    mod.publish(full["report"], text)
